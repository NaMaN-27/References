// lob/book.hpp — the order book.
//
// Structure
// ---------
//     price ladder (flat array, one Level per tick)
//        │
//        └─ Level { qty, count, head, tail }        16 bytes, 4 per cache line
//              │
//              └─ intrusive doubly-linked list of Order, FIFO
//                    │
//                    └─ Order { next, prev, qty, level|side }   16 bytes
//                          (lives in a preallocated Pool, addressed by uint32)
//
//     order_ref (uint64) ──► OrderMap ──► pool index (uint32)
//
//     occupancy bitmap ──► best bid / best ask in O(1)
//
// Complexity, and why each one is what it is:
//
//   add      O(1)  — hash insert, pool alloc, tail append, bitmap set
//   cancel   O(1)  — hash lookup, qty decrement
//   delete   O(1)  — hash lookup + erase, unlink, pool free, maybe bitmap reset
//   execute  O(1)  — hash lookup, qty decrement, remove if filled
//   best     O(1)  — cached; recomputed via bitmap only when a level empties
//   top-K    O(K)  — K bitmap scans
//
// Why a flat array instead of std::map<price, Level>
// --------------------------------------------------
// std::map is a red-black tree: every node separately allocated, ~48 bytes of
// overhead per node, and a lookup is 3-5 dependent pointer chases through
// memory the allocator scattered. The comparison at each node is cheap; the
// cache miss before it is not. Measured in bench_main at roughly 4x slower on
// the same message stream.
//
// The flat array looks profligate — 65536 levels × 16 bytes = 1 MB per side,
// for a book that is typically 50 levels deep. It is not, and the reason is
// worth understanding: the allocation is virtual. Untouched pages are never
// faulted in, never occupy physical memory, and never enter the TLB. The
// resident set is the pages actually touched, which is the few hundred ticks
// around the touch — a handful of 4 KB pages. You are trading address space,
// which is free on 64-bit, for pointer chasing, which is not.
//
// The real cost of the flat ladder is the bounded price range. That is handled
// explicitly rather than hidden: prices outside the window are counted and
// reported, and a production system would rebase the ladder (or keep a
// secondary map) rather than pretend it cannot happen.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

#include "lob/bitmap.hpp"
#include "lob/itch.hpp"
#include "lob/pool.hpp"

#if defined(__linux__)
#  include <sys/mman.h>
#endif

namespace lob {

struct Order {
    std::uint32_t next;        // intrusive list next / free-list link
    std::uint32_t prev;        // intrusive list prev
    std::uint32_t qty;         // remaining displayed shares
    std::uint32_t level_side;  // bit31 = side (1 == ask), bits 0..30 = ladder index
};
static_assert(sizeof(Order) == 16, "4 orders per cache line");

struct Level {
    std::uint32_t qty;    // total shares resting at this price
    std::uint32_t count;  // number of resting orders
    std::uint32_t head;   // pool index of the oldest order (front of the FIFO)
    std::uint32_t tail;   // pool index of the newest order
};
static_assert(sizeof(Level) == 16, "4 levels per cache line");

struct TopOfBook {
    std::uint32_t bid_px{0}, bid_qty{0};
    std::uint32_t ask_px{0}, ask_qty{0};

    bool operator==(const TopOfBook& o) const noexcept {
        return bid_px == o.bid_px && bid_qty == o.bid_qty &&
               ask_px == o.ask_px && ask_qty == o.ask_qty;
    }
};

struct BookLevel {
    std::uint32_t price{0};
    std::uint32_t qty{0};
    std::uint32_t count{0};
};

struct BookStats {
    std::uint64_t adds{0}, cancels{0}, deletes{0}, executes{0}, replaces{0};
    std::uint64_t exec_shares{0};
    std::uint64_t unknown_ref{0};        // message for an order we never saw
    std::uint64_t price_out_of_range{0}; // outside the ladder window
    std::uint64_t pool_exhausted{0};
    std::uint64_t map_overflows{0};     // order map hit its load ceiling
    std::uint64_t tob_changes{0};        // top-of-book transitions
    std::uint64_t best_recomputes{0};    // bitmap scans triggered
    std::uint64_t crossed_observations{0};
    std::uint64_t scan_steps{0};        // ladder steps walked by the non-bitmap variant
    std::uint64_t max_scan_steps{0};    // worst single walk — the tail that matters
};

// UseBitmap=false swaps the O(1) bitmap best-price search for the naive
// linear walk down the ladder. Everything else is identical, including the
// bitmap maintenance itself — so the measured difference is a LOWER BOUND on
// what the bitmap buys, since the scan variant still pays for keeping it up
// to date. An honest A/B is worth more than a flattering one.
template <std::size_t LevelsLog2 = 16, bool UseBitmap = true>
class Book {
public:
    using Bmp = HierBitmap<LevelsLog2>;
    static constexpr std::size_t kLevels = std::size_t{1} << LevelsLog2;
    // Sentinel bridging: best_bid_/best_ask_ are uint32 and use kNoLevel;
    // the bitmap returns size_t and uses kNone. Convert explicitly at the
    // boundary rather than relying on integer promotion happening to work.
    static constexpr std::size_t to_scan(std::uint32_t v) noexcept {
        return v == kNoLevel ? Bmp::kNone : static_cast<std::size_t>(v);
    }
    static constexpr std::uint32_t kAskBit = 0x80000000u;
    static constexpr std::uint32_t kNoLevel = 0xFFFFFFFFu;

    // base_price and tick are in ITCH price units (1/10000 dollar).
    // tick = 100 is one cent, which is the quoting increment for any US equity
    // priced at or above $1.00 under Reg NMS Rule 612.
    // Capacity defaults matter far more than they look. See bench experiment
    // G: sizing the order map 130x larger than needed costs 3x throughput,
    // because the hash spreads lookups uniformly over the WHOLE allocation, so
    // the allocation size *is* the working set. The pool is different — it is
    // accessed densely from slot 0 upward, so an oversized pool wastes address
    // space but costs almost no time. Size the map tightly; the pool can be
    // generous.
    explicit Book(std::size_t order_capacity = 1u << 18,
                  std::size_t map_capacity_log2 = 18,
                  std::uint32_t base_price = 0,
                  std::uint32_t tick = 100)
        : pool_(order_capacity),
          map_(map_capacity_log2),
          base_(base_price),
          tick_(tick) {
        const std::size_t bytes = kLevels * sizeof(Level);
        bid_ = alloc_ladder(bytes);
        ask_ = alloc_ladder(bytes);
    }

    ~Book() {
        ::operator delete(bid_, std::align_val_t{64});
        ::operator delete(ask_, std::align_val_t{64});
    }
    Book(const Book&) = delete;
    Book& operator=(const Book&) = delete;

    void reset() {
        std::memset(static_cast<void*>(bid_), 0, kLevels * sizeof(Level));
        std::memset(static_cast<void*>(ask_), 0, kLevels * sizeof(Level));
        bid_map_.clear();
        ask_map_.clear();
        pool_.reset();
        map_.clear();
        best_bid_ = kNoLevel;
        best_ask_ = kNoLevel;
        stats_ = BookStats{};
    }

    // ---------------------------------------------------------------------
    // message handlers
    // ---------------------------------------------------------------------

    [[gnu::always_inline]] inline void add(std::uint64_t ref, bool is_buy,
                                           std::uint32_t shares,
                                           std::uint32_t price) noexcept {
        ++stats_.adds;
        const std::uint32_t idx = to_index(price);
        if (idx == kNoLevel) [[unlikely]] { ++stats_.price_out_of_range; return; }

        const std::uint32_t oi = pool_.alloc();
        if (oi == kNullIdx) [[unlikely]] { ++stats_.pool_exhausted; return; }

        Order& o = pool_[oi];
        o.qty = shares;
        o.level_side = idx | (is_buy ? 0u : kAskBit);
        o.next = kNullIdx;

        Level* lv = is_buy ? &bid_[idx] : &ask_[idx];
        // FIFO append preserves price-time priority, which is the entire point
        // of keeping a list rather than just an aggregate quantity. If you only
        // ever need L2 depth you can drop the list and keep qty/count — but
        // then you cannot answer "where am I in the queue", which is the
        // question a passive strategy actually cares about.
        o.prev = lv->tail;
        if (lv->tail != kNullIdx) pool_[lv->tail].next = oi;
        else                      lv->head = oi;
        lv->tail = oi;

        const bool was_empty = (lv->count == 0);
        lv->qty += shares;
        ++lv->count;

        if (was_empty) {
            if (is_buy) { bid_map_.set(idx); if (best_bid_ == kNoLevel || idx > best_bid_) best_bid_ = idx; }
            else        { ask_map_.set(idx); if (best_ask_ == kNoLevel || idx < best_ask_) best_ask_ = idx; }
        } else {
            // Level already occupied, so the best cannot have improved past it.
            if (is_buy) { if (best_bid_ == kNoLevel || idx > best_bid_) best_bid_ = idx; }
            else        { if (best_ask_ == kNoLevel || idx < best_ask_) best_ask_ = idx; }
        }

        map_.insert(ref, oi);
        stats_.map_overflows = map_.overflows();
    }

    // Partial cancel: reduce displayed size, order stays in the book.
    [[gnu::always_inline]] inline void cancel(std::uint64_t ref,
                                              std::uint32_t shares) noexcept {
        ++stats_.cancels;
        const std::uint32_t oi = map_.find(ref);
        if (oi == kNullIdx) [[unlikely]] { ++stats_.unknown_ref; return; }
        reduce(oi, ref, shares);
    }

    // Full delete.
    [[gnu::always_inline]] inline void remove(std::uint64_t ref) noexcept {
        ++stats_.deletes;
        const std::uint32_t oi = map_.find(ref);
        if (oi == kNullIdx) [[unlikely]] { ++stats_.unknown_ref; return; }
        unlink_and_free(oi, ref);
    }

    // Execution against a resting order.
    [[gnu::always_inline]] inline void execute(std::uint64_t ref,
                                               std::uint32_t shares) noexcept {
        ++stats_.executes;
        stats_.exec_shares += shares;
        const std::uint32_t oi = map_.find(ref);
        if (oi == kNullIdx) [[unlikely]] { ++stats_.unknown_ref; return; }
        reduce(oi, ref, shares);
    }

    // Replace: cancel the old reference entirely, add a new one. The new order
    // goes to the BACK of the queue at its price — replace loses time priority,
    // which is why a strategy that wants to keep its queue position modifies
    // size downward with a cancel instead of a replace.
    [[gnu::always_inline]] inline void replace(std::uint64_t old_ref,
                                               std::uint64_t new_ref,
                                               std::uint32_t shares,
                                               std::uint32_t price) noexcept {
        ++stats_.replaces;
        const std::uint32_t oi = map_.find(old_ref);
        if (oi == kNullIdx) [[unlikely]] { ++stats_.unknown_ref; return; }
        const bool is_buy = (pool_[oi].level_side & kAskBit) == 0;
        unlink_and_free(oi, old_ref);
        // `add` bumps stats_.adds; undo that so the counters attribute this
        // message to `replaces` alone and the totals still reconcile.
        add(new_ref, is_buy, shares, price);
        --stats_.adds;
    }

    // ---------------------------------------------------------------------
    // queries
    // ---------------------------------------------------------------------

    [[gnu::always_inline]] inline TopOfBook top() const noexcept {
        TopOfBook t;
        if (best_bid_ != kNoLevel) { t.bid_px = to_price(best_bid_); t.bid_qty = bid_[best_bid_].qty; }
        if (best_ask_ != kNoLevel) { t.ask_px = to_price(best_ask_); t.ask_qty = ask_[best_ask_].qty; }
        return t;
    }

    bool has_bid() const noexcept { return best_bid_ != kNoLevel; }
    bool has_ask() const noexcept { return best_ask_ != kNoLevel; }

    // L2 aggregated depth, best first.
    void depth(bool is_buy, std::size_t k, std::vector<BookLevel>& out) const {
        out.clear();
        if (is_buy) {
            std::size_t i = to_scan(best_bid_);
            while (out.size() < k && i != Bmp::kNone) {
                out.push_back({to_price(static_cast<std::uint32_t>(i)), bid_[i].qty, bid_[i].count});
                if (i == 0) break;
                i = bid_map_.highest_at_or_below(i - 1);
            }
        } else {
            std::size_t i = to_scan(best_ask_);
            while (out.size() < k && i != Bmp::kNone) {
                out.push_back({to_price(static_cast<std::uint32_t>(i)), ask_[i].qty, ask_[i].count});
                if (i + 1 >= kLevels) break;
                i = ask_map_.lowest_at_or_above(i + 1);
            }
        }
    }

    // Queue position of an order at its price level, in shares ahead of it.
    // This is the number a passive strategy actually wants and the reason the
    // per-level FIFO list exists at all.
    std::uint64_t shares_ahead(std::uint64_t ref) const noexcept {
        const std::uint32_t oi = map_.find(ref);
        if (oi == kNullIdx) return ~std::uint64_t{0};
        const Order& o = pool_[oi];
        const std::uint32_t idx = o.level_side & ~kAskBit;
        const Level* lv = (o.level_side & kAskBit) ? &ask_[idx] : &bid_[idx];
        std::uint64_t ahead = 0;
        for (std::uint32_t i = lv->head; i != kNullIdx && i != oi; i = pool_[i].next)
            ahead += pool_[i].qty;
        return ahead;
    }

    // FNV-1a over the full occupied book. Used to prove that two replays of
    // the same input produce the same state, and that the fast book agrees
    // with the reference book.
    std::uint64_t checksum() const noexcept {
        std::uint64_t h = 1469598103934665603ULL;
        auto mix = [&h](std::uint64_t v) {
            for (int b = 0; b < 8; ++b) {
                h ^= (v >> (b * 8)) & 0xFF;
                h *= 1099511628211ULL;
            }
        };
        std::size_t i = to_scan(best_bid_);
        while (i != Bmp::kNone) {
            mix(to_price(static_cast<std::uint32_t>(i)));
            mix(bid_[i].qty);
            mix(bid_[i].count);
            if (i == 0) break;
            i = bid_map_.highest_at_or_below(i - 1);
        }
        mix(0xFFFFFFFFFFFFFFFFULL);   // side separator
        i = to_scan(best_ask_);
        while (i != Bmp::kNone) {
            mix(to_price(static_cast<std::uint32_t>(i)));
            mix(ask_[i].qty);
            mix(ask_[i].count);
            if (i + 1 >= kLevels) break;
            i = ask_map_.lowest_at_or_above(i + 1);
        }
        return h;
    }

    const BookStats& stats()      const noexcept { return stats_; }
    BookStats&       stats_mut()        noexcept { return stats_; }
    const OrderMap&  order_map()  const noexcept { return map_; }
    std::size_t      live_orders() const noexcept { return pool_.live(); }
    std::size_t      pool_high_water() const noexcept { return pool_.high_water(); }
    std::size_t      bid_levels() const noexcept { return bid_map_.popcount(); }
    std::size_t      ask_levels() const noexcept { return ask_map_.popcount(); }

    [[gnu::always_inline]] inline std::uint32_t to_price(std::uint32_t idx) const noexcept {
        return base_ + idx * tick_;
    }

private:
    static Level* alloc_ladder(std::size_t bytes) {
        Level* p = static_cast<Level*>(::operator new(bytes, std::align_val_t{64}));
        std::memset(static_cast<void*>(p), 0, bytes);
#if defined(__linux__) && defined(MADV_HUGEPAGE)
        ::madvise(p, bytes, MADV_HUGEPAGE);
#endif
        return p;
    }

    [[gnu::always_inline]] inline std::uint32_t to_index(std::uint32_t price) const noexcept {
        if (price < base_) [[unlikely]] return kNoLevel;
        const std::uint32_t d = price - base_;
        // Sub-tick prices (legal below $1.00, and for midpoint executions)
        // cannot be represented on a penny ladder. Reject explicitly rather
        // than silently rounding — a book that quietly rounds is a book that
        // quietly disagrees with the exchange.
        if (d % tick_ != 0) [[unlikely]] return kNoLevel;
        const std::uint32_t i = d / tick_;
        return i < kLevels ? i : kNoLevel;
    }

    [[gnu::always_inline]] inline void reduce(std::uint32_t oi, std::uint64_t ref,
                                              std::uint32_t shares) noexcept {
        Order& o = pool_[oi];
        const std::uint32_t idx = o.level_side & ~kAskBit;
        const bool is_ask = (o.level_side & kAskBit) != 0;
        Level* lv = is_ask ? &ask_[idx] : &bid_[idx];

        // Defensive clamp. A well-formed feed never over-executes an order,
        // but a gap in the stream (a dropped multicast packet, a bad replay
        // start point) produces exactly this, and underflowing a uint32 turns
        // a small inconsistency into a book with 4 billion shares on the bid.
        const std::uint32_t take = shares < o.qty ? shares : o.qty;
        o.qty -= take;
        lv->qty -= take;

        if (o.qty == 0) unlink_and_free_known(oi, ref, idx, is_ask, lv);
    }

    [[gnu::always_inline]] inline void unlink_and_free(std::uint32_t oi,
                                                       std::uint64_t ref) noexcept {
        Order& o = pool_[oi];
        const std::uint32_t idx = o.level_side & ~kAskBit;
        const bool is_ask = (o.level_side & kAskBit) != 0;
        Level* lv = is_ask ? &ask_[idx] : &bid_[idx];
        lv->qty -= o.qty;
        unlink_and_free_known(oi, ref, idx, is_ask, lv);
    }

    [[gnu::always_inline]] inline void unlink_and_free_known(std::uint32_t oi,
                                                              std::uint64_t ref,
                                                              std::uint32_t idx,
                                                              bool is_ask,
                                                              Level* lv) noexcept {
        Order& o = pool_[oi];
        if (o.prev != kNullIdx) pool_[o.prev].next = o.next; else lv->head = o.next;
        if (o.next != kNullIdx) pool_[o.next].prev = o.prev; else lv->tail = o.prev;
        --lv->count;

        map_.erase(ref);
        pool_.free(oi);

        if (lv->count == 0) {
            lv->qty = 0;   // guard against accumulated rounding from clamps
            if (is_ask) {
                ask_map_.reset(idx);
                if (idx == best_ask_) {
                    ++stats_.best_recomputes;
                    std::size_t n = Bmp::kNone;
                    if constexpr (UseBitmap) {
                        if (idx + 1 < kLevels) n = ask_map_.lowest_at_or_above(idx + 1);
                    } else {
                        std::uint64_t steps = 0;
                        for (std::size_t j = idx + 1; j < kLevels; ++j) {
                            ++steps;
                            if (ask_[j].count) { n = j; break; }
                        }
                        stats_.scan_steps += steps;
                        if (steps > stats_.max_scan_steps) stats_.max_scan_steps = steps;
                    }
                    best_ask_ = (n == Bmp::kNone)
                              ? kNoLevel : static_cast<std::uint32_t>(n);
                }
            } else {
                bid_map_.reset(idx);
                if (idx == best_bid_) {
                    ++stats_.best_recomputes;
                    std::size_t n = Bmp::kNone;
                    if constexpr (UseBitmap) {
                        if (idx > 0) n = bid_map_.highest_at_or_below(idx - 1);
                    } else {
                        std::uint64_t steps = 0;
                        for (std::size_t j = idx; j-- > 0;) {
                            ++steps;
                            if (bid_[j].count) { n = j; break; }
                        }
                        stats_.scan_steps += steps;
                        if (steps > stats_.max_scan_steps) stats_.max_scan_steps = steps;
                    }
                    best_bid_ = (n == Bmp::kNone)
                              ? kNoLevel : static_cast<std::uint32_t>(n);
                }
            }
        }
    }

    Level*                    bid_{nullptr};
    Level*                    ask_{nullptr};
    HierBitmap<LevelsLog2>    bid_map_{};
    HierBitmap<LevelsLog2>    ask_map_{};
    Pool<Order>               pool_;
    OrderMap                  map_;
    std::uint32_t             best_bid_{kNoLevel};
    std::uint32_t             best_ask_{kNoLevel};
    std::uint32_t             base_{0};
    std::uint32_t             tick_{100};
    BookStats                 stats_{};
};

}  // namespace lob

// lob/pool.hpp — fixed-capacity object pool + open-addressing order map.
//
// Two rules the hot path must obey:
//   1. No malloc. Ever. Not `new`, not `push_back` that might grow, not a
//      std::list node. A single unlucky allocation that walks into the kernel
//      for a fresh page is a 10-microsecond outlier in a 100-nanosecond
//      budget, and it will land on the message you cared about most.
//   2. No 64-bit pointers where a 32-bit index will do. An index halves the
//      size of every link field, which doubles the number of orders per cache
//      line. For an intrusive list that is walked on every execution, that is
//      the difference between one cache line and two.
//
// So: orders live in one contiguous, pre-faulted array. Links are uint32
// indices into it. Index 0 is reserved as the null sentinel so that a
// zero-initialised structure is already "empty" and there is no separate
// invalid-value to remember.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>

#if defined(__linux__)
#  include <sys/mman.h>
#endif

namespace lob {

inline constexpr std::uint32_t kNullIdx = 0;

// ---------------------------------------------------------------------------
// Pool<T> — index-addressed slab with an intrusive free list.
//
// Freed slots are threaded through a free list stored *in the freed objects
// themselves*, so the free list costs no extra memory. Allocation is a load
// and a store; deallocation is two stores. Both are branchless apart from the
// exhaustion check.
//
// T must have a `uint32_t next` member usable as the free-list link.
// ---------------------------------------------------------------------------
template <class T>
class Pool {
public:
    explicit Pool(std::size_t capacity) : cap_(static_cast<std::uint32_t>(capacity + 1)) {  // +1 for the sentinel
        const std::size_t bytes = cap_ * sizeof(T);
        slots_ = static_cast<T*>(::operator new(bytes, std::align_val_t{64}));
        std::memset(static_cast<void*>(slots_), 0, bytes);

#if defined(__linux__) && defined(MADV_HUGEPAGE)
        // Ask for transparent huge pages. An order pool of a few million
        // entries is tens of MB; at 4KB pages that is thousands of dTLB
        // entries and the walk will miss constantly. 2MB pages cut the TLB
        // footprint by ~512x. Advisory — the kernel may decline.
        ::madvise(slots_, bytes, MADV_HUGEPAGE);
#endif
        // Pre-fault every page now, at construction, so the first trading
        // message never pays a minor page fault. `memset` above already did
        // this; the explicit touch below is belt-and-braces if the memset is
        // ever replaced by a lazier init.
        reset();
    }

    ~Pool() { ::operator delete(slots_, std::align_val_t{64}); }
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    void reset() noexcept {
        // Thread every slot (except the 0 sentinel) onto the free list, in
        // ascending order so the first allocations are sequential in memory
        // and the prefetcher can help.
        for (std::uint32_t i = 1; i < cap_ - 1; ++i)
            slots_[i].next = i + 1;
        slots_[cap_ - 1].next = kNullIdx;
        free_head_ = 1;
        live_ = 0;
        high_water_ = 0;
    }

    [[gnu::always_inline]] inline std::uint32_t alloc() noexcept {
        const std::uint32_t i = free_head_;
        if (i == kNullIdx) [[unlikely]] return kNullIdx;   // exhausted
        free_head_ = slots_[i].next;
        ++live_;
        if (live_ > high_water_) high_water_ = live_;
        return i;
    }

    [[gnu::always_inline]] inline void free(std::uint32_t i) noexcept {
        slots_[i].next = free_head_;
        free_head_ = i;
        --live_;
    }

    [[gnu::always_inline]] inline T&       operator[](std::uint32_t i)       noexcept { return slots_[i]; }
    [[gnu::always_inline]] inline const T& operator[](std::uint32_t i) const noexcept { return slots_[i]; }

    std::size_t capacity()   const noexcept { return cap_ - 1; }
    std::size_t live()       const noexcept { return live_; }
    std::size_t high_water() const noexcept { return high_water_; }

private:
    T*            slots_{nullptr};
    std::uint32_t cap_{0};
    std::uint32_t free_head_{kNullIdx};
    std::size_t   live_{0};
    std::size_t   high_water_{0};
};

// ---------------------------------------------------------------------------
// OrderMap — open-addressing hash map from 64-bit ITCH order reference to a
// 32-bit pool index.
//
// Why not std::unordered_map: it is a bucket array of pointers to separately
// allocated nodes. Every lookup is a hash, a load of the bucket pointer, then
// a dependent load of the node — two cache misses minimum, and the nodes are
// scattered by malloc. It also allocates on every insert, which violates rule
// 1 above. Measured in bench_main: roughly 3x slower here.
//
// Design choices:
//   * Linear probing, not quadratic or double hashing. Probe sequences stay
//     within a cache line most of the time, so a collision is usually free.
//   * Fibonacci hashing (multiply by 2^64/phi, take the high bits). ITCH order
//     references are near-sequential, and sequential keys through a
//     multiplicative hash land in scattered-but-uniform buckets. Taking the
//     LOW bits of a sequential key instead would cluster catastrophically.
//   * Backward-shift deletion, not tombstones. Tombstones degrade a heavily
//     churned map — and an order book is nothing but churn, with the vast
//     majority of orders cancelled rather than executed. With tombstones this
//     map would slowly turn into a linear scan over a trading day.
//   * Key 0 is the empty sentinel. ITCH order reference 0 does not occur.
// ---------------------------------------------------------------------------
class OrderMap {
public:
    explicit OrderMap(std::size_t capacity_log2 = 20)
        : cap_(std::size_t{1} << capacity_log2),
          mask_(cap_ - 1),
          shift_(static_cast<unsigned>(64 - capacity_log2)) {
        const std::size_t bytes = cap_ * sizeof(Entry);
        entries_ = static_cast<Entry*>(::operator new(bytes, std::align_val_t{64}));
        std::memset(static_cast<void*>(entries_), 0, bytes);
#if defined(__linux__) && defined(MADV_HUGEPAGE)
        ::madvise(entries_, bytes, MADV_HUGEPAGE);
#endif
    }
    ~OrderMap() { ::operator delete(entries_, std::align_val_t{64}); }
    OrderMap(const OrderMap&) = delete;
    OrderMap& operator=(const OrderMap&) = delete;

    void clear() noexcept {
        std::memset(static_cast<void*>(entries_), 0, cap_ * sizeof(Entry));
        size_ = 0;
        probes_ = 0;
        lookups_ = 0;
        overflows_ = 0;
    }

    // An open-addressing table with linear probing does NOT degrade gracefully
    // when it fills — it hangs. There is no empty slot to terminate the probe,
    // so the loop runs forever. I found this the way everyone finds it: a
    // benchmark that produced no output and no crash, just a process sitting
    // at 100% CPU.
    //
    // In a feed handler that failure mode is worse than a crash: the process
    // is alive, the health check passes, and the book silently stops updating.
    // So the guard is explicit and loud. Sizing is the caller's job — see
    // BookStats::pool_exhausted and the load_factor() accessor — but the map
    // refuses to hang.
    static constexpr double kMaxLoad = 0.85;

    [[gnu::always_inline]] inline bool full() const noexcept {
        return size_ >= static_cast<std::size_t>(static_cast<double>(cap_) * kMaxLoad);
    }
    std::size_t overflows() const noexcept { return overflows_; }

    [[gnu::always_inline]] inline void insert(std::uint64_t key, std::uint32_t val) noexcept {
        if (full()) [[unlikely]] { ++overflows_; return; }
        std::size_t i = slot(key);
        ++lookups_;
        while (entries_[i].key != 0) {
            ++probes_;
            if (entries_[i].key == key) { entries_[i].val = val; return; }  // replace
            i = (i + 1) & mask_;
        }
        entries_[i].key = key;
        entries_[i].val = val;
        ++size_;
    }

    // Returns kNullIdx when absent.
    [[gnu::always_inline]] inline std::uint32_t find(std::uint64_t key) const noexcept {
        std::size_t i = slot(key);
        ++lookups_;
        for (;;) {
            const Entry e = entries_[i];
            if (e.key == key) return e.val;
            if (e.key == 0)   return kNullIdx;
            ++probes_;
            i = (i + 1) & mask_;
        }
    }

    // Backward-shift deletion: after removing an entry, pull forward any
    // subsequent entry whose ideal slot is at or before the hole, so the
    // invariant "no empty slot inside a probe chain" is preserved without
    // tombstones.
    [[gnu::always_inline]] inline bool erase(std::uint64_t key) noexcept {
        std::size_t i = slot(key);
        ++lookups_;
        for (;;) {
            if (entries_[i].key == key) break;
            if (entries_[i].key == 0) return false;
            ++probes_;
            i = (i + 1) & mask_;
        }
        std::size_t hole = i;
        std::size_t j = i;
        for (;;) {
            j = (j + 1) & mask_;
            const std::uint64_t k = entries_[j].key;
            if (k == 0) break;
            const std::size_t ideal = slot(k);
            // Is `ideal` cyclically within (hole, j]?  If not, this entry can
            // legally move back into the hole.
            const std::size_t d_hole = (j - hole) & mask_;
            const std::size_t d_ideal = (j - ideal) & mask_;
            if (d_ideal >= d_hole) {
                entries_[hole] = entries_[j];
                hole = j;
            }
        }
        entries_[hole].key = 0;
        entries_[hole].val = 0;
        --size_;
        return true;
    }

    std::size_t size() const noexcept { return size_; }
    double load_factor() const noexcept {
        return static_cast<double>(size_) / static_cast<double>(cap_);
    }
    // Average probes beyond the first slot. Above ~1.0 means the map is too
    // full or the hash is clustering; both are worth knowing before you blame
    // the book.
    double avg_probes() const noexcept {
        return lookups_ ? static_cast<double>(probes_) / static_cast<double>(lookups_) : 0.0;
    }

private:
    struct Entry {
        std::uint64_t key;
        std::uint32_t val;
        std::uint32_t _pad;
    };
    static_assert(sizeof(Entry) == 16, "4 entries per cache line");

    [[gnu::always_inline]] inline std::size_t slot(std::uint64_t k) const noexcept {
        // 2^64 / golden ratio. High bits of the product are well mixed even
        // for sequential keys, which is exactly what ITCH gives us.
        return static_cast<std::size_t>((k * 0x9E3779B97F4A7C15ULL) >> shift_);
    }

    Entry*      entries_{nullptr};
    std::size_t cap_{0};
    std::size_t mask_{0};
    unsigned    shift_{0};
    std::size_t size_{0};
    std::size_t overflows_{0};
    mutable std::size_t probes_{0};
    mutable std::size_t lookups_{0};
};

}  // namespace lob

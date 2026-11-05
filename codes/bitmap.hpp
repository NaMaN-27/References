// lob/bitmap.hpp — three-level hierarchical bitmap over the price ladder.
//
// The problem it solves
// ---------------------
// A flat price ladder gives O(1) add/cancel/execute at a known price. What it
// does not give you for free is "what is the best bid *now*" after the current
// best level is depleted. The naive answer is to walk down the ladder until a
// non-empty level is found. Usually that walk is 1-2 steps. Occasionally —
// after a sweep through thin book, or on a stock that quotes in wide
// increments — it is hundreds of steps, and it happens at exactly the moment
// the market is moving and you least want a stall. That is a latency spike
// correlated with market activity, which is the worst kind.
//
// The fix: keep an occupancy bitmap over the ladder and find the next
// occupied level with hardware bit-scan instructions instead of a loop.
//
// Layout for a ladder of 2^16 = 65536 levels:
//   L0: 1024 words   (1 bit per price level)
//   L1:   16 words   (1 bit per L0 word that is non-zero)
//   L2:    1 word    (1 bit per L1 word that is non-zero)
//
// find_highest_below / find_lowest_above are then at most 3 dependent loads
// plus 3 bit-scans, regardless of how far away the next level is. No loop, no
// data-dependent iteration count, and — the property that actually matters —
// no correlation between cost and market conditions.
//
// Cost: 8KB + 128B + 8B of bitmap for a 1MB ladder, and two extra stores on
// the transition edges (level becomes occupied / becomes empty). Those stores
// only happen when a level's order count crosses zero, not on every message.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace lob {

template <std::size_t LevelsLog2>
class HierBitmap {
    static_assert(LevelsLog2 >= 12 && LevelsLog2 <= 24);

public:
    static constexpr std::size_t kLevels = std::size_t{1} << LevelsLog2;
    static constexpr std::size_t kL0Words = kLevels / 64;
    static constexpr std::size_t kL1Words = (kL0Words + 63) / 64;
    static constexpr std::size_t kL2Words = (kL1Words + 63) / 64;
    static_assert(kL2Words <= 64, "need a 4th level for a ladder this large");

    HierBitmap() { clear(); }

    void clear() noexcept {
        std::memset(l0_, 0, sizeof(l0_));
        std::memset(l1_, 0, sizeof(l1_));
        std::memset(l2_, 0, sizeof(l2_));
    }

    [[gnu::always_inline]] inline void set(std::size_t i) noexcept {
        const std::size_t w0 = i >> 6;
        l0_[w0] |= bit(i);
        const std::size_t w1 = w0 >> 6;
        l1_[w1] |= bit(w0);
        l2_[w1 >> 6] |= bit(w1);
    }

    // Only clears the summary bits when the child word actually became zero.
    // Doing this unconditionally would be wrong: another level in the same
    // 64-level block may still be occupied.
    [[gnu::always_inline]] inline void reset(std::size_t i) noexcept {
        const std::size_t w0 = i >> 6;
        l0_[w0] &= ~bit(i);
        if (l0_[w0]) return;

        const std::size_t w1 = w0 >> 6;
        l1_[w1] &= ~bit(w0);
        if (l1_[w1]) return;

        l2_[w1 >> 6] &= ~bit(w1);
    }

    [[gnu::always_inline]] inline bool test(std::size_t i) const noexcept {
        return (l0_[i >> 6] & bit(i)) != 0;
    }

    bool empty() const noexcept {
        for (std::size_t i = 0; i < kL2Words; ++i)
            if (l2_[i]) return false;
        return true;
    }

    static constexpr std::size_t kNone = ~std::size_t{0};

    // Highest occupied index <= from.  Used for the bid side.
    [[gnu::always_inline]] inline std::size_t highest_at_or_below(std::size_t from) const noexcept {
        if (from >= kLevels) from = kLevels - 1;

        // 1. Try within the starting L0 word, masking off bits above `from`.
        std::size_t w0 = from >> 6;
        std::uint64_t m = l0_[w0] & mask_at_or_below(from & 63);
        if (m) return (w0 << 6) | msb(m);

        // 2. Find an earlier non-empty L0 word via the L1 summary.
        std::size_t w1 = w0 >> 6;
        if (w0 & 63) {
            m = l1_[w1] & mask_below(w0 & 63);
            if (m) { w0 = (w1 << 6) | msb(m); return (w0 << 6) | msb(l0_[w0]); }
        }

        // 3. Find an earlier non-empty L1 word via the L2 summary.
        const std::size_t w2 = w1 >> 6;
        if (w1 & 63) {
            m = l2_[w2] & mask_below(w1 & 63);
            if (m) {
                w1 = (w2 << 6) | msb(m);
                w0 = (w1 << 6) | msb(l1_[w1]);
                return (w0 << 6) | msb(l0_[w0]);
            }
        }
        // 4. Walk the (tiny) L2 array backwards.
        for (std::size_t i = w2; i-- > 0;) {
            if (l2_[i]) {
                w1 = (i << 6) | msb(l2_[i]);
                w0 = (w1 << 6) | msb(l1_[w1]);
                return (w0 << 6) | msb(l0_[w0]);
            }
        }
        return kNone;
    }

    // Lowest occupied index >= from.  Used for the ask side.
    [[gnu::always_inline]] inline std::size_t lowest_at_or_above(std::size_t from) const noexcept {
        if (from >= kLevels) return kNone;

        std::size_t w0 = from >> 6;
        std::uint64_t m = l0_[w0] & mask_at_or_above(from & 63);
        if (m) return (w0 << 6) | lsb(m);

        std::size_t w1 = w0 >> 6;
        if ((w0 & 63) != 63) {
            m = l1_[w1] & mask_above(w0 & 63);
            if (m) { w0 = (w1 << 6) | lsb(m); return (w0 << 6) | lsb(l0_[w0]); }
        }

        const std::size_t w2 = w1 >> 6;
        if ((w1 & 63) != 63) {
            m = l2_[w2] & mask_above(w1 & 63);
            if (m) {
                w1 = (w2 << 6) | lsb(m);
                w0 = (w1 << 6) | lsb(l1_[w1]);
                return (w0 << 6) | lsb(l0_[w0]);
            }
        }
        for (std::size_t i = w2 + 1; i < kL2Words; ++i) {
            if (l2_[i]) {
                w1 = (i << 6) | lsb(l2_[i]);
                w0 = (w1 << 6) | lsb(l1_[w1]);
                return (w0 << 6) | lsb(l0_[w0]);
            }
        }
        return kNone;
    }

    std::size_t popcount() const noexcept {
        std::size_t n = 0;
        for (std::size_t i = 0; i < kL0Words; ++i)
            n += static_cast<std::size_t>(__builtin_popcountll(l0_[i]));
        return n;
    }

private:
    static constexpr std::uint64_t bit(std::size_t i) noexcept {
        return std::uint64_t{1} << (i & 63);
    }
    // Bits [0, b] inclusive.
    static constexpr std::uint64_t mask_at_or_below(std::size_t b) noexcept {
        return b >= 63 ? ~std::uint64_t{0} : ((std::uint64_t{1} << (b + 1)) - 1);
    }
    static constexpr std::uint64_t mask_below(std::size_t b) noexcept {
        return (std::uint64_t{1} << b) - 1;
    }
    static constexpr std::uint64_t mask_at_or_above(std::size_t b) noexcept {
        return ~std::uint64_t{0} << b;
    }
    static constexpr std::uint64_t mask_above(std::size_t b) noexcept {
        return b >= 63 ? 0 : (~std::uint64_t{0} << (b + 1));
    }
    static std::size_t msb(std::uint64_t x) noexcept {
        return 63 - static_cast<std::size_t>(__builtin_clzll(x));
    }
    static std::size_t lsb(std::uint64_t x) noexcept {
        return static_cast<std::size_t>(__builtin_ctzll(x));
    }

    alignas(64) std::uint64_t l0_[kL0Words];
    alignas(64) std::uint64_t l1_[kL1Words];
    alignas(64) std::uint64_t l2_[kL2Words];
};

}  // namespace lob

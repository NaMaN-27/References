// ctse/cores.hpp — the actual strategy arithmetic, written exactly once.
//
// THIS FILE IS THE INTEGRITY OF THE WHOLE BENCHMARK.
//
// A "static vs virtual dispatch" comparison is worthless if the two arms
// compute different things. The usual failure mode is subtle: someone writes
// the virtual version with a `double alpha` member and the CRTP version with a
// template constant, then reports a 4x win that is mostly constant folding of
// the alpha and only a little dispatch.
//
// So: every arm holds a `*Core` object from this file and forwards to
// `core.step(tick)`. The cores are the only place arithmetic exists. The
// engine layers differ *only* in how `step` is reached:
//
//     static arm  : CRTP  -> inlined, no call
//     virtual arm : vtable -> indirect call, opaque across TU boundary
//
// `verify` then asserts the two arms emit bit-identical Signal streams over
// the full input. If that assertion fails, the benchmark is void.
//
// Note the cores are still templated on their windows. Both arms get that
// benefit equally, so it cancels. What does NOT cancel is inlining — and that
// is precisely the effect we are trying to isolate.

#pragma once

#include <cstdint>

#include "ctse/indicators.hpp"
#include "ctse/types.hpp"

namespace ctse::core {

// Convert a raw score to a Signal with a symmetric deadband.
// Written as a free function so both arms share the branch structure too.
CTSE_ALWAYS_INLINE Signal to_signal(double score, double threshold,
                                    std::uint16_t id) noexcept {
    if (score > threshold) {
        const double s = score > 1.0 ? 1.0 : score;
        return Signal{s, Side::Buy,
                      static_cast<std::uint8_t>(s * 255.0), id, 0};
    }
    if (score < -threshold) {
        const double s = score < -1.0 ? -1.0 : score;
        return Signal{s, Side::Sell,
                      static_cast<std::uint8_t>(-s * 255.0), id, 0};
    }
    return no_signal();
}

// ---------------------------------------------------------------------------
// 1. EMA crossover. Fast EMA above slow EMA => long. Normalised by mid so the
//    score is comparable across instruments and price levels.
// ---------------------------------------------------------------------------
template <std::size_t Fast, std::size_t Slow, int ThreshBps>
struct EmaCrossCore {
    static_assert(Fast < Slow, "fast period must be shorter than slow");
    static constexpr std::uint16_t kId = 1;
    static constexpr double kThresh = static_cast<double>(ThreshBps) / 10000.0;

    CTSE_ALWAYS_INLINE Signal step(const Tick& t) noexcept {
        const double m = t.mid();
        fast_.push(m);
        slow_.push(m);
        if (!slow_.warm()) [[unlikely]] return no_signal();
        const double d = (fast_.value() - slow_.value()) / m;
        return to_signal(d / kThresh, 1.0, kId);
    }

    ind::Ema<Fast> fast_{};
    ind::Ema<Slow> slow_{};
};

// ---------------------------------------------------------------------------
// 2. Z-score mean reversion. Price far above its own recent mean => fade it.
//    Sign is inverted relative to momentum: high z => sell.
// ---------------------------------------------------------------------------
template <std::size_t Window, int EntryZx100>
struct ZReversionCore {
    static constexpr std::uint16_t kId = 2;
    static constexpr double kEntry = static_cast<double>(EntryZx100) / 100.0;

    CTSE_ALWAYS_INLINE Signal step(const Tick& t) noexcept {
        mom_.push(t.mid());
        if (!mom_.full()) [[unlikely]] return no_signal();
        const double z = mom_.zscore();
        return to_signal(-z / kEntry, 1.0, kId);
    }

    ind::RollingMoments<Window> mom_{};
};

// ---------------------------------------------------------------------------
// 3. Book-imbalance momentum, smoothed. Persistent bid-side pressure => long.
// ---------------------------------------------------------------------------
template <std::size_t Smooth, int ThreshX100>
struct ImbalanceCore {
    static constexpr std::uint16_t kId = 3;
    static constexpr double kThresh = static_cast<double>(ThreshX100) / 100.0;

    CTSE_ALWAYS_INLINE Signal step(const Tick& t) noexcept {
        smooth_.push(t.imbalance());
        if (!smooth_.warm()) [[unlikely]] return no_signal();
        return to_signal(smooth_.value() / kThresh, 1.0, kId);
    }

    ind::Ema<Smooth> smooth_{};
};

// ---------------------------------------------------------------------------
// 4. Spread gate. Not a signal generator — a veto. Wide spread => stand down.
//    Included because a real chain is mostly filters, and filters are where
//    the inlining win compounds: each one is a couple of instructions wrapped
//    in a call the virtual arm cannot remove.
// ---------------------------------------------------------------------------
template <int MaxSpreadBps>
struct SpreadGateCore {
    static constexpr std::uint16_t kId = 4;
    static constexpr double kMax = static_cast<double>(MaxSpreadBps) / 10000.0;

    CTSE_ALWAYS_INLINE Signal step(const Tick& t) noexcept {
        const double m = t.mid();
        const double rel = m > 0.0 ? t.spread() / m : 1.0;
        // Emits a tiny negative-confidence marker when the gate is closed so
        // the combiner sees it; strength stays 0 so it never wins on merit.
        blocked_ = rel > kMax;
        return no_signal();
    }
    CTSE_ALWAYS_INLINE bool blocked() const noexcept { return blocked_; }

    bool blocked_{false};
};

}  // namespace ctse::core

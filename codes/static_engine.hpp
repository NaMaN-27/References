// ctse/static_engine.hpp — compile-time dispatch via CRTP.
//
// The pattern
// -----------
//     template <class Derived>
//     struct StrategyBase {
//         Signal on_tick(const Tick& t) {
//             return static_cast<Derived*>(this)->on_tick_impl(t);
//         }
//     };
//     struct MyStrat : StrategyBase<MyStrat> {
//         Signal on_tick_impl(const Tick&);
//     };
//
// The base knows its derived type as a template parameter, so the downcast is
// resolved at compile time and `on_tick_impl` is a direct, inlinable call.
// This is "static polymorphism": you get an interface contract and code reuse
// with zero runtime indirection.
//
// Why CRTP over plain free functions or a concept-constrained template?
//   * CRTP lets the base implement *shared behaviour* in terms of the derived
//     hooks — see `on_tick_counted` below, which adds instrumentation every
//     strategy inherits without any of them writing it, and without a call.
//   * With C++20 concepts you can additionally constrain the contract so a
//     malformed strategy fails at the definition site with a readable error
//     instead of 40 lines of template spew. Use both: CRTP for the mechanism,
//     concepts for the diagnostics.
//
// The costs, honestly:
//   * The type is part of the caller's type. You cannot store heterogeneous
//     strategies in one container, cannot swap them at runtime, cannot cross
//     an ABI boundary with them, and every distinct instantiation is separate
//     code (I-cache pressure, longer builds).
//   * Debug builds are *slower* than virtual, because none of it inlines.
// Reach for it on a hot path with a known, small set of strategies. Do not
// rewrite your config-driven backtester in it.

#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

#include "ctse/cores.hpp"
#include "ctse/types.hpp"

namespace ctse::stat {

// --- Concept: the contract every static strategy must satisfy --------------
template <class S>
concept Strategy = requires(S s, const Tick& t) {
    { s.on_tick(t) } noexcept -> std::same_as<Signal>;
    { S::id() } -> std::convertible_to<std::uint16_t>;
};

// --- CRTP base -------------------------------------------------------------
template <class Derived>
struct StrategyBase {
    CTSE_ALWAYS_INLINE Signal on_tick(const Tick& t) noexcept {
        // Compile-time downcast. No vtable, no runtime check, fully inlinable.
        return static_cast<Derived*>(this)->on_tick_impl(t);
    }

    // Shared behaviour written once in the base, injected into every derived
    // class with no indirection. This is the thing plain templates give you
    // less cleanly.
    CTSE_ALWAYS_INLINE Signal on_tick_counted(const Tick& t) noexcept {
        ++ticks_seen_;
        const Signal s = on_tick(t);
        fires_ += static_cast<std::uint64_t>(s.active());
        return s;
    }

    CTSE_ALWAYS_INLINE std::uint64_t ticks_seen() const noexcept { return ticks_seen_; }
    CTSE_ALWAYS_INLINE std::uint64_t fires() const noexcept { return fires_; }

protected:
    std::uint64_t ticks_seen_{0};
    std::uint64_t fires_{0};
};

// --- Concrete strategies: thin CRTP shells over the shared cores -----------

template <std::size_t Fast, std::size_t Slow, int ThreshBps>
struct EmaCross : StrategyBase<EmaCross<Fast, Slow, ThreshBps>> {
    using Core = core::EmaCrossCore<Fast, Slow, ThreshBps>;
    static constexpr std::uint16_t id() noexcept { return Core::kId; }

    CTSE_ALWAYS_INLINE Signal on_tick_impl(const Tick& t) noexcept {
        return core_.step(t);
    }
    Core core_{};
};

template <std::size_t Window, int EntryZx100>
struct ZReversion : StrategyBase<ZReversion<Window, EntryZx100>> {
    using Core = core::ZReversionCore<Window, EntryZx100>;
    static constexpr std::uint16_t id() noexcept { return Core::kId; }

    CTSE_ALWAYS_INLINE Signal on_tick_impl(const Tick& t) noexcept {
        return core_.step(t);
    }
    Core core_{};
};

template <std::size_t Smooth, int ThreshX100>
struct Imbalance : StrategyBase<Imbalance<Smooth, ThreshX100>> {
    using Core = core::ImbalanceCore<Smooth, ThreshX100>;
    static constexpr std::uint16_t id() noexcept { return Core::kId; }

    CTSE_ALWAYS_INLINE Signal on_tick_impl(const Tick& t) noexcept {
        return core_.step(t);
    }
    Core core_{};
};

template <int MaxSpreadBps>
struct SpreadGate : StrategyBase<SpreadGate<MaxSpreadBps>> {
    using Core = core::SpreadGateCore<MaxSpreadBps>;
    static constexpr std::uint16_t id() noexcept { return Core::kId; }

    CTSE_ALWAYS_INLINE Signal on_tick_impl(const Tick& t) noexcept {
        return core_.step(t);
    }
    Core core_{};
};

// --- Chain: N strategies composed at compile time --------------------------
//
// The strategies live in a std::tuple, so they are laid out contiguously
// inside the chain object — one allocation, no pointer chasing. `std::apply`
// plus a comma fold expands to a straight-line sequence of inlined bodies.
// There is no loop over strategies at runtime; the loop was unrolled by the
// type system.
template <Strategy... S>
class Chain {
public:
    static constexpr std::size_t kSize = sizeof...(S);

    CTSE_ALWAYS_INLINE Signal on_tick(const Tick& t) noexcept {
        Signal acc = no_signal();
        std::apply(
            [&](auto&... s) {
                ((acc = combine(acc, s.on_tick(t))), ...);
            },
            legs_);
        return acc;
    }

    // Same traversal but with the base-class instrumentation enabled, to show
    // the CRTP shared-behaviour path also costs nothing.
    CTSE_ALWAYS_INLINE Signal on_tick_counted(const Tick& t) noexcept {
        Signal acc = no_signal();
        std::apply(
            [&](auto&... s) {
                ((acc = combine(acc, s.on_tick_counted(t))), ...);
            },
            legs_);
        return acc;
    }

    template <std::size_t I>
    CTSE_ALWAYS_INLINE auto& leg() noexcept { return std::get<I>(legs_); }

private:
    std::tuple<S...> legs_{};
};

// --- Compile-time feature toggles ------------------------------------------
//
// Experiment D. `if constexpr` deletes the untaken branch entirely: no test,
// no jump, no I-cache footprint, and the dead code's state is not even
// initialised. A runtime `if (flag)` must keep both paths resident and pay a
// (usually well-predicted, but not free) branch plus a load of the flag.
struct Features {
    bool use_imbalance;
    bool use_reversion;
    bool clamp_output;
};

template <Features F, std::size_t Fast, std::size_t Slow>
class TunedStrategy {
public:
    CTSE_ALWAYS_INLINE Signal on_tick(const Tick& t) noexcept {
        Signal acc = ema_.on_tick(t);
        if constexpr (F.use_imbalance) acc = combine(acc, imb_.on_tick(t));
        if constexpr (F.use_reversion) acc = combine(acc, rev_.on_tick(t));
        if constexpr (F.clamp_output) {
            if (acc.strength >  0.75) acc.strength =  0.75;
            if (acc.strength < -0.75) acc.strength = -0.75;
        }
        return acc;
    }

private:
    EmaCross<Fast, Slow, 5> ema_{};
    // Members exist unconditionally for layout simplicity; the *code* paths
    // are what `if constexpr` removes. To also drop the storage, specialise
    // on F or use [[no_unique_address]] with an empty tag type.
    Imbalance<16, 30>  imb_{};
    ZReversion<64, 200> rev_{};
};

// Runtime-flag equivalent, for the A/B.
template <std::size_t Fast, std::size_t Slow>
class TunedStrategyRuntime {
public:
    explicit TunedStrategyRuntime(Features f) noexcept : f_(f) {}

    CTSE_ALWAYS_INLINE Signal on_tick(const Tick& t) noexcept {
        Signal acc = ema_.on_tick(t);
        if (f_.use_imbalance) acc = combine(acc, imb_.on_tick(t));
        if (f_.use_reversion) acc = combine(acc, rev_.on_tick(t));
        if (f_.clamp_output) {
            if (acc.strength >  0.75) acc.strength =  0.75;
            if (acc.strength < -0.75) acc.strength = -0.75;
        }
        return acc;
    }

private:
    Features f_;
    EmaCross<Fast, Slow, 5> ema_{};
    Imbalance<16, 30>  imb_{};
    ZReversion<64, 200> rev_{};
};

}  // namespace ctse::stat

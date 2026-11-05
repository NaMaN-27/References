// ctse/virtual_engine.hpp — the runtime-dispatch baseline.
//
// Two deliberately different virtual arms, because "virtual is slow" is a lazy
// claim and the interesting question is *which part* is slow.
//
//   A) OPAQUE  — concrete classes are defined only in virtual_impl.cpp. At the
//      call site the compiler sees an `IStrategy*` and nothing else. It cannot
//      devirtualise, cannot inline, cannot see that the body is six FLOPs.
//      This is what real codebases look like: strategy in one TU, engine in
//      another, no LTO because the build already takes nine minutes.
//
//   B) VISIBLE + FINAL — concrete classes defined here, marked `final`, and
//      constructed as concrete objects the optimiser can trace. GCC/Clang will
//      speculatively devirtualise and often fully inline. Including this arm
//      is the honest thing to do: it shows the compiler can claw most of the
//      gap back *when it can see everything*, and it makes the point that the
//      real enemy is the optimisation barrier, not the `call *%rax`.
//
// Anatomy of the cost, in the order that matters:
//
//   1. INLINING BARRIER (dominant). An opaque call forces the ABI: arguments
//      into registers, caller-saved registers spilled, no constant propagation
//      into the body, no cross-strategy CSE (every leg recomputes t.mid()), no
//      vectorisation across legs. This is typically 70-90% of the delta.
//   2. INDIRECT BRANCH. Load vptr, load slot, `call *`. If one type dominates,
//      the BTB predicts it and this is ~2-3 cycles. With several types
//      interleaved unpredictably it mispredicts and costs 15-20.
//   3. MEMORY LAYOUT. `vector<unique_ptr<IStrategy>>` is a pointer array to
//      scattered heap objects. Each leg is a dependent load: pointer -> object
//      -> vptr -> vtable -> function. Four serialised loads before the first
//      instruction of the strategy runs, each a potential cache miss.
//      The tuple-based static chain has all state contiguous and prefetchable.
//
// Point 3 is why an arena-allocated virtual chain (also provided) is a fair
// intermediate: it isolates the dispatch cost from the allocation-layout cost.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "ctse/cores.hpp"
#include "ctse/types.hpp"

namespace ctse::virt {

// --- The runtime interface -------------------------------------------------
class IStrategy {
public:
    virtual ~IStrategy() = default;
    virtual Signal on_tick(const Tick& t) noexcept = 0;
    virtual std::uint16_t id() const noexcept = 0;
};

// ===========================================================================
// ARM A: OPAQUE. Only factories are visible here; definitions live in
// virtual_impl.cpp. Without LTO the optimiser has no idea what these are.
// ===========================================================================
std::unique_ptr<IStrategy> make_ema_cross();    // EmaCrossCore<8,64,5>
std::unique_ptr<IStrategy> make_z_reversion();  // ZReversionCore<64,200>
std::unique_ptr<IStrategy> make_imbalance();    // ImbalanceCore<16,30>
std::unique_ptr<IStrategy> make_spread_gate();  // SpreadGateCore<20>

// Round-robin factory used to build chains of length N with a controllable
// number of distinct dynamic types.
std::unique_ptr<IStrategy> make_nth(std::size_t i, std::size_t distinct_types);

// CLONES: four DISTINCT dynamic types that wrap the IDENTICAL core, so they
// perform byte-for-byte the same arithmetic and differ only in vtable address.
// This is the control that isolates indirect-branch prediction from every
// other effect. Without it, "monomorphic vs polymorphic" secretly compares
// different amounts of floating-point work and the result is meaningless.
//
// NOTE: virtual_impl.cpp must be compiled with -fno-ipa-icf. GCC's identical
// code folding will otherwise merge the four identical on_tick bodies into one
// function, every vtable slot will point at the same address, the BTB will see
// a single target, and the experiment silently measures nothing.
std::unique_ptr<IStrategy> make_ema_clone(std::size_t k);  // k in [0, 4)

// --- Runtime chain ---------------------------------------------------------
class Chain {
public:
    void add(std::unique_ptr<IStrategy> s) { legs_.push_back(std::move(s)); }

    // Marked noinline so it shows up as its own symbol in the disassembly and
    // cannot be melted into the benchmark loop, which would confuse the
    // instruction counts in scripts/inspect_asm.sh.
    CTSE_NOINLINE Signal on_tick(const Tick& t) noexcept {
        Signal acc = no_signal();
        for (auto& s : legs_) acc = combine(acc, s->on_tick(t));
        return acc;
    }

    std::size_t size() const noexcept { return legs_.size(); }

private:
    std::vector<std::unique_ptr<IStrategy>> legs_;
};

// --- Arena-backed chain: same dispatch, better layout ----------------------
//
// Objects are placement-new'd into one contiguous, cache-line-aligned buffer
// and referenced by raw pointer. Removes the unique_ptr indirection and the
// heap scatter, so the residual gap vs the static chain is dispatch + inlining
// only. This is the control that makes the experiment interpretable.
class ArenaChain {
public:
    explicit ArenaChain(std::size_t bytes = 1 << 16)
        : buf_(static_cast<std::byte*>(::operator new(bytes, std::align_val_t{64}))),
          cap_(bytes) {}

    ArenaChain(const ArenaChain&) = delete;
    ArenaChain& operator=(const ArenaChain&) = delete;

    ~ArenaChain() {
        for (auto* p : legs_) p->~IStrategy();
        ::operator delete(buf_, std::align_val_t{64});
    }

    // Constructs T in the arena. Requires the concrete type, so this only
    // works with ARM B (visible) types — which is fine: its job is to isolate
    // layout, and ARM A already covers the opaque case.
    template <class T, class... Args>
    T* emplace(Args&&... args) {
        const std::size_t align = alignof(T) < 64 ? 64 : alignof(T);
        used_ = (used_ + align - 1) & ~(align - 1);
        T* p = new (buf_ + used_) T(static_cast<Args&&>(args)...);
        used_ += sizeof(T);
        legs_.push_back(p);
        return p;
    }

    CTSE_NOINLINE Signal on_tick(const Tick& t) noexcept {
        Signal acc = no_signal();
        for (auto* s : legs_) acc = combine(acc, s->on_tick(t));
        return acc;
    }

    std::size_t size() const noexcept { return legs_.size(); }

private:
    std::byte*               buf_{nullptr};
    std::size_t              cap_{0};
    std::size_t              used_{0};
    std::vector<IStrategy*>  legs_;
};

// ===========================================================================
// ARM B: VISIBLE + FINAL. Same cores, definitions in the header.
// `final` lets the compiler prove the dynamic type at a call site whose static
// type is the concrete class, enabling devirtualisation and inlining.
// ===========================================================================
namespace visible {

template <std::size_t Fast, std::size_t Slow, int ThreshBps>
class EmaCross final : public IStrategy {
public:
    Signal on_tick(const Tick& t) noexcept override { return core_.step(t); }
    std::uint16_t id() const noexcept override { return Core::kId; }
private:
    using Core = core::EmaCrossCore<Fast, Slow, ThreshBps>;
    Core core_{};
};

template <std::size_t Window, int EntryZx100>
class ZReversion final : public IStrategy {
public:
    Signal on_tick(const Tick& t) noexcept override { return core_.step(t); }
    std::uint16_t id() const noexcept override { return Core::kId; }
private:
    using Core = core::ZReversionCore<Window, EntryZx100>;
    Core core_{};
};

template <std::size_t Smooth, int ThreshX100>
class Imbalance final : public IStrategy {
public:
    Signal on_tick(const Tick& t) noexcept override { return core_.step(t); }
    std::uint16_t id() const noexcept override { return Core::kId; }
private:
    using Core = core::ImbalanceCore<Smooth, ThreshX100>;
    Core core_{};
};

template <int MaxSpreadBps>
class SpreadGate final : public IStrategy {
public:
    Signal on_tick(const Tick& t) noexcept override { return core_.step(t); }
    std::uint16_t id() const noexcept override { return Core::kId; }
private:
    using Core = core::SpreadGateCore<MaxSpreadBps>;
    Core core_{};
};

}  // namespace visible
}  // namespace ctse::virt

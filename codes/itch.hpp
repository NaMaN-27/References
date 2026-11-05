// lob/itch.hpp — Nasdaq TotalView-ITCH 5.0 wire decoding.
//
// Format facts that bite people:
//
//  * BIG-ENDIAN. Every multi-byte integer. x86 is little-endian, so every
//    field needs a byte swap. `__builtin_bswap*` compiles to one `bswap` or
//    `movbe` instruction — free.
//
//  * UNALIGNED. Messages are packed with no padding and are not aligned to
//    anything. `*reinterpret_cast<const uint32_t*>(p + 11)` is undefined
//    behaviour even on x86 where it happens to work: the compiler is entitled
//    to assume alignment and vectorise accordingly, and on ARM it faults
//    outright. Always `memcpy` into a local; the compiler turns a 4-byte
//    memcpy from a known-size source into a single `mov`. Zero cost, defined
//    behaviour.
//
//  * SIX-BYTE TIMESTAMPS. Nanoseconds since midnight, as a 48-bit big-endian
//    integer. There is no 6-byte load instruction, so read 8 bytes starting 2
//    bytes earlier and mask, or read 4+2 and combine. Reading 8 bytes from
//    offset 5 would run past the end of the shortest message, so the safe
//    version composes a 4-byte and a 2-byte load.
//
//  * PRICES are uint32 with 4 implied decimals. $123.45 is 1234500. Keep them
//    as integers all the way through. Converting to double for storage is the
//    single most common way to produce a book that disagrees with the exchange
//    in the third decimal place after a few million messages.
//
//  * BinaryFILE framing: each message is preceded by a 2-byte big-endian
//    length. The MoldUDP64 wire feed does not have this framing; it has
//    packet headers with sequence numbers and message counts instead. This
//    file handles the BinaryFILE form, which is what the downloadable daily
//    samples use.
//
// The parser is zero-copy and non-allocating: it hands the handler a pointer
// into the mapped file plus decoded scalars, and never materialises a message
// struct it does not need.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#if defined(__linux__)
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace lob::itch {

// --- endian-safe unaligned loads -------------------------------------------

[[gnu::always_inline]] inline std::uint16_t rd_be16(const std::uint8_t* p) noexcept {
    std::uint16_t v;
    std::memcpy(&v, p, 2);
    return __builtin_bswap16(v);
}
[[gnu::always_inline]] inline std::uint32_t rd_be32(const std::uint8_t* p) noexcept {
    std::uint32_t v;
    std::memcpy(&v, p, 4);
    return __builtin_bswap32(v);
}
[[gnu::always_inline]] inline std::uint64_t rd_be64(const std::uint8_t* p) noexcept {
    std::uint64_t v;
    std::memcpy(&v, p, 8);
    return __builtin_bswap64(v);
}
// 48-bit big-endian, composed from a 4-byte and a 2-byte load so we never read
// past the end of a short message.
[[gnu::always_inline]] inline std::uint64_t rd_be48(const std::uint8_t* p) noexcept {
    return (static_cast<std::uint64_t>(rd_be32(p)) << 16) | rd_be16(p + 4);
}

inline void wr_be16(std::uint8_t* p, std::uint16_t v) noexcept {
    v = __builtin_bswap16(v); std::memcpy(p, &v, 2);
}
inline void wr_be32(std::uint8_t* p, std::uint32_t v) noexcept {
    v = __builtin_bswap32(v); std::memcpy(p, &v, 4);
}
inline void wr_be64(std::uint8_t* p, std::uint64_t v) noexcept {
    v = __builtin_bswap64(v); std::memcpy(p, &v, 8);
}
inline void wr_be48(std::uint8_t* p, std::uint64_t v) noexcept {
    wr_be32(p, static_cast<std::uint32_t>(v >> 16));
    wr_be16(p + 4, static_cast<std::uint16_t>(v & 0xFFFF));
}

// --- message types and sizes -----------------------------------------------

enum : std::uint8_t {
    kSystemEvent      = 'S',
    kStockDirectory   = 'R',
    kTradingAction    = 'H',
    kAddOrder         = 'A',
    kAddOrderMPID     = 'F',
    kOrderExecuted    = 'E',
    kOrderExecutedPx  = 'C',
    kOrderCancel      = 'X',
    kOrderDelete      = 'D',
    kOrderReplace     = 'U',
    kTrade            = 'P',
    kCrossTrade       = 'Q',
    kBrokenTrade      = 'B',
};

// Body length by message type, EXCLUDING the 2-byte BinaryFILE length prefix.
// Zero means "unknown type" — the framing length prefix is authoritative and
// the parser skips such messages, which is what lets this code survive a
// future spec revision that adds a message type it has never seen.
inline constexpr std::uint16_t message_length(std::uint8_t type) noexcept {
    switch (type) {
        case kSystemEvent:     return 12;
        case kStockDirectory:  return 39;
        case kTradingAction:   return 25;
        case kAddOrder:        return 36;
        case kAddOrderMPID:    return 40;
        case kOrderExecuted:   return 31;
        case kOrderExecutedPx: return 36;
        case kOrderCancel:     return 23;
        case kOrderDelete:     return 19;
        case kOrderReplace:    return 35;
        case kTrade:           return 44;
        case kCrossTrade:      return 40;
        case kBrokenTrade:     return 19;
        default:               return 0;
    }
}

// Common header: type(1) locate(2) tracking(2) timestamp(6) = 11 bytes.
inline constexpr std::size_t kHeaderLen = 11;

// --- decoded views (no copies of the payload) ------------------------------

struct AddOrder {
    std::uint64_t ref;
    std::uint32_t shares;
    std::uint32_t price;      // 4 implied decimals
    std::uint16_t locate;
    bool          is_buy;
};

struct Executed {
    std::uint64_t ref;
    std::uint64_t match;
    std::uint32_t shares;
};

struct Cancel {
    std::uint64_t ref;
    std::uint32_t shares;
};

struct Delete {
    std::uint64_t ref;
};

struct Replace {
    std::uint64_t old_ref;
    std::uint64_t new_ref;
    std::uint32_t shares;
    std::uint32_t price;
};

// --- field accessors, offsets straight from the spec -----------------------

[[gnu::always_inline]] inline std::uint8_t  msg_type(const std::uint8_t* p) noexcept { return p[0]; }
[[gnu::always_inline]] inline std::uint16_t msg_locate(const std::uint8_t* p) noexcept { return rd_be16(p + 1); }
[[gnu::always_inline]] inline std::uint64_t msg_timestamp(const std::uint8_t* p) noexcept { return rd_be48(p + 5); }

[[gnu::always_inline]] inline AddOrder decode_add(const std::uint8_t* p) noexcept {
    return AddOrder{
        rd_be64(p + 11),                    // order reference
        rd_be32(p + 20),                    // shares
        rd_be32(p + 32),                    // price
        rd_be16(p + 1),                     // stock locate
        p[19] == 'B'                        // buy/sell indicator
    };
}
[[gnu::always_inline]] inline Executed decode_executed(const std::uint8_t* p) noexcept {
    return Executed{rd_be64(p + 11), rd_be64(p + 23), rd_be32(p + 19)};
}
[[gnu::always_inline]] inline Cancel decode_cancel(const std::uint8_t* p) noexcept {
    return Cancel{rd_be64(p + 11), rd_be32(p + 19)};
}
[[gnu::always_inline]] inline Delete decode_delete(const std::uint8_t* p) noexcept {
    return Delete{rd_be64(p + 11)};
}
[[gnu::always_inline]] inline Replace decode_replace(const std::uint8_t* p) noexcept {
    return Replace{rd_be64(p + 11), rd_be64(p + 19), rd_be32(p + 27), rd_be32(p + 31)};
}

// Stock symbol: 8 bytes, right-padded with spaces.
inline std::string decode_stock(const std::uint8_t* p, std::size_t off) {
    std::string s(reinterpret_cast<const char*>(p + off), 8);
    while (!s.empty() && s.back() == ' ') s.pop_back();
    return s;
}

// --- mapped file -----------------------------------------------------------
//
// mmap rather than read(): the file is 1-10 GB for a real trading day and the
// point of the exercise is deterministic replay, which means reading it many
// times. mmap lets the page cache do the work and costs no user-space copy.
//
// MADV_SEQUENTIAL tells the kernel to read ahead aggressively and drop pages
// behind us, which keeps a 10 GB replay from evicting everything else on the
// box. MADV_HUGEPAGE is not useful here — file-backed mappings do not get THP.

class MappedFile {
public:
    explicit MappedFile(const std::string& path) {
#if defined(__linux__)
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) return;
        struct stat st{};
        if (::fstat(fd_, &st) != 0 || st.st_size <= 0) { ::close(fd_); fd_ = -1; return; }
        size_ = static_cast<std::size_t>(st.st_size);
        void* m = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (m == MAP_FAILED) { ::close(fd_); fd_ = -1; size_ = 0; return; }
        data_ = static_cast<const std::uint8_t*>(m);
        ::madvise(const_cast<std::uint8_t*>(data_), size_, MADV_SEQUENTIAL);
#else
        (void)path;
#endif
    }
    ~MappedFile() {
#if defined(__linux__)
        if (data_) ::munmap(const_cast<std::uint8_t*>(data_), size_);
        if (fd_ >= 0) ::close(fd_);
#endif
    }
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    bool                 ok()   const noexcept { return data_ != nullptr; }
    const std::uint8_t*  data() const noexcept { return data_; }
    std::size_t          size() const noexcept { return size_; }

    // Pull every page into the page cache before timing anything. Without
    // this the first replay measures disk, not code.
    void prefault() const noexcept {
        volatile std::uint64_t sum = 0;
        for (std::size_t i = 0; i < size_; i += 4096) sum += data_[i];
        (void)sum;
    }

private:
    int                 fd_{-1};
    const std::uint8_t* data_{nullptr};
    std::size_t         size_{0};
};

// --- the parse loop --------------------------------------------------------
//
// Templated on the handler so the dispatch is inlined — the same argument as
// the previous project, applied where it actually pays: this switch runs once
// per message, tens of millions of times per day.
//
// Returns the number of messages consumed. `truncated` is set if the file ends
// mid-message, which is normal for a capture that was cut off and should be
// reported rather than silently ignored.
template <class Handler>
std::size_t parse(const std::uint8_t* data, std::size_t size, Handler& h,
                  bool* truncated = nullptr) {
    std::size_t off = 0;
    std::size_t count = 0;

    while (off + 2 <= size) {
        const std::uint16_t len = rd_be16(data + off);
        if (len == 0) break;
        if (off + 2 + len > size) { if (truncated) *truncated = true; break; }

        const std::uint8_t* m = data + off + 2;
        const std::uint8_t  t = msg_type(m);

        // The framing length is authoritative; the table is a cross-check.
        // A mismatch means either a corrupt file or a spec version this build
        // does not know about, and both deserve to be loud.
        const std::uint16_t expect = message_length(t);
        if (expect != 0 && expect != len) h.on_length_mismatch(t, len, expect);

        switch (t) {
            case kAddOrder:
            case kAddOrderMPID:    h.on_add(decode_add(m), m);         break;
            case kOrderExecuted:   h.on_executed(decode_executed(m), m, false); break;
            case kOrderExecutedPx: h.on_executed(decode_executed(m), m, true);  break;
            case kOrderCancel:     h.on_cancel(decode_cancel(m), m);   break;
            case kOrderDelete:     h.on_delete(decode_delete(m), m);   break;
            case kOrderReplace:    h.on_replace(decode_replace(m), m); break;
            default:               h.on_other(t, m, len);              break;
        }

        off += 2u + len;
        ++count;
    }
    return count;
}

}  // namespace lob::itch

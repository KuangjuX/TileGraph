#pragma once
#include <string>
#include <vector>

namespace tilegraph {

    struct Platform {
        using underlying_t = uint16_t;

        enum : underlying_t { CUDA, BANG } type;

        constexpr Platform(decltype(type) t) : type(t) {}
        constexpr explicit Platform(underlying_t val)
            : type((decltype(type))val) {}
        constexpr underlying_t underlying() const { return type; }

        bool operator==(Platform others) const { return type == others.type; }
        bool operator!=(Platform others) const { return type != others.type; }

        const char *toString() const;
        bool isCUDA() const;
        bool isBANG() const;
    };

}  // namespace tilegraph
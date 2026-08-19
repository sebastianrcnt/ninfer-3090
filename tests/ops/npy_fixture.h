#pragma once

// Minimal reader for the C-order .npy arrays produced by tools/dflash2 oracle dumps.
// Only the dtypes those dumps use are accepted; anything else is a hard error rather
// than a silent reinterpretation.

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::test {

struct NpyArray {
    std::vector<std::size_t> shape;
    std::vector<float> f32;
    std::vector<std::int64_t> i64;
    bool is_integer = false;

    [[nodiscard]] std::size_t count() const {
        std::size_t total = 1;
        for (std::size_t dimension : shape) { total *= dimension; }
        return total;
    }
};

inline NpyArray read_npy(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("cannot open " + path); }

    char magic[6] = {};
    file.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error(path + ": not a .npy file");
    }
    std::uint8_t major = 0;
    std::uint8_t minor = 0;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);
    std::size_t header_length = 0;
    if (major == 1) {
        std::uint16_t length = 0;
        file.read(reinterpret_cast<char*>(&length), 2);
        header_length = length;
    } else {
        std::uint32_t length = 0;
        file.read(reinterpret_cast<char*>(&length), 4);
        header_length = length;
    }
    std::string header(header_length, '\0');
    file.read(header.data(), static_cast<std::streamsize>(header_length));

    if (header.find("'fortran_order': False") == std::string::npos) {
        throw std::runtime_error(path + ": only C-order arrays are read");
    }
    NpyArray out;
    if (header.find("'<f4'") != std::string::npos) {
        out.is_integer = false;
    } else if (header.find("'<i8'") != std::string::npos) {
        out.is_integer = true;
    } else {
        throw std::runtime_error(path + ": unsupported dtype");
    }

    const std::size_t open = header.find('(', header.find("'shape'"));
    const std::size_t close = header.find(')', open);
    std::string dims = header.substr(open + 1, close - open - 1);
    std::string digits;
    for (char character : dims + ",") {
        if (character >= '0' && character <= '9') {
            digits.push_back(character);
        } else if (character == ',' && !digits.empty()) {
            out.shape.push_back(static_cast<std::size_t>(std::stoull(digits)));
            digits.clear();
        }
    }

    const std::size_t count = out.count();
    if (out.is_integer) {
        out.i64.resize(count);
        file.read(reinterpret_cast<char*>(out.i64.data()),
                  static_cast<std::streamsize>(count * sizeof(std::int64_t)));
    } else {
        out.f32.resize(count);
        file.read(reinterpret_cast<char*>(out.f32.data()),
                  static_cast<std::streamsize>(count * sizeof(float)));
    }
    if (!file) { throw std::runtime_error(path + ": truncated payload"); }
    return out;
}

} // namespace ninfer::test

#pragma once
#include <tools/config.h>
#include <runtime/image.h>
#include <runtime/volume.h>
#include <runtime/command_buffer.h>
namespace luisa::compute {
class IBinaryStream;
class LC_TOOL_API ImageLib {
public:
    using WriteFunc = luisa::move_only_function<void(luisa::span<std::byte const> data)>;

private:
    ImageLib() = delete;
    static Image<float> load_float_image(IBinaryStream *bin_stream, Device &device, CommandBuffer &cmd_buffer);
    static Image<int> load_int_image(IBinaryStream *bin_stream, Device &device, CommandBuffer &cmd_buffer);
    static Image<uint32_t> load_uint_image(IBinaryStream *bin_stream, Device &device, CommandBuffer &cmd_buffer);
    static Volume<float> load_float_volume(IBinaryStream *bin_stream, Device &device, CommandBuffer &cmd_buffer);
    static Volume<int> load_int_volume(IBinaryStream *bin_stream, Device &device, CommandBuffer &cmd_buffer);
    static Volume<uint32_t> load_uint_volume(IBinaryStream *bin_stream, Device &device, CommandBuffer &cmd_buffer);
    static size_t header_size();
    static void save_header(
        luisa::span<std::byte> data,
        uint width, uint height, uint volume,
        PixelStorage storage,
        uint mip);

public:
    template<typename T>
        requires(is_legal_image_element<T>)
    static Image<T> load_image(IBinaryStream *bin_stream, Device &device, CommandBuffer &cmd_buffer) {
        if constexpr (std::is_same_v<T, float>) {
            return load_float_image(bin_stream, device, cmd_buffer);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return load_int_image(bin_stream, device, cmd_buffer);
        } else {
            return load_uint_image(bin_stream, device, cmd_buffer);
        }
    }
    template<typename T>
        requires(is_legal_image_element<T>)
    static Volume<T> load_volume(IBinaryStream *bin_stream, Device &device, CommandBuffer &cmd_buffer) {
        if constexpr (std::is_same_v<T, float>) {
            return load_float_image(bin_stream, device, cmd_buffer);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return load_int_image(bin_stream, device, cmd_buffer);
        } else {
            return load_uint_image(bin_stream, device, cmd_buffer);
        }
    }
    template<typename T>
    static void save_image(Image<T> const &image, CommandBuffer &cmd_buffer, WriteFunc &&func) {
        luisa::vector<std::byte> bytes;
        auto header = header_size();
        bytes.push_back_uninitialized(header + image.byte_size());
        cmd_buffer << image.copy_to(bytes.data() + header) << [func = std::move(func), bytes = std::move(bytes), size = image.size(), storage = image.storage(), mip = image.mip_levels()] {
            save_header(bytes, size.x, size.y, 1, storage, mip);
        };
    }
    template<typename T>
    static void save_volume(Volume<T> const &image, CommandBuffer &cmd_buffer, WriteFunc &&func) {
        luisa::vector<std::byte> bytes;
        auto header = header_size();
        bytes.push_back_uninitialized(header + image.byte_size());
        cmd_buffer << image.copy_to(bytes.data() + header) << [func = std::move(func), bytes = std::move(bytes), size = image.size(), storage = image.storage(), mip = image.mip_levels()] {
            save_header(bytes, size.x, size.y, size.z, storage, mip);
        };
    }
};
}// namespace luisa::compute
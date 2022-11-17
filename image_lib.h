#pragma once
#include <tools/config.h>
#include <runtime/image.h>
#include <runtime/volume.h>
#include <runtime/command_buffer.h>
#include <runtime/shader.h>
namespace luisa::compute {
class IBinaryStream;
class LC_TOOL_API ImageLib {
public:
    using WriteFunc = luisa::move_only_function<void(luisa::span<std::byte const> data)>;

private:
    Device _device;
    Shader2D<Image<float>, Image<float>> _mip1_shader;
    Shader2D<Image<float>, Image<float>, Image<float>> _mip2_shader;
    Shader2D<Image<float>, Image<float>, Image<float>, Image<float>> _mip3_shader;
    Shader2D<Image<float>, Image<float>, Image<float>, Image<float>, Image<float>> _mip4_shader;
    Shader2D<Image<float>, Image<float>, Image<float>, Image<float>, Image<float>, Image<float>> _mip5_shader;
    ImageLib() = delete;
    Image<float> load_float_image(IBinaryStream *bin_stream, CommandBuffer &cmd_buffer);
    Image<int> load_int_image(IBinaryStream *bin_stream, CommandBuffer &cmd_buffer);
    Image<uint32_t> load_uint_image(IBinaryStream *bin_stream, CommandBuffer &cmd_buffer);
    Volume<float> load_float_volume(IBinaryStream *bin_stream, CommandBuffer &cmd_buffer);
    Volume<int> load_int_volume(IBinaryStream *bin_stream, CommandBuffer &cmd_buffer);
    Volume<uint32_t> load_uint_volume(IBinaryStream *bin_stream, CommandBuffer &cmd_buffer);
    size_t header_size();
    void save_header(
        luisa::span<std::byte> data,
        uint width, uint height, uint volume,
        PixelStorage storage,
        uint mip);

public:
    ImageLib(Device device, luisa::string shader_dir);
    ImageLib(ImageLib const &) = delete;
    ImageLib(ImageLib &&) = delete;
    template<typename T>
        requires(is_legal_image_element<T>)
    Image<T> load_image(IBinaryStream *bin_stream, CommandBuffer &cmd_buffer) {
        if constexpr (std::is_same_v<T, float>) {
            return load_float_image(bin_stream, cmd_buffer);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return load_int_image(bin_stream, cmd_buffer);
        } else {
            return load_uint_image(bin_stream, cmd_buffer);
        }
    }
    template<typename T>
        requires(is_legal_image_element<T>)
    Volume<T> load_volume(IBinaryStream *bin_stream, CommandBuffer &cmd_buffer) {
        if constexpr (std::is_same_v<T, float>) {
            return load_float_image(bin_stream, cmd_buffer);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return load_int_image(bin_stream, cmd_buffer);
        } else {
            return load_uint_image(bin_stream, cmd_buffer);
        }
    }
    template<typename T>
    void save_image(Image<T> const &image, CommandBuffer &cmd_buffer, WriteFunc &&func) {
        luisa::vector<std::byte> bytes;
        auto header = header_size();
        bytes.push_back_uninitialized(header + image.byte_size());
        cmd_buffer << image.copy_to(bytes.data() + header) << [this, func = std::move(func), bytes = std::move(bytes), size = image.size(), storage = image.storage(), mip = image.mip_levels()] {
            save_header(bytes, size.x, size.y, 1, storage, mip);
            func(bytes);
        };
    }
    template<typename T>
    void save_volume(Volume<T> const &image, CommandBuffer &cmd_buffer, WriteFunc &&func) {
        luisa::vector<std::byte> bytes;
        auto header = header_size();
        bytes.push_back_uninitialized(header + image.byte_size());
        cmd_buffer << image.copy_to(bytes.data() + header) << [this, func = std::move(func), bytes = std::move(bytes), size = image.size(), storage = image.storage(), mip = image.mip_levels()] {
            save_header(bytes, size.x, size.y, size.z, storage, mip);
            func(bytes);
        };
    }
    Image<float> read_ldr(luisa::string const &file_name, CommandBuffer &cmd_buffer, uint mip_level);
    Image<float> read_hdr(luisa::string const &file_name, CommandBuffer &cmd_buffer, uint mip_level);
    void generate_mip(Image<float> const &img, CommandBuffer &cmd_buffer);
};
}// namespace luisa::compute
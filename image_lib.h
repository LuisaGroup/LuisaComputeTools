#pragma once
#include <tools/config.h>
#include <runtime/image.h>
#include <runtime/volume.h>
#include <runtime/command_buffer.h>
#include <runtime/shader.h>
#include <runtime/bindless_array.h>
#include <vstl/common.h>
#include <vstl/functional.h>
#include <filesystem>

namespace luisa::compute {
class IBinaryStream;
namespace detail {
template<size_t i, template<typename...> typename Collection, typename T, typename... Ts>
static constexpr decltype(auto) TypeAccumulator() {
    if constexpr (i > 0) {
        return TypeAccumulator<i - 1, Collection, T, T, Ts...>();
    } else {
        return vstd::TypeOf<Collection<T, Ts...>>{};
    }
}
}// namespace detail
class LC_TOOL_API ImageLib {
public:
    using WriteFunc = luisa::move_only_function<void(luisa::span<std::byte const> data)>;

private:
    Device _device;

    template<typename... T>
    struct ShaderOptional2D {
        using value_type = vstd::optional<Shader2D<T...>>;
        value_type value;
        vstd::function<void(value_type &)> init_func;
        Shader2D<T...> &operator*() {
            if (!value) {
                init_func(value);
            }
            return *value;
        }
        Shader2D<T...> *operator->() {
            if (!value) {
                init_func(value);
            }
            return value.GetPtr();
        }
    };
    template<typename T, size_t i>
    using MipgenType = typename decltype(detail::TypeAccumulator<i, ShaderOptional2D, T>())::Type;
    std::filesystem::path _path;
    MipgenType<Image<float>, 1> _mip1_shader;
    MipgenType<Image<float>, 2> _mip2_shader;
    MipgenType<Image<float>, 3> _mip3_shader;
    MipgenType<Image<float>, 4> _mip4_shader;
    MipgenType<Image<float>, 5> _mip5_shader;
    // src_tex, output_texture, roughness
    ShaderOptional2D<Image<float>, float2, Image<float>, float> _refl_map_gen;
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
        auto ptr = bytes.data() + header;
        for (auto i : vstd::range(image.mip_levels())) {
            auto view = image.view(i);
            cmd_buffer << view.copy_to(ptr);
            ptr += view.byte_size();
        }

        cmd_buffer << [this, func = std::move(func), bytes = std::move(bytes), size = image.size(), storage = image.storage(), mip = image.mip_levels()]() mutable {
            save_header(bytes, size.x, size.y, 1, storage, mip);
            func(bytes);
        };
    }
    template<typename T>
    void save_volume(Volume<T> const &image, CommandBuffer &cmd_buffer, WriteFunc &&func) {
        luisa::vector<std::byte> bytes;
        auto header = header_size();
        bytes.push_back_uninitialized(header + image.byte_size());
        auto ptr = bytes.data() + header;
        for (auto i : vstd::range(image.mip_levels())) {
            auto view = image.view(i);
            cmd_buffer << view.copy_to(ptr);
            ptr += view.byte_size();
        }
        cmd_buffer << [this, func = std::move(func), bytes = std::move(bytes), size = image.size(), storage = image.storage(), mip = image.mip_levels()]() mutable {
            save_header(bytes, size.x, size.y, size.z, storage, mip);
            func(bytes);
        };
    }
    Image<float> read_ldr(luisa::string const &file_name, CommandBuffer &cmd_buffer, uint mip_level);
    Image<float> read_hdr(luisa::string const &file_name, CommandBuffer &cmd_buffer, uint mip_level);
    Image<float> read_exr(luisa::string const &file_name, CommandBuffer &cmd_buffer, uint mip_level);
    void gen_cubemap_mip(CommandBuffer &cmd_buffer, ImageView<float> const &src, ImageView<float> const &dst, float roughness);
    void generate_mip(Image<float> const &img, CommandBuffer &cmd_buffer);
};
}// namespace luisa::compute
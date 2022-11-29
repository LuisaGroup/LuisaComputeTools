#include <tools/image_lib.h>
#include <core/binary_io_visitor.h>
#include <stb/stb_image.h>
#include <dsl/syntax.h>
#include <dsl/sugar.h>
#include <tinyexr.h>
#include <core/logging.h>
namespace luisa::compute {
size_t img_byte_size(PixelStorage storage, uint32_t width, uint32_t height, uint32_t volume, uint32_t mip_level) noexcept {
    size_t size = 0;
    for (auto i : vstd::range(mip_level)) {
        size += pixel_storage_size(storage, width, height, volume);
        width /= 2;
        height /= 2;
    }
    return size;
}

namespace imglib_detail {
struct ImageHeader {
    uint32_t width, height, mip_level, volume;
    PixelStorage storage;
};
template<typename T>
Image<T> load_image_impl(IBinaryStream *bin_stream, Device &device, CommandBuffer &cmd_buffer) noexcept {
    ImageHeader header;
    bin_stream->read({reinterpret_cast<std::byte *>(&header), sizeof(ImageHeader)});
    auto byte_size = img_byte_size(header.storage, header.width, header.height, 1, header.mip_level);
    vstd::vector<std::byte> data;
    data.push_back_uninitialized(byte_size);
    bin_stream->read(data);
    auto img = device.create_image<T>(header.storage, header.width, header.height, header.mip_level);
    auto ptr = data.data();
    for (auto i : vstd::range(header.mip_level)) {
        auto view = img.view(i);
        cmd_buffer << view.copy_from(ptr);
        ptr += view.byte_size();
    }
    cmd_buffer << [data = std::move(data)] {};
    return img;
}
template<typename T>
Volume<T> load_volume_impl(IBinaryStream *bin_stream, Device &device, CommandBuffer &cmd_buffer) noexcept {
    ImageHeader header;
    bin_stream->read({reinterpret_cast<std::byte *>(&header), sizeof(ImageHeader)});
    auto byte_size = img_byte_size(header.storage, header.width, header.height, header.volume, header.mip_level);
    vstd::vector<std::byte> data;
    data.push_back_uninitialized(byte_size);
    bin_stream->read(data);
    auto img = device.create_volume<T>(header.storage, header.width, header.height, header.mip_level);
    auto ptr = data.data();
    for (auto i : vstd::range(header.mip_level)) {
        auto view = img.view(i);
        cmd_buffer << view.copy_from(ptr);
        ptr += view.byte_size();
    }
    cmd_buffer << [data = std::move(data)] {};
    return img;
}

}// namespace imglib_detail
Image<float> ImageLib::load_float_image(IBinaryStream *bin_stream, CommandBuffer &cmd_buffer) noexcept {
    return imglib_detail::load_image_impl<float>(bin_stream, _device, cmd_buffer);
}
Image<int32_t> ImageLib::load_int_image(IBinaryStream *bin_stream, CommandBuffer &cmd_buffer) noexcept {
    return imglib_detail::load_image_impl<int32_t>(bin_stream, _device, cmd_buffer);
}
Image<uint32_t> ImageLib::load_uint_image(IBinaryStream *bin_stream, CommandBuffer &cmd_buffer) noexcept {
    return imglib_detail::load_image_impl<uint32_t>(bin_stream, _device, cmd_buffer);
}
Volume<float> ImageLib::load_float_volume(IBinaryStream *bin_stream, CommandBuffer &cmd_buffer) noexcept {
    return imglib_detail::load_volume_impl<float>(bin_stream, _device, cmd_buffer);
}
Volume<int32_t> ImageLib::load_int_volume(IBinaryStream *bin_stream, CommandBuffer &cmd_buffer) noexcept {
    return imglib_detail::load_volume_impl<int32_t>(bin_stream, _device, cmd_buffer);
}
Volume<uint32_t> ImageLib::load_uint_volume(IBinaryStream *bin_stream, CommandBuffer &cmd_buffer) noexcept {
    return imglib_detail::load_volume_impl<uint32_t>(bin_stream, _device, cmd_buffer);
}
size_t ImageLib::header_size() noexcept {
    return sizeof(imglib_detail::ImageHeader);
}
void ImageLib::save_header(
    luisa::span<std::byte> data,
    uint32_t width, uint32_t height, uint32_t volume,
    PixelStorage storage,
    uint32_t mip) noexcept {
    imglib_detail::ImageHeader header{
        .width = width,
        .height = height,
        .mip_level = mip,
        .volume = volume,
        .storage = storage};
    memcpy(data.data(), &header, sizeof(imglib_detail::ImageHeader));
}
Image<float> ImageLib::read_ldr(luisa::string const &file_name, CommandBuffer &cmd_buffer, uint32_t mip_level) noexcept {
    int32_t x, y, channel;
    auto ptr = stbi_load(file_name.c_str(), &x, &y, &channel, 4);
    auto img = _device.create_image<float>(PixelStorage::BYTE4, x, y, mip_level);
    cmd_buffer << img.copy_from(ptr) << [ptr] {
        stbi_image_free(ptr);
    };
    if (mip_level > 1) {
        generate_mip(img, cmd_buffer);
    }
    return img;
}
Image<float> ImageLib::read_hdr(luisa::string const &file_name, CommandBuffer &cmd_buffer, uint32_t mip_level) noexcept {
    int32_t x, y, channel;
    auto ptr = stbi_loadf(file_name.c_str(), &x, &y, &channel, 4);
    auto img = _device.create_image<float>(PixelStorage::FLOAT4, x, y, mip_level);
    cmd_buffer << img.copy_from(ptr) << [ptr] {
        stbi_image_free(ptr);
    };
    if (mip_level > 1) {
        generate_mip(img, cmd_buffer);
    }
    return img;
}
Image<float> ImageLib::read_exr(luisa::string const &file_name, CommandBuffer &cmd_buffer, uint32_t mip_level) noexcept {
    float *ptr;
    int32_t width, height;
    char const *err;
    if (LoadEXR(&ptr, &width, &height, file_name.c_str(), &err) < 0) {
        LUISA_ERROR("Load EXR Error: {}", err);
    }
    auto img = _device.create_image<float>(PixelStorage::FLOAT4, width, height, mip_level);
    cmd_buffer << img.copy_from(ptr) << [ptr] {
        free(ptr);
    };
    if (mip_level > 1) {
        generate_mip(img, cmd_buffer);
    }
    return img;
}
Image<float> ImageLib::read_exr_cubemap(luisa::string const &file_name, CommandBuffer &cmd_buffer, uint32_t mip_level, float roughness) noexcept {
    float *ptr;
    int32_t width, height;
    char const *err;
    if (LoadEXR(&ptr, &width, &height, file_name.c_str(), &err) < 0) {
        LUISA_ERROR("Load EXR Error: {}", err);
    }
    auto img = _device.create_image<float>(PixelStorage::FLOAT4, width, height, mip_level);
    cmd_buffer << img.copy_from(ptr) << [ptr] {
        free(ptr);
    };
    if (mip_level > 1) {
        for (auto &&i : vstd::range(1, mip_level)) {
            auto src_view = img.view(i - 1);
            auto dst_view = img.view(i);
            auto rate = float(i) / (mip_level - 1);
            auto rough = 1 * (1 - rate) + roughness * rate;
            cmd_buffer << (*_refl_map_gen)(src_view, make_float2(src_view.size()), dst_view, rough).dispatch(dst_view.size());
        }
    }
    return img;
}

namespace imglib_detail {
decltype(auto) mip_syncblock_func(uint32_t block_count) noexcept {
    return [=](UInt &ava_thread_count, Float4 &tex_value, ImageVar<float> img) {
        auto sample_2d = [](Expr<uint2> uv, Expr<uint32_t> width) {
            return uv.y * width + uv.x;
        };
        UInt2 local_coord = thread_id().xy();
        Shared<float4> shared_floats{block_count * block_count};
        $if(all(local_coord < make_uint2(ava_thread_count))) {
            shared_floats.write(sample_2d(local_coord, ava_thread_count), tex_value);
        };
        sync_block();
        auto next_thread_count = ava_thread_count / 2;
        $if(all(local_coord < make_uint2(next_thread_count))) {
            auto sample_local = local_coord * make_uint2(2);
            tex_value =
                shared_floats.read(sample_2d(sample_local, ava_thread_count)) +
                shared_floats.read(sample_2d(sample_local + make_uint2(1, 0), ava_thread_count)) +
                shared_floats.read(sample_2d(sample_local + make_uint2(0, 1), ava_thread_count)) +
                shared_floats.read(sample_2d(sample_local + make_uint2(1, 1), ava_thread_count));
            tex_value *= make_float4(0.25f);
            img.write(block_id().xy() * make_uint2(next_thread_count) + local_coord, tex_value);
        };
        ava_thread_count = next_thread_count;
    };
}
template<typename... Args>
static void gen_mip_func(vstd::optional<Shader2D<Args...>> &shader, Device &device, uint32_t mip_level, std::filesystem::path const &dir) noexcept {
    uint32_t block_count = 1 << mip_level;
    Callable mip_group = mip_syncblock_func(block_count);
    Kernel2D k = [&](Var<Args>... imgs) {
        set_block_size(block_count, block_count);
        auto local_coord = thread_id().xy();
        UInt ava_thread_count = block_count;
        auto coord = dispatch_id().xy();
        ImageVar<float> *img_arr[] = {(&imgs)...};
        Float4 tex_value = img_arr[0]->read(coord);
        for (uint32_t i = 0; i < mip_level; ++i) {
            mip_group(ava_thread_count, tex_value, *img_arr[i + 1]);
        }
    };
    luisa::string file_name = "__gen_mip";
    file_name += vstd::to_string(mip_level);
    auto path = (dir / file_name).string<char, std::char_traits<char>, luisa::allocator<char>>();
    shader.New(device.compile_to(k, path));
}
static UInt reverse_bits(UInt bits) noexcept {
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x00ff00ff) << 8) | ((bits & 0xff00ff00) >> 8);
    bits = ((bits & 0x0f0f0f0f) << 4) | ((bits & 0xf0f0f0f0) >> 4);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xcccccccc) >> 2);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xaaaaaaaa) >> 1);
    return bits;
}
static Float2 hammersley(UInt const &Index, UInt const &NumSamples) noexcept {
    return make_float2((Index.cast<float>() + 0.5f) / NumSamples.cast<float>(), (reverse_bits(Index).cast<float>() / 0xffffffffu));
}
constexpr float pi = 3.141592653589793f;

Float3 ImportanceSampleGGX(Float3 const &N, Float2 const &E, Float const &Roughness) noexcept {
    auto m = Roughness * Roughness;

    auto Phi = 2.0f * pi * E.x;
    auto CosTheta = sqrt((1.0f - E.y) / (1.0f + (m * m - 1.0f) * E.y));
    auto SinTheta = sqrt(1.0f - CosTheta * CosTheta);

    // from spherical coordinates to cartesian coordinates - halfway vector
    Float3 H = Float3(SinTheta * cos(Phi), SinTheta * sin(Phi), CosTheta);

    // from tangent-space H vector to world-space sample vector
    auto UpVector = select(select(float3{0, 1, 0}, float3{1, 0, 0}, abs(N.x) < 0.7f), float3{0, 0, 1}, abs(N.z) < 0.7f);
    auto TangentX = normalize(cross(UpVector, N));
    auto TangentY = cross(N, TangentX);

    return normalize(TangentX * H.x + TangentY * H.y + N * H.z);
}
static Float2 DirToUv(Float3 w) noexcept {
    Float theta = acos(w.y);
    Float phi = atan2(w.x, w.z);
    return fract(make_float2(1.f - 0.5f * inv_pi * phi, theta * inv_pi - 1.0f));
}
static Float3 UvToDir(Float2 uv) noexcept {
    uv.x = 1.0f - uv.x;
    Float phi = 2.f * pi * uv.x;
    Float theta = pi * uv.y;
    Float sin_theta = sin(theta);
    return normalize(make_float3(sin(phi) * sin_theta, cos(theta), cos(phi) * sin_theta));
};

static Float3 refl(ImageVar<float> const &tex, Float2 const &img_size, Float3 const &sampleDir, Float const &roughness) noexcept {
    const uint32_t spp = 65536;
    Float3 result = make_float3(0.0f);
    for (auto i : range(spp)) {
        auto rand = hammersley(i, spp);
        auto dir = ImportanceSampleGGX(sampleDir, rand, roughness);
        auto uv = DirToUv(dir) * img_size;
        result += clamp(tex.read(make_uint2(uv)).xyz(), float3{0.0f}, float3{256.0f}) * make_float3(1.0 / spp);
    }
    return result;
}
static void refl_cubegen(ImageVar<float> read_img, Float2 img_size, ImageVar<float> out_img, Float roughness) {
    auto coord = dispatch_id().xy();
    auto uv = (make_float2(coord) + float2{0.5f}) / make_float2(dispatch_size().xy());
    auto dir = UvToDir(uv);
    auto color = refl(read_img, img_size, dir, roughness);
    out_img.write(coord, make_float4(color, 1.0f));
};
}// namespace imglib_detail
ImageLib::ImageLib(Device device, luisa::string shader_dir) noexcept : _device(std::move(device)), _path(shader_dir) {
    using namespace imglib_detail;
    std::filesystem::path path(shader_dir);
    _mip1_shader.init_func = [this](auto &&opt) { gen_mip_func(opt, _device, 1, _path); };
    _mip2_shader.init_func = [this](auto &&opt) { gen_mip_func(opt, _device, 2, _path); };
    _mip3_shader.init_func = [this](auto &&opt) { gen_mip_func(opt, _device, 3, _path); };
    _mip4_shader.init_func = [this](auto &&opt) { gen_mip_func(opt, _device, 4, _path); };
    _mip5_shader.init_func = [this](auto &&opt) { gen_mip_func(opt, _device, 5, _path); };
    _refl_map_gen.init_func = [this](auto &&opt) {
        opt.New(_device.compile_to(Kernel2D{refl_cubegen}, "__refl_gen"));
    };
}
void ImageLib::generate_mip(Image<float> const &img, CommandBuffer &cmd_buffer) noexcept {
    switch (img.mip_levels()) {
        case 0:
        case 1:
            return;
        case 2:
            cmd_buffer << (*_mip1_shader)(img.view(0), img.view(1)).dispatch(img.size());
            return;
        case 3:
            cmd_buffer << (*_mip2_shader)(img.view(0), img.view(1), img.view(2)).dispatch(img.size());
            return;
        case 4:
            cmd_buffer << (*_mip3_shader)(img.view(0), img.view(1), img.view(2), img.view(3)).dispatch(img.size());
            return;
        case 5:
            cmd_buffer << (*_mip4_shader)(img.view(0), img.view(1), img.view(2), img.view(3), img.view(4)).dispatch(img.size());
            return;
        case 6:
            cmd_buffer << (*_mip5_shader)(img.view(0), img.view(1), img.view(2), img.view(3), img.view(4), img.view(5)).dispatch(img.size());
            return;
        default:
            LUISA_ERROR("Mip-level larger than 6 can-not be supported!");
            return;
    }
}
void ImageLib::generate_cubemap_mip(Image<float> const &img, CommandBuffer &cmd_buffer, float roughness) noexcept {
    auto mip_level = img.mip_levels();
    if (mip_level > 1) {
        for (auto &&i : vstd::range(1, mip_level)) {
            auto src_view = img.view(i - 1);
            auto dst_view = img.view(i);
            auto rate = float(i) / (mip_level - 1);
            auto rough = 1 * (1 - rate) + roughness * rate;
            cmd_buffer << (*_refl_map_gen)(src_view, make_float2(src_view.size()), dst_view, rough).dispatch(dst_view.size());
        }
    }
}

}// namespace luisa::compute
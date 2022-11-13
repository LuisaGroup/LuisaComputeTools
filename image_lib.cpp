#include <tools/image_lib.h>
#include <vstl/common.h>
#include <core/binary_io_visitor.h>
#include <stb/stb_image.h>
#include <dsl/syntax.h>
#include <dsl/sugar.h>
namespace luisa::compute {
size_t img_byte_size(PixelStorage storage, uint width, uint height, uint volume, uint mip_level) {
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
    uint width, height, mip_level, volume;
    PixelStorage storage;
};
template<typename T>
Image<T> load_image_impl(IBinaryStream *bin_stream, Device &device, CommandBuffer &cmd_buffer) {
    ImageHeader header;
    bin_stream->read({reinterpret_cast<std::byte *>(&header), sizeof(ImageHeader)});
    auto byte_size = img_byte_size(header.storage, header.width, header.height, 1, header.mip_level);
    vstd::vector<std::byte> data;
    data.push_back_uninitialized(byte_size);
    bin_stream->read(data);
    auto img = device.create_image<T>(header.storage, header.width, header.height, header.mip_level);
    cmd_buffer << img.copy_from(data.data()) << [data = std::move(data)] {};
    return img;
}
template<typename T>
Volume<T> load_volume_impl(IBinaryStream *bin_stream, Device &device, CommandBuffer &cmd_buffer) {
    ImageHeader header;
    bin_stream->read({reinterpret_cast<std::byte *>(&header), sizeof(ImageHeader)});
    auto byte_size = img_byte_size(header.storage, header.width, header.height, header.volume, header.mip_level);
    vstd::vector<std::byte> data;
    data.push_back_uninitialized(byte_size);
    bin_stream->read(data);
    auto img = device.create_volume<T>(header.storage, header.width, header.height, header.mip_level);
    cmd_buffer << img.copy_from(data.data()) << [data = std::move(data)] {};
    return img;
}

}// namespace imglib_detail
Image<float> ImageLib::load_float_image(IBinaryStream *bin_stream, CommandBuffer &cmd_buffer) {
    return imglib_detail::load_image_impl<float>(bin_stream, _device, cmd_buffer);
}
Image<int> ImageLib::load_int_image(IBinaryStream *bin_stream, CommandBuffer &cmd_buffer) {
    return imglib_detail::load_image_impl<int>(bin_stream, _device, cmd_buffer);
}
Image<uint> ImageLib::load_uint_image(IBinaryStream *bin_stream, CommandBuffer &cmd_buffer) {
    return imglib_detail::load_image_impl<uint>(bin_stream, _device, cmd_buffer);
}
Volume<float> ImageLib::load_float_volume(IBinaryStream *bin_stream, CommandBuffer &cmd_buffer) {
    return imglib_detail::load_volume_impl<float>(bin_stream, _device, cmd_buffer);
}
Volume<int> ImageLib::load_int_volume(IBinaryStream *bin_stream, CommandBuffer &cmd_buffer) {
    return imglib_detail::load_volume_impl<int>(bin_stream, _device, cmd_buffer);
}
Volume<uint> ImageLib::load_uint_volume(IBinaryStream *bin_stream, CommandBuffer &cmd_buffer) {
    return imglib_detail::load_volume_impl<uint>(bin_stream, _device, cmd_buffer);
}
size_t ImageLib::header_size() {
    return sizeof(imglib_detail::ImageHeader);
}
void ImageLib::save_header(
    luisa::span<std::byte> data,
    uint width, uint height, uint volume,
    PixelStorage storage,
    uint mip) {
    imglib_detail::ImageHeader header{
        .width = width,
        .height = height,
        .mip_level = mip,
        .volume = volume,
        .storage = storage};
    memcpy(data.data(), &header, sizeof(imglib_detail::ImageHeader));
}
Image<float> ImageLib::read_ldr(luisa::string const &file_name, CommandBuffer &cmd_buffer, uint mip_level) {
    int x, y, channel;
    auto ptr = stbi_load(file_name.c_str(), &x, &y, &channel, 4);
    auto img = _device.create_image<float>(PixelStorage::BYTE4, x, y, mip_level);
    cmd_buffer << img.copy_from(ptr) << [ptr] {
        stbi_image_free(ptr);
    };
    return img;
}
Image<float> ImageLib::read_hdr(luisa::string const &file_name, CommandBuffer &cmd_buffer, uint mip_level) {
    int x, y, channel;
    auto ptr = stbi_loadf(file_name.c_str(), &x, &y, &channel, 4);
    auto img = _device.create_image<float>(PixelStorage::FLOAT4, x, y, mip_level);
    cmd_buffer << img.copy_from(ptr) << [ptr] {
        stbi_image_free(ptr);
    };
    return img;
}

namespace imglib_detail {
decltype(auto) mip_syncblock_func(uint block_count) {
    return [=](UInt &ava_thread_count, Float4 &tex_value, ImageVar<float> img) {
        auto sample_2d = [](Expr<uint2> uv, Expr<uint> width) {
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
            img.write(block_id().xy() * make_uint2(ava_thread_count) + local_coord, tex_value);
        };
        ava_thread_count = next_thread_count;
    };
}
template<typename Shader>
struct GenMipFunc;
template<typename... Args>
struct GenMipFunc<Shader2D<Args...>> {
    using value_type = Shader2D<Args...>;
    static value_type get(Device &device, uint mip_level) {
        uint32_t block_count = 1 << mip_level;
        Callable mip_group = mip_syncblock_func(block_count);
        Kernel2D k = [&](Var<Args>... imgs) {
            set_block_size(block_count, block_count);
            auto local_coord = thread_id().xy();
            UInt ava_thread_count = block_count;
            auto coord = dispatch_id().xy();
            ImageVar<float> *img_arr[] = {(&imgs)...};
            Float4 tex_value = img_arr[0]->read(coord);
            for (uint32_t i = 0; i < mip_level + 1; ++i) {
                mip_group(ava_thread_count, tex_value, *img_arr[i + 1]);
            }
        };
        return device.compile(k);
    }
};
}// namespace imglib_detail
ImageLib::ImageLib(Device device) : _device(std::move(device)) {
    using namespace imglib_detail;
    mip1_shader = GenMipFunc<decltype(mip1_shader)>::get(device, 0);
    mip2_shader = GenMipFunc<decltype(mip2_shader)>::get(device, 1);
    mip3_shader = GenMipFunc<decltype(mip3_shader)>::get(device, 2);
    mip4_shader = GenMipFunc<decltype(mip4_shader)>::get(device, 3);
    mip5_shader = GenMipFunc<decltype(mip5_shader)>::get(device, 4);
    mip6_shader = GenMipFunc<decltype(mip6_shader)>::get(device, 5);
}
void ImageLib::generate_mip(Image<float> const &img, CommandBuffer &cmd_buffer) {
    switch (img.mip_levels()) {
        case 0:
        case 1:
            return;
        case 2:
            cmd_buffer << mip1_shader(img.view(0), img.view(1)).dispatch(img.size());
            return;
        case 3:
            cmd_buffer << mip2_shader(img.view(0), img.view(1), img.view(2)).dispatch(img.size());
            return;
        case 4:
            cmd_buffer << mip3_shader(img.view(0), img.view(1), img.view(2), img.view(3)).dispatch(img.size());
            return;
        case 5:
            cmd_buffer << mip4_shader(img.view(0), img.view(1), img.view(2), img.view(3), img.view(4)).dispatch(img.size());
            return;
        case 6:
            cmd_buffer << mip5_shader(img.view(0), img.view(1), img.view(2), img.view(3), img.view(4), img.view(5)).dispatch(img.size());
            return;
        case 7:
            cmd_buffer << mip6_shader(img.view(0), img.view(1), img.view(2), img.view(3), img.view(4), img.view(5), img.view(6)).dispatch(img.size());
            return;
        default:
            LUISA_ERROR("Mip-level larger than 7 can-not be supported!");
            return;
    }
}
}// namespace luisa::compute
#include <tools/image_lib.h>
#include <vstl/common.h>
#include <core/binary_io_visitor.h>
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
Image<float> ImageLib::load_float_image(IBinaryStream *bin_stream, Device &device, CommandBuffer &cmd_buffer) {
    return imglib_detail::load_image_impl<float>(bin_stream, device, cmd_buffer);
}
Image<int> ImageLib::load_int_image(IBinaryStream *bin_stream, Device &device, CommandBuffer &cmd_buffer) {
    return imglib_detail::load_image_impl<int>(bin_stream, device, cmd_buffer);
}
Image<uint> ImageLib::load_uint_image(IBinaryStream *bin_stream, Device &device, CommandBuffer &cmd_buffer) {
    return imglib_detail::load_image_impl<uint>(bin_stream, device, cmd_buffer);
}
Volume<float> ImageLib::load_float_volume(IBinaryStream *bin_stream, Device &device, CommandBuffer &cmd_buffer) {
    return imglib_detail::load_volume_impl<float>(bin_stream, device, cmd_buffer);
}
Volume<int> ImageLib::load_int_volume(IBinaryStream *bin_stream, Device &device, CommandBuffer &cmd_buffer) {
    return imglib_detail::load_volume_impl<int>(bin_stream, device, cmd_buffer);
}
Volume<uint> ImageLib::load_uint_volume(IBinaryStream *bin_stream, Device &device, CommandBuffer &cmd_buffer) {
    return imglib_detail::load_volume_impl<uint>(bin_stream, device, cmd_buffer);
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
}// namespace luisa::compute
#include <tools/image_lib.h>
#include <vstl/common.h>
#include <core/binary_io_visitor.h>
namespace luisa::compute {
namespace imglib_detail {
class ImageHeader {
    uint32_t width, height, volume, mip_level;
    PixelStorage storage;   
};
}// namespace imglib_detail
Image<float> ImageLib::load_float_image(IBinaryStream *io_visitor, Device &device, CommandBuffer &cmd_buffer) {
    return {};
}
Image<int> ImageLib::load_int_image(IBinaryStream *io_visitor, Device &device, CommandBuffer &cmd_buffer) {
    return {};

}
Image<uint32_t> ImageLib::load_uint_image(IBinaryStream *io_visitor, Device &device, CommandBuffer &cmd_buffer) {
    return {};
}
}// namespace luisa::compute
#pragma once
#include <tools/config.h>
#include <runtime/image.h>
#include <runtime/command_buffer.h>
namespace luisa::compute {
class IBinaryStream;
class LC_TOOL_API ImageLib {
    ImageLib() = delete;
    static Image<float> load_float_image(IBinaryStream *io_visitor, Device &device, CommandBuffer &cmd_buffer);
    static Image<int> load_int_image(IBinaryStream *io_visitor, Device &device, CommandBuffer &cmd_buffer);
    static Image<uint32_t> load_uint_image(IBinaryStream *io_visitor, Device &device, CommandBuffer &cmd_buffer);

public:
    template<typename T>
        requires(is_legal_image_element<T>)
    static Image<T> load_image(IBinaryStream *io_visitor, Device &device, CommandBuffer &cmd_buffer) {
        if constexpr (std::is_same_v<T, float>) {
            return load_float_image(io_visitor, device, cmd_buffer);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return load_int_image(io_visitor, device, cmd_buffer);
        } else {
            return load_uint_image(io_visitor, device, cmd_buffer);
        }
    }
};
}// namespace luisa::compute
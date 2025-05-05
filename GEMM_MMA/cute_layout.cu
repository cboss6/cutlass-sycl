// cutlass_nested_layout_extra_examples.cu
#include <iostream>
// #include <cute/shape.hpp>
#include <cute/layout.hpp>

using namespace cute;

int main() {
  //
  // Example A: plain 2-D row-major [4,8]
  //
  {
    Shape<4,8> shape{};
    auto layout = make_layout<Stride<8,1>>(shape);

    std::cout << "Example A: shape [4,8], strides = ["
              << layout.stride(0) << ", "
              << layout.stride(1) << "]\n";
    for(int i = 0; i < 4; ++i)
      for(int j = 0; j < 8; ++j)
        std::cout << "(" << i << "," << j << ") -> "
                  << layout(i,j) << "\n";
    std::cout << "\n";
  }

  //
  // Example B: nested [4,[2,4]]  → dims [4,2,4]
  //
  {
    Shape<2,4> inner_shape{};
    auto inner_layout = make_layout<Stride<4,1>>(inner_shape);
    Shape<4> outer_shape{};
    auto nested = make_nested_layout(outer_shape, inner_layout);

    std::cout << "Example B: shape [4,[2,4]], strides = ["
              << nested.stride(0) << ", "
              << nested.stride(1) << ", "
              << nested.stride(2) << "]\n";
    // strides = [8, 4, 1]
    for(int i = 0; i < 4; ++i)
      for(int a = 0; a < 2; ++a)
        for(int b = 0; b < 4; ++b)
          std::cout << "(" << i << "," << a << "," << b << ") -> "
                    << nested(i,a,b) << "\n";
    std::cout << "\n";
  }

  //
  // Example C: double-nested [[2,2],[2,4]] → dims [2,2,2,4]
  //
  {
    Shape<2,4> inner_shape{};
    auto inner_layout = make_layout<Stride<4,1>>(inner_shape);
    Shape<2,2> outer_shape{};
    auto nested = make_nested_layout(outer_shape, inner_layout);

    std::cout << "Example C: shape [[2,2],[2,4]], strides = ["
              << nested.stride(0) << ", "
              << nested.stride(1) << ", "
              << nested.stride(2) << ", "
              << nested.stride(3) << "]\n";
    // strides = [16, 8, 4, 1]
    for(int i = 0; i < 2; ++i)
      for(int j = 0; j < 2; ++j)
        for(int a = 0; a < 2; ++a)
          for(int b = 0; b < 4; ++b)
            std::cout << "(" << i << "," << j << "," << a << "," << b << ") -> "
                      << nested(i,j,a,b) << "\n";
    std::cout << "\n";
  }

  return 0;
}

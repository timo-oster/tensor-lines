#ifndef CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_HH
#define CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_HH

#include "utils.hh"

#include <Eigen/Core>

#include <utility>
#include <type_traits>

namespace pev
{

template<typename T, typename C,
         int... Degree>
class TensorProductBezierTriangle
{

};

}

#include "TensorProductBezierTriangle1.hh"
#include "TensorProductBezierTriangle3.hh"
#include "TensorProductBezierTriangle1_2.hh"
#include "TensorProductBezierTriangle1_3.hh"

#endif

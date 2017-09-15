#ifndef CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_HH
#define CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_HH

#include <Eigen/Core>

namespace pev
{

template<typename T,
         int... Degree>
class TensorProductBezierTriangle
{

};

}

#include "TensorProductBezierTriangle3.hh"
#include "TensorProductBezierTriangle1_2.hh"
#include "TensorProductBezierTriangle1_3.hh"

#endif

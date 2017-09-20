#ifndef CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_1_HH
#define CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_1_HH

#include "TensorProductBezierTriangle.hh"

namespace pev
{

template <typename T, typename C>
struct TensorProductTraits<TensorProductBezierTriangle<T, C, 1>>
{
    static constexpr std::size_t NCoords = 3;
    static constexpr std::size_t NCoeffs = 3;
};

template <typename T, typename C>
class TensorProductBezierTriangle<T, C, 1>
        : public TensorProductBezierTriangleBase<
                TensorProductBezierTriangle<T, C, 1>, T, C, 1>
{
public:
    using Base = TensorProductBezierTriangleBase<
                    TensorProductBezierTriangle<T, C, 1>, T, C, 1>;
    using Coords = typename Base::Coords;
    using Coeffs = typename Base::Coeffs;

    enum Indices : int
    {
        i100 = 0,
        i010,
        i001
    };

    friend Base;

    using Base::Base;

private:
    using Self = TensorProductBezierTriangle<T, C, 1>;
    using Traits = TensorProductTraits<Self>;
    using Basis = Eigen::Matrix<C, Traits::NCoeffs, 1>;
    using DomainPoints = Eigen::Matrix<C, Traits::NCoeffs, Traits::NCoords>;

    static const DomainPoints& domainPoints();

    static Basis makeBasis(const Coords& pos);

    // Splitting and interpolating operators
    template<int I, int D=0>
    static Coeffs splitCoeffs(const Coeffs& in);

    static Coeffs computeCoeffs(const Coeffs& samples);
};

}

#include "TensorProductBezierTriangle1.tcc"

#endif

#ifndef CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_2_HH
#define CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_2_HH

#include "TensorProductBezierTriangle.hh"

namespace pev
{

template <typename T, typename C>
struct TensorProductTraits<TensorProductBezierTriangle<T, C, 2>>
{
    static constexpr std::size_t NCoords = 3;
    static constexpr std::size_t NCoeffs = 6;
};

template <typename T, typename C>
class TensorProductBezierTriangle<T, C, 2>
        : public TensorProductBezierTriangleBase<
                TensorProductBezierTriangle<T, C, 2>, T, C, 2>
{
public:
    using Base = TensorProductBezierTriangleBase<
                    TensorProductBezierTriangle<T, C, 2>, T, C, 2>;
    using Coords = typename Base::Coords;
    using Coeffs = typename Base::Coeffs;

    enum Indices : std::size_t
    {
        i200 = 0,
        i110,
        i101,
        i020,
        i011,
        i002
    };

    friend Base;

    using Base::Base;

private:
    using Self = TensorProductBezierTriangle<T, C, 2>;
    using Traits = TensorProductTraits<Self>;
    using Basis = Eigen::Matrix<C, Traits::NCoeffs, 1>;
    using DomainPoints = Eigen::Matrix<C, Traits::NCoeffs, Traits::NCoords>;

    static const DomainPoints& domainPoints();

    static Basis makeBasis(const Coords& pos);

    // Splitting and interpolating operators
    template<std::size_t I, std::size_t D=0>
    static Coeffs splitCoeffs(const Coeffs& in);

    static Coeffs computeCoeffs(const Coeffs& samples);
};

} // namespace pev

#include "TensorProductBezierTriangle2.tcc"

#endif

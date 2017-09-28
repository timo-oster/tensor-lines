#ifndef CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_1_1_HH
#define CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_1_1_HH

#include "TensorProductBezierTriangle.hh"

namespace pev
{

template <typename T, typename C>
struct TensorProductTraits<TensorProductBezierTriangle<T, C, 1, 1>>
{
    static constexpr std::size_t NCoords = 6;
    static constexpr std::size_t NCoeffs = 9;
};

template <typename T, typename C>
class TensorProductBezierTriangle<T, C, 1, 1>
        : public TensorProductBezierTriangleBase<
                    TensorProductBezierTriangle<T, C, 1, 1>, T, C, 1, 1>
{
public:
    using Self = TensorProductBezierTriangle<T, C, 1, 1>;
    using Base = TensorProductBezierTriangleBase<
                    TensorProductBezierTriangle<T, C, 1, 1>, T, C, 1, 1>;
    using Traits = TensorProductTraits<Self>;
    using Coords = typename Base::Coords;
    using Coeffs = typename Base::Coeffs;

    enum Indices : std::size_t
    {
        i100100 = 0,
        i100010,
        i100001,
        i010100,
        i010010,
        i010001,
        i001100,
        i001010,
        i001001
    };

    friend Base;

    using Base::Base;

private:
    using Basis = Eigen::Matrix<C, Traits::NCoeffs, 1>;
    using DomainPoints = Eigen::Matrix<C, Traits::NCoeffs, Traits::NCoords>;

    static const DomainPoints& domainPoints();

    static Basis makeBasis(const Coords& pos);

    // Splitting and interpolating operators
    template<std::size_t I, std::size_t D>
    static Coeffs splitCoeffs(const Coeffs& in);

    static Coeffs computeCoeffs(const Coeffs& samples);
};


// template <typename T, typename C>
// struct TensorProductDerivativeType<0, T, C, 1, 1>
// {
//     using type = TensorProductBezierTriangle<T, C, 2>;
// };


// template <typename T, typename C>
// struct TensorProductDerivativeType<1, T, C, 1, 1>
// {
//     using type = TensorProductBezierTriangle<T, C, 1, 1>;
// };

// template <typename T, typename C>
// struct TensorProductDerivative<0, T, C, 1, 1>
// {
//     using Coeffs = typename TensorProductBezierTriangle<T, C, 1, 1>::Coeffs;
//     using DerivCoeffs =
//             typename TensorProductDerivativeType<0, T, C, 1, 1>::type::Coeffs;
//     static DerivCoeffs deriv_op(const Coeffs& in, int dir);
// };

// template <typename T, typename C>
// struct TensorProductDerivative<1, T, C, 1, 1>
// {
//     using Coeffs = typename TensorProductBezierTriangle<T, C, 1, 1>::Coeffs;
//     using DerivCoeffs =
//             typename TensorProductDerivativeType<1, T, C, 1, 1>::type::Coeffs;
//     static DerivCoeffs deriv_op(const Coeffs& in, int dir);
// };

} // namespace pev

#include "TensorProductBezierTriangle1_1.tcc"

#endif

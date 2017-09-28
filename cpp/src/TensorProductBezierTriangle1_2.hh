#ifndef CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_1_2_HH
#define CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_1_2_HH

#include "TensorProductBezierTriangle.hh"

namespace pev
{

template <typename T, typename C>
struct TensorProductTraits<TensorProductBezierTriangle<T, C, 1, 2>>
{
    static constexpr std::size_t NCoords = 6;
    static constexpr std::size_t NCoeffs = 18;
};

template <typename T, typename C>
class TensorProductBezierTriangle<T, C, 1, 2>
        : public TensorProductBezierTriangleBase<
                    TensorProductBezierTriangle<T, C, 1, 2>, T, C, 1, 2>
{
public:
    using Self = TensorProductBezierTriangle<T, C, 1, 2>;
    using Base = TensorProductBezierTriangleBase<
                    TensorProductBezierTriangle<T, C, 1, 2>, T, C, 1, 2>;
    using Traits = TensorProductTraits<Self>;
    using Coords = typename Base::Coords;
    using Coeffs = typename Base::Coeffs;

    enum Indices : std::size_t
    {
        i100200 = 0,
        i100110,
        i100101,
        i100020,
        i100011,
        i100002,
        i010200,
        i010110,
        i010101,
        i010020,
        i010011,
        i010002,
        i001200,
        i001110,
        i001101,
        i001020,
        i001011,
        i001002
    };

    friend Base;

    using Base::Base;

private:
    using Basis = Eigen::Matrix<C, Traits::NCoeffs, 1>;
    using DomainPoints = Eigen::Matrix<C, Traits::NCoeffs, Traits::NCoords>;
    template <std::size_t D>
    using DerivCoeffs =
            typename TensorProductDerivativeType<D, T, C, 1, 2>::Coeffs;

    static const DomainPoints& domainPoints();

    static Basis makeBasis(const Coords& pos);

    // Splitting and interpolating operators
    template<std::size_t I, std::size_t D>
    static Coeffs splitCoeffs(const Coeffs& in);

    static Coeffs computeCoeffs(const Coeffs& samples);
};


template <typename T, typename C>
struct TensorProductDerivativeType<0, T, C, 1, 2>
{
    using type = TensorProductBezierTriangle<T, C, 2>;
};


template <typename T, typename C>
struct TensorProductDerivativeType<1, T, C, 1, 2>
{
    using type = TensorProductBezierTriangle<T, C, 1, 1>;
};

template <typename T, typename C>
struct TensorProductDerivative<0, T, C, 1, 2>
{
    using Coeffs = typename TensorProductBezierTriangle<T, C, 1, 2>::Coeffs;
    using DerivCoeffs =
            typename TensorProductDerivativeType<0, T, C, 1, 2>::type::Coeffs;
    static DerivCoeffs deriv_op(const Coeffs& in, std::size_t dir);
};

template <typename T, typename C>
struct TensorProductDerivative<1, T, C, 1, 2>
{
    using Coeffs = typename TensorProductBezierTriangle<T, C, 1, 2>::Coeffs;
    using DerivCoeffs =
            typename TensorProductDerivativeType<1, T, C, 1, 2>::type::Coeffs;
    static DerivCoeffs deriv_op(const Coeffs& in, std::size_t dir);
};

} // namespace pev

#include "TensorProductBezierTriangle1_2.tcc"

#endif

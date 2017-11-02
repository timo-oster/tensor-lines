#ifndef CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_2_2_HH
#define CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_2_2_HH

#include "TensorProductBezierTriangle.hh"

namespace pev
{

template <typename T, typename C>
struct TensorProductTraits<TensorProductBezierTriangle<T, C, 2, 2>>
{
    static constexpr std::size_t NCoords = 6;
    static constexpr std::size_t NCoeffs = 36;
};

template <typename T, typename C>
class TensorProductBezierTriangle<T, C, 2, 2>
        : public TensorProductBezierTriangleBase<
                    TensorProductBezierTriangle<T, C, 2, 2>, T, C, 2, 2>
{
public:
    using Self = TensorProductBezierTriangle<T, C, 2, 2>;
    using Base = TensorProductBezierTriangleBase<
                    TensorProductBezierTriangle<T, C, 2, 2>, T, C, 2, 2>;
    using Traits = TensorProductTraits<Self>;
    using Coords = typename Base::Coords;
    using Coeffs = typename Base::Coeffs;

    enum Indices : std::size_t
    {
        i200200 = 0,
        i200110,
        i200101,
        i200020,
        i200011,
        i200002,
        i110200,
        i110110,
        i110101,
        i110020,
        i110011,
        i110002,
        i101200,
        i101110,
        i101101,
        i101020,
        i101011,
        i101002,
        i020200,
        i020110,
        i020101,
        i020020,
        i020011,
        i020002,
        i011200,
        i011110,
        i011101,
        i011020,
        i011011,
        i011002,
        i002200,
        i002110,
        i002101,
        i002020,
        i002011,
        i002002
    };

    friend Base;

    using Base::Base;

private:
    using Basis = Eigen::Matrix<C, Traits::NCoeffs, 1>;
    using DomainPoints = Eigen::Matrix<C, Traits::NCoeffs, Traits::NCoords>;
    template <std::size_t D>
    using DerivCoeffs =
            typename TensorProductDerivativeType<D, T, C, 2, 2>::Coeffs;

    static const DomainPoints& domainPoints();

    static Basis makeBasis(const Coords& pos);

    // Splitting and interpolating operators
    template<std::size_t I, std::size_t D>
    static Coeffs splitCoeffs(const Coeffs& in);

    static Coeffs computeCoeffs(const Coeffs& samples);
};


template <typename T, typename C>
struct TensorProductDerivativeType<0, T, C, 2, 2>
{
    using type = TensorProductBezierTriangle<T, C, 1, 2>;
};


template <typename T, typename C>
struct TensorProductDerivativeType<1, T, C, 2, 2>
{
    using type = TensorProductBezierTriangle<T, C, 2, 1>;
};

template <typename T, typename C>
struct TensorProductDerivative<0, T, C, 2, 2>
{
    using Coeffs = typename TensorProductBezierTriangle<T, C, 2, 2>::Coeffs;
    using DerivCoeffs =
            typename TensorProductDerivativeType_t<0, T, C, 2, 2>::Coeffs;
    static DerivCoeffs deriv_op(const Coeffs& in, std::size_t dir);
};

template <typename T, typename C>
struct TensorProductDerivative<1, T, C, 2, 2>
{
    using Coeffs = typename TensorProductBezierTriangle<T, C, 2, 2>::Coeffs;
    using DerivCoeffs =
            typename TensorProductDerivativeType_t<1, T, C, 2, 2>::Coeffs;
    static DerivCoeffs deriv_op(const Coeffs& in, std::size_t dir);
};

} // namespace pev

#include "TensorProductBezierTriangle2_2.tcc"

#endif

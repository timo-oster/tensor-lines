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

    enum Indices : int
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

    static const DomainPoints& domainPoints();

    static Basis makeBasis(const Coords& pos);

    // Splitting and interpolating operators
    template<int I, int D>
    static Coeffs splitCoeffs(const Coeffs& in);

    static Coeffs computeCoeffs(const Coeffs& samples);
};

}

#include "TensorProductBezierTriangle1_2.tcc"

#endif

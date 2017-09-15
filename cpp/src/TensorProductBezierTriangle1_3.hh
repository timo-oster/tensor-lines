#ifndef CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_1_3_HH
#define CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_1_3_HH

#include "TensorProductBezierTriangle.hh"

#include <utility>
#include <type_traits>

namespace pev
{

template <typename T, typename C>
struct TensorProductTraits<TensorProductBezierTriangle<T, C, 1, 3>>
{
    static constexpr std::size_t NCoords = 6;
    static constexpr std::size_t NCoeffs = 30;
    using Coords = Eigen::Matrix<C, NCoords, 1>;
    using Coeffs = std::array<T, NCoeffs>;
};

template <typename T, typename C>
class TensorProductBezierTriangle<T, C, 1, 3>
        : public TensorProductBezierTriangleBase<
                    TensorProductBezierTriangle<T, C, 1, 3>, T, C, 1, 3>
{
public:
    using Self = TensorProductBezierTriangle<T, C, 1, 3>;
    using Base = TensorProductBezierTriangleBase<
                    TensorProductBezierTriangle<T, C, 1, 3>,
                    T, C, 1, 3>;
    static constexpr std::size_t NCoords = TensorProductTraits<Self>::NCoords;
    static constexpr std::size_t NCoeffs = TensorProductTraits<Self>::NCoeffs;
    using Coords = typename TensorProductTraits<Self>::Coords;
    using Coeffs = typename TensorProductTraits<Self>::Coeffs;

    enum Indices : int
    {
        i100300 = 0,
        i100210,
        i100201,
        i100120,
        i100111,
        i100102,
        i100030,
        i100021,
        i100012,
        i100003,
        i010300,
        i010210,
        i010201,
        i010120,
        i010111,
        i010102,
        i010030,
        i010021,
        i010012,
        i010003,
        i001300,
        i001210,
        i001201,
        i001120,
        i001111,
        i001102,
        i001030,
        i001021,
        i001012,
        i001003
    };

    friend Base;

    using Base::Base;

private:
    using Basis = Eigen::Matrix<C, NCoeffs, 1>;
    using DomainPoints = Eigen::Matrix<T, NCoeffs, NCoords>;

    static const DomainPoints& domainPoints();

    static Basis makeBasis(const Coords& pos);

    // Splitting and interpolating operators
    template<int I, int D>
    static Coeffs splitCoeffs(const Coeffs& in);

    static Coeffs computeCoeffs(const Coeffs& samples);
};

}

#include "TensorProductBezierTriangle1_3.tcc"

#endif

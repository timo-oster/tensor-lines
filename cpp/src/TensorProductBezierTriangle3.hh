#ifndef CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_3_HH
#define CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_3_HH

#include "TensorProductBezierTriangle.hh"

namespace pev
{
template <typename T, typename C>
struct TensorProductTraits<TensorProductBezierTriangle<T, C, 3>>
{
    static constexpr std::size_t NCoords = 3;
    static constexpr std::size_t NCoeffs = 3;
    using Coords = Eigen::Matrix<C, NCoords, 1>;
    using Coeffs = std::array<T, NCoeffs>;
};

template <typename T, typename C>
class TensorProductBezierTriangle<T, C, 3>
        : public TensorProductBezierTriangleBase<TensorProductBezierTriangle<T,
                                                                             C,
                                                                             3>,
                                                 T,
                                                 C,
                                                 3>
{
public:
    using Self = TensorProductBezierTriangle<T, C, 3>;
    using Base = TensorProductBezierTriangleBase<
                    TensorProductBezierTriangle<T, C, 3>,
                    T, C, 3>;
    static constexpr std::size_t NCoords = TensorProductTraits<Self>::NCoords;
    static constexpr std::size_t NCoeffs = TensorProductTraits<Self>::NCoeffs;
    using Coords = typename TensorProductTraits<Self>::Coords;
    using Coeffs = typename TensorProductTraits<Self>::Coeffs;

    enum Indices : int
    {
        i300 = 0,
        i210,
        i201,
        i120,
        i111,
        i102,
        i030,
        i021,
        i012,
        i003
    };

    friend Base;

    using Base::Base;

private:
    using Basis = Eigen::Matrix<T, NCoeffs, 1>;
    using DomainPoints = Eigen::Matrix<T, NCoeffs, NCoords>;

    static const DomainPoints& domainPoints();

    static Basis makeBasis(const Coords& pos);

    // Splitting and interpolating operators
    template<int I, int D=0>
    static Coeffs splitCoeffs(const Coeffs& in);

    static Coeffs computeCoeffs(const Coeffs& samples);
};

}

#include "TensorProductBezierTriangle3.tcc"

#endif

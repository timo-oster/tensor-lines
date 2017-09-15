#ifndef CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_1_3_HH
#define CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_1_3_HH

#include "TensorProductBezierTriangle.hh"

#include <utility>
#include <type_traits>

namespace pev
{

template<typename T>
class TensorProductBezierTriangle<T, 1, 3>
{
public:
    static constexpr std::size_t NCoords = 6;
    static constexpr std::size_t NCoeffs = 30;
    using Self = TensorProductBezierTriangle<T, 1, 3>;
    using Coords = Eigen::Matrix<T, NCoords, 1>;
    using Coeffs = Eigen::Matrix<T, NCoeffs, 1>;

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

    TensorProductBezierTriangle()
            : _coeffs(Coeffs::Zero())
    {
    }

    explicit TensorProductBezierTriangle(const Coeffs& coefficients)
            : _coeffs(coefficients)
    {
    }

    // Check for type that is callable with Coords and returns something convertible to T
    template <typename Function>
    using is_legal_init_function =
            typename std::is_convertible<
                typename std::result_of<Function(Coords)>::type,
                T>;

    template<
        typename Function,
        typename std::enable_if<
            is_legal_init_function<Function>::value,
            int
        >::type = 0>
    explicit TensorProductBezierTriangle(Function func)
    {
        auto rhs = Coeffs{};
        const auto& dp = domainPoints();
        for(auto i: range(NCoeffs))
        {
            rhs[i] = func(dp.row(i));
        }

        _coeffs = computeCoeffs(rhs);
    }

    T operator()(const Coords& pos) const
    {
        return makeBasis(pos).dot(_coeffs);
    }

    T& operator[](int i)
    {
        return _coeffs[i];
    }

    const T& operator[](int i) const
    {
        return _coeffs[i];
    }

    Coeffs& coefficients()
    {
        return _coeffs;
    }

    const Coeffs& coefficients() const
    {
        return _coeffs;
    }

    template<int D>
    std::array<Self, 4> split() const;


private:
    using SysMatrix = Eigen::Matrix<T, NCoeffs, NCoeffs>;
    using SplitMatrix = Eigen::Matrix<T, NCoeffs*4, NCoeffs>;
    using Basis = Eigen::Matrix<T, NCoeffs, 1>;
    using DomainPoints = Eigen::Matrix<T, NCoeffs, NCoords>;

    // Coefficients of the Bezier Triangle
    Coeffs _coeffs;

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

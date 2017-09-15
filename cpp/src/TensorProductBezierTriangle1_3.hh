#ifndef CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_1_3_HH
#define CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_1_3_HH

#include "TensorProductBezierTriangle.hh"

#include <utility>
#include <type_traits>

namespace pev
{

template<typename T, typename C>
class TensorProductBezierTriangle<T, C, 1, 3>
{
public:
    static constexpr std::size_t NCoords = 6;
    static constexpr std::size_t NCoeffs = 30;
    using Self = TensorProductBezierTriangle<T, C, 1, 3>;
    using Coords = Eigen::Matrix<C, NCoords, 1>;
    using Coeffs = std::array<T, NCoeffs>;

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

    TensorProductBezierTriangle() = default;

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
        auto base = makeBasis(pos);
        auto result = T{base[0] * _coeffs[0]};
        for(auto i: range(1, NCoeffs))
        {
            result += base[i] * _coeffs[i];
        }
        return result;
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

    friend bool operator==(const Self& o1, const Self& o2)
    {
        return o1.coefficients() == o2.coefficients();
    }
    friend bool operator!=(const Self& o1, const Self& o2)
    {
        return !(o1 == o2);
    }


private:
    using Basis = Eigen::Matrix<C, NCoeffs, 1>;
    using DomainPoints = Eigen::Matrix<T, NCoeffs, NCoords>;

    // Coefficients of the Bezier Triangle
    Coeffs _coeffs = {};

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

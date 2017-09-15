#ifndef CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_1_2_HH
#define CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_1_2_HH

#include "TensorProductBezierTriangle.hh"

#include <utility>
#include <type_traits>

namespace pev
{

template<typename T>
class TensorProductBezierTriangle<T, 1, 2>
{
public:
    static constexpr std::size_t NCoords = 6;
    static constexpr std::size_t NCoeffs = 18;
    using Self = TensorProductBezierTriangle<T, 1, 2>;
    using Coords = Eigen::Matrix<T, NCoords, 1>;
    using Coeffs = Eigen::Matrix<T, NCoeffs, 1>;

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

#include "TensorProductBezierTriangle1_2.tcc"

#endif

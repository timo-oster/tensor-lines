#ifndef CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_3_HH
#define CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_3_HH

#include "TensorProductBezierTriangle.hh"

namespace pev
{

template<typename T>
class TensorProductBezierTriangle<T, 3>
{
public:
    static constexpr std::size_t NCoords = 3;
    static constexpr std::size_t NCoeffs = 10;
    using Self = TensorProductBezierTriangle<T, 3>;
    using Coords = Eigen::Matrix<T, NCoords, 1>;
    using Coeffs = Eigen::Matrix<T, NCoeffs, 1>;

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

    template<int D=0>
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
    template<int I, int D=0>
    static Coeffs splitCoeffs(const Coeffs& in);

    static Coeffs computeCoeffs(const Coeffs& samples);
};

}

#include "TensorProductBezierTriangle3.tcc"

#endif

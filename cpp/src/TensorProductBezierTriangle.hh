#ifndef CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_HH
#define CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_HH

#include "utils.hh"

#include <Eigen/Core>

#include <utility>
#include <type_traits>

namespace pev
{
template <typename Derived>
struct TensorProductTraits;

template <typename Derived, typename T, typename C, int... Degrees>
class TensorProductBezierTriangleBase
{
public:
    using Coords = typename TensorProductTraits<Derived>::Coords;
    using Coeffs = typename TensorProductTraits<Derived>::Coeffs;

    TensorProductBezierTriangleBase() = default;

    explicit TensorProductBezierTriangleBase(const Coeffs& coefficients)
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
    explicit TensorProductBezierTriangleBase(Function func)
    {
        auto rhs = Coeffs{};
        const auto& dp = Derived::domainPoints();
        for(auto i: range(Derived::NCoeffs))
        {
            rhs[i] = func(dp.row(i));
        }
        _coeffs = Derived::computeCoeffs(rhs);
    }

    T operator()(const Coords& pos) const
    {
        auto base = Derived::makeBasis(pos);
        auto result = T{base[0] * _coeffs[0]};
        for(auto i: range(1, Derived::NCoeffs))
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

    template<int D=0>
    std::array<Derived, 4> split() const
    {
        static_assert(D >=0 && D < sizeof...(Degrees),
                      "D must be smaller than the number of Degrees");
        return {Derived{Derived::template splitCoeffs<0, D>(_coeffs)},
                Derived{Derived::template splitCoeffs<1, D>(_coeffs)},
                Derived{Derived::template splitCoeffs<2, D>(_coeffs)},
                Derived{Derived::template splitCoeffs<3, D>(_coeffs)}};
    }

    friend bool operator==(const Derived& o1, const Derived& o2)
    {
        return o1.coefficients() == o2.coefficients();
    }
    friend bool operator!=(const Derived& o1, const Derived& o2)
    {
        return !(o1 == o2);
    }
private:

    // Coefficients of the Bezier Triangle
    Coeffs _coeffs = {};
};

template<typename T,
         typename C,
         int... Degree>
class TensorProductBezierTriangle
{

};

}

#include "TensorProductBezierTriangle1.hh"
#include "TensorProductBezierTriangle3.hh"
#include "TensorProductBezierTriangle1_2.hh"
#include "TensorProductBezierTriangle1_3.hh"

#endif

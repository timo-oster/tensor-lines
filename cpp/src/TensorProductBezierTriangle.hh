#ifndef CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_HH
#define CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_HH

#include "utils.hh"

#include <Eigen/Core>

#include <type_traits>
#include <utility>
#include <stdexcept>

namespace pev
{
template <std::size_t D, typename T, typename C, std::size_t... Degrees>
struct TensorProductDerivativeType;

template <std::size_t D, typename T, typename C, std::size_t... Degrees>
using TensorProductDerivativeType_t =
        typename TensorProductDerivativeType<D, T, C, Degrees...>::type;

template <std::size_t D, typename T, typename C, std::size_t... Degrees>
struct TensorProductDerivative;


/**
 * Traits class for accessing typedefs from specializations of
 * TensorProductBezierTriangle.
 *
 * Must declare two `static constexpr std::size_t` members `NCoords` and
 * `NCoeffs` that specify the number of coordinates and coefficients of the
 * polynomial. E.g.:
 *
 * ```cpp
 * template <typename T, typename C>
 * struct TensorProductTraits<TensorProductBezierTriangle<T, C, 3>>
 * {
 *     static constexpr std::size_t NCoords = 3;
 *     static constexpr std::size_t NCoeffs = 3;
 * };
 * ```
 *
 * @tparam Derived Specialization of TensorProductBezierTriangle
 */
template <typename Derived>
struct TensorProductTraits;


/**
 * @brief Base class for specializations of TensorProductBezierTriangle.
 * @details Base class holding the public interface and common functionality for
 *      specializations of TensorProductBezierTriangle.
 *
 *      Specializations of TensorProductBezierTriangle that derive from this
 *      class need to define a Traits class (see TensorProductTraits) as well a
 *      number of static member functions. It also needs to declare the base
 *      class as a friend and inherit its constructors. A minimal example for a
 *      specialization looks as follows:
 *
 *      ```cpp
 *      template <typename T, typename C>
 *      struct TensorProductTraits<TensorProductBezierTriangle<T, C, 1>>
 *      {
 *          static constexpr std::size_t NCoords = 3;
 *          static constexpr std::size_t NCoeffs = 3;
 *      };
 *
 *      template <typename T, typename C>
 *      class TensorProductBezierTriangle<T, C, 1>
 *              : public TensorProductBezierTriangleBase<
 *                      TensorProductBezierTriangle<T, C, 1>, T, C, 1>
 *      {
 *      public:
 *          using Base = TensorProductBezierTriangleBase<
 *                          TensorProductBezierTriangle<T, C, 1>, T, C, 1>;
 *          using Coords = typename Base::Coords;
 *          using Coeffs = typename Base::Coeffs;
 *
 *          friend Base;
 *
 *          using Base::Base;
 *
 *      private:
 *          using Self = TensorProductBezierTriangle<T, C, 1>;
 *          using Traits = TensorProductTraits<Self>;
 *          using Basis = Eigen::Matrix<C, Traits::NCoeffs, 1>;
 *          using DomainPoints = Eigen::Matrix<C, Traits::NCoeffs, Traits::NCoords>;
 *
 *          // Coordinates of the control points
 *          static const DomainPoints& domainPoints();
 *
 *          // Convert coordinates into polynomial basis
 *          static Basis makeBasis(const Coords& pos);
 *
 *          // Compute coefficients for one of four subdivision triangles when
 *          // splitting in the given dimension D
 *          template<int I, int D=0>
 *          static Coeffs splitCoeffs(const Coeffs& in);
 *
 *          // Compute coefficients of an interpolating polynomial given sampled
 *          // values at the control points.
 *          static Coeffs computeCoeffs(const Coeffs& samples);
 *      };
 *
 *      ```
 *
 * @tparam Derived The specialization of TensorProductBezierTriangle that
 *     inherits from this class.
 * @tparam T Type of the coefficients
 * @tparam C Type of the coordinates (must be scalar)
 * @tparam Degrees Degrees of the polynomials participating in the tensor
 *     product
 */
template <typename Derived, typename T, typename C, std::size_t... Degrees>
class TensorProductBezierTriangleBase
{
public:
    using Traits = TensorProductTraits<Derived>;
    using Coords = Eigen::Matrix<C, Traits::NCoords, 1>;
    using Coeffs = std::array<T, Traits::NCoeffs>;
    // constexpr static std::size_t NCoords = sizeof...(Degrees) * 3;
    // constexpr static std::size_t NCoeffs = (((Degrees+1)*(Degrees+2))/2) * ...;

    TensorProductBezierTriangleBase() = default;

    /**
     * Create polynomial with the given coefficients.
     */
    explicit TensorProductBezierTriangleBase(const Coeffs& coefficients)
            : _coeffs(coefficients)
    {
    }

    // Check for type that is callable with Coords and returns something convertible to T
    template <typename Function>
    using is_legal_init_function =
            std::is_convertible<std::result_of_t<Function(Coords)>, T>;

    /**
     * Create polynomial interpolating the given function at the control points.
     *
     * @tparam Function Callable that takes Coords and returns a value
     *      convertible to T.
     */
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
        for(auto i: cpp_utils::range(Traits::NCoeffs))
        {
            rhs[i] = func(dp.row(i));
        }
        _coeffs = Derived::computeCoeffs(rhs);
    }

    /**
     * Evaluate the polynomial at the given coordinates.
     */
    T operator()(const Coords& pos) const
    {
        auto base = Derived::makeBasis(pos);
        auto result = T{base[0] * _coeffs[0]};
        for(auto i: cpp_utils::range(1, Traits::NCoeffs))
        {
            result += base[i] * _coeffs[i];
        }
        return result;
    }

    /**
     * Access the i-th coefficient.
     */
    T& operator[](std::size_t i)
    {
        return _coeffs[i];
    }

    /**
     * Access the i-th coefficient.
     */
    const T& operator[](std::size_t i) const
    {
        return _coeffs[i];
    }

    /**
     * Access the coefficients directly
     */
    Coeffs& coefficients()
    {
        return _coeffs;
    }

    /**
     * Access the coefficients directly
     */
    const Coeffs& coefficients() const
    {
        return _coeffs;
    }

    /**
     * @brief Split the triangle into four new ones in the given space.
     * @details The triangle A-B-C is split into four new ones numbered
     *      according to the following diagram:
     * ```
     *          C
     *         / \
     *        / 2 \
     *       o-----o
     *      / \ 3 / \
     *     / 0 \ / 1 \
     *    A-----o-----B
     * ```
     * The corners of the new ones are ordered starting with the bottom left
     * (bottom for the middle one) and counting in counter-clockwise direction.
     *
     * @tparam D Space in which to split
     * @return Array of four new polynomials representing the split parts.
     */
    template <std::size_t D=0>
    std::array<Derived, 4> split() const
    {
        return {split<D>(0), split<D>(1), split<D>(2), split<D>(3)};
    }

    /**
     * Equivalent to `split<D>()[part]`
     *
     * @param part index of the sub-triangle of the split (0-3)
     * @tparam std::D=0 Space in which to split
     * @return One new polynomial representing one of the four sub-triangles
     */
    template <std::size_t D=0>
    Derived split(std::size_t part) const
    {
        if(D >= sizeof...(Degrees))
            throw(std::invalid_argument(
                    "D must be smaller than the number of Degrees"));
        switch(part)
        {
            case 0: return Derived{Derived::template splitCoeffs<0, D>(_coeffs)};
            case 1: return Derived{Derived::template splitCoeffs<1, D>(_coeffs)};
            case 2: return Derived{Derived::template splitCoeffs<2, D>(_coeffs)};
            case 3: return Derived{Derived::template splitCoeffs<3, D>(_coeffs)};
            default: throw std::invalid_argument("Part must be smaller than 4!");
        }
    }

    template <std::size_t D>
    TensorProductDerivativeType_t<D, T, C, Degrees...>
    derivative(std::size_t i) const
    {
        using Derivative =
                TensorProductDerivativeType_t<D, T, C, Degrees...>;
        using OpType = TensorProductDerivative<D, T, C, Degrees...>;

        assert(i < 3);

        return Derivative{OpType::deriv_op(_coeffs, i)};
    }

    void operator+=(const Derived& other)
    {
        for(auto i: cpp_utils::range(_coeffs.size()))
        {
            _coeffs[i] += other[i];
        }
    }

    void operator-=(const Derived& other)
    {
        for(auto i: cpp_utils::range(_coeffs.size()))
        {
            _coeffs[i] -= other[i];
        }
    }

    template <typename Scalar>
    void operator*=(const Scalar& scalar)
    {
        for(auto& c: _coeffs)
        {
            c *= scalar;
        }
    }

    template <typename Scalar>
    void operator/=(const Scalar& scalar)
    {
        for(auto& c: _coeffs)
        {
            c /= scalar;
        }
    }

    Derived operator-() const
    {
        auto r_coeffs = Coeffs{};
        for(auto i: cpp_utils::range(r_coeffs.size()))
        {
            r_coeffs[i] = -this->coefficients()[i];
        }
        return Derived{r_coeffs};
    }

    Derived operator+(const Derived& other) const
    {
        auto result = Derived{*this};
        result += other;
        return result;
    }

    Derived operator-(const Derived& other) const
    {
        return *this + (-other);
    }

    template <typename Scalar>
    Derived operator*(const Scalar& scalar) const
    {
        auto result = Derived{*this};
        result *= scalar;
        return result;
    }

    template <typename Scalar>
    Derived operator/(const Scalar& scalar) const
    {
        auto result = Derived{*this};
        result /= scalar;
        return result;
    }

    template <typename Scalar>
    friend Derived operator*(const Scalar& scalar, const Derived& poly)
    {
        return poly * scalar;
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


/**
 * Unspecialized class template for a tensor product of Bernstein-Bézier
 * triangles.
 *
 * A tensor product of Bernstein-Bézier triangles has the form
 *
 * \f{align*}{
 *     f(\alpha_1, \beta_1, \gamma_1, \dots, \alpha_n, \beta_n, \gamma_n) &=
 *     \sum_{i_1+j_1+k_1=d_1} \frac{d_1!}{i_1!j_1!k_1!} \; b_{i_1j_1k_1} \;
 *          \alpha_1^{i_1} \beta_1^{j_1} \gamma_1^{k_1} \cdots
 *     \sum_{i_n+j_n+k_n=d_n} \frac{d_n!}{i_n!j_n!k_n!} \; b_{i_nj_nk_n} \;
 *          \alpha_n^{i_n} \beta_n^{j_n} \gamma_n^{k_n}\\
 *     &= \sum_{i_1+j_1+k_1=d_1} \cdots \sum_{i_n+j_n+k_n=d_n}
 *          \frac{d_1!}{i_1!j_1!k_1!} \cdots \frac{d_n!}{i_n!j_n!k_n!} \;
 *          b_{i_1j_1k_1\dots\i_nj_nk_n} \;
 *          \alpha_1^{i_1} \beta_1^{j_1} \gamma_1^{k_1} \cdots
 *          \alpha_n^{i_n} \beta_n^{j_n} \gamma_n^{k_n}\\
 *     &\text{with}\; \alpha_i + \beta_i + \gamma_i = 1
 * \f}
 *
 * Where \f$\alpha_i, \beta_i, \gamma_i\f$ are the barycentric coordinates of
 * the i-th Bernstein-Bézier triangle, \f$d_i\f$ is the degree of the i-th
 * Bernstein-Bézier triangle, and \f$b_{\dots}\f$ are the coefficients.
 *
 * @tparam T Type of the coefficients
 * @tparam C Type of the coordinates (must be scalar)
 * @tparam Degrees Degrees of the polynomials
 */
template <typename T, typename C, std::size_t... Degrees>
class TensorProductBezierTriangle
{
};

} // namespace pev

#endif

#ifndef CPP_EVALUATOR_UTILS_HH
#define CPP_EVALUATOR_UTILS_HH

#include "Evaluator.hh"

#include <array>
#include <iterator>

#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/max_element.hpp>
#include <boost/range/algorithm/transform.hpp>
#include <boost/range/numeric.hpp>

namespace tl
{

/**
 * Linear tensor field expressed in barycentric coordinates
 */
using TensorInterp = TensorProductBezierTriangle<Mat3d, double, 1>;

/**
 * Compute the derivatives of a TensorProductBezierTriangle along two
 * orthogonal directions on the triangle in the space indicated by @a D.
 *
 * @param poly The polynomial.
 * @tparam D The space (0-sizeof...(Degrees)) in which to perform the derivative
 */
template <std::size_t D,
          typename TPBT,
          typename T,
          typename C,
          std::size_t... Degrees>
auto derivatives(
        const TensorProductBezierTriangleBase<TPBT, T, C, Degrees...>& poly)
        -> std::array<
                typename TensorProductDerivativeType<D, T, C, Degrees...>::type,
                2>
{
    static_assert(D < sizeof...(Degrees), "Invalid index");
    auto d0 = poly.template derivative<D>(0);
    auto d1 = poly.template derivative<D>(1);
    auto d2 = poly.template derivative<D>(2);

    auto da = (d1 - d0) / std::sqrt(2);

    auto db = (2 * d2 - d0 - d1) / std::sqrt(6);

    return {da, db};
}


/**
 * @brief Compute an upper bound for the magnitude of the function value.
 * @details Finds the coefficient with the maximum absolute value.
 *
 * @param poly The polynomial
 * @return An upper bound for std::abs(poly(x)) on the triangles
 */
template <typename TPBT,
          typename T,
          typename C,
          std::size_t... Degrees>
double abs_upper_bound(
        const TensorProductBezierTriangleBase<TPBT, T, C, Degrees...>& poly)
{
    return boost::accumulate(poly.coefficients(), 0., MaxAbs{});
}


/**
 * @brief Compute an upper bound for the magnitude of a number of polynomials.
 * @details Finds the coefficient with the maximum absolute value of all
 *     polynomials.
 *
 * @param polys A sequence of polynomials
 * @return An upper bound for std::abs(poly(x)) over all polynomials in the
 *     sequence
 */
template <typename TPBTSeq>
double abs_max_upper_bound(const TPBTSeq& polys)
{
    using namespace boost;
    using namespace boost::adaptors;
    return accumulate(polys | transformed([](const auto& f) {
                          return abs_upper_bound(f);
                      }),
                      0.,
                      Max{});
}


/**
 * @brief Compute an upper bound for the gradient magnitude of the polynomial in
 *      the space indicated by @a D.
 *
 * @param poly The polynomial
 * @tparam D The space (0-sizeof...(Degrees)) in which to perform the derivative
 * @return an estimate for the upper bound of the gradient magnitude of poly(x)
 *      on the triangles
 */
template <std::size_t D,
          typename TPBT,
          typename T,
          typename C,
          std::size_t... Degrees>
double derivatives_upper_bound(
        const TensorProductBezierTriangleBase<TPBT, T, C, Degrees...>& poly)
{
    static_assert(D < sizeof...(Degrees), "Invalid index");
    auto upper_bound = typename TPBT::Coeffs{};
    // Estimate upper bound of gradient magnitude by L1 norm of control points
    // todo: Can we do better? (Represent gradient magnitude as polynomial)
    auto abssum = [](double v1, double v2) {
        return std::abs(v1) + std::abs(v2);
    };
    auto derivs = derivatives<D>(poly);
    boost::transform(derivs[0].coefficients(),
                     derivs[1].coefficients(),
                     std::begin(upper_bound),
                     abssum);
    return *boost::max_element(upper_bound);
}

/**
 * @brief Compute an upper bound for the gradient magnitude of a sequence of
 *      polynomials in the space indicated by @a D.
 *
 * @param polys A sequence (range) of polynomials
 * @tparam D The space in which to perform the derivatives
 * @return Upper bound for the gradient magnitude on the triangles
 */
template <std::size_t D, typename TPBTSeq>
double derivatives_max_upper_bound(const TPBTSeq& polys)
{
    using namespace boost;
    using namespace boost::adaptors;
    return accumulate(polys | transformed([](const auto& f) {
                          return derivatives_upper_bound<D>(f);
                      }),
                      0.,
                      Max{});
}


/**
 * @brief Compute an upper bound for the norm of a vector of values from
 *     multiple polynomials.
 * @details Finds an upper bound for sqrt(poly_1(x)^2 + poly_2(x)^2 + ...)
 *      where poly_i are the elements of the sequence @a polys.
 *
 * @param polys A sequence (range) of polynomials
 * @return Upper bound for the norm of the vector od polynomials on the
 *     triangles
 */
template <typename TPBTSeq>
double upper_bound_norm(const TPBTSeq& polys)
{
    using namespace boost;
    using namespace boost::adaptors;
    return std::sqrt(accumulate(polys | transformed([](const auto& f) {
                                    return std::pow(
                                            *max_element(f.coefficients()), 2);
                                }),
                                0.));
}

} // namespace tl

#endif

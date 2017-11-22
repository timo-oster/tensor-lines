#ifndef CPP_EVALUATOR_UTILS_HH
#define CPP_EVALUATOR_UTILS_HH

#include "Evaluator.hh"

#include <array>
#include <iterator>

#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/max_element.hpp>
#include <boost/range/algorithm/transform.hpp>
#include <boost/range/numeric.hpp>

namespace pev
{

/**
 * Linear tensor field expressed in barycentric coordinates
 */
using TensorInterp = TensorProductBezierTriangle<Mat3d, double, 1>;

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
    auto d0 = poly.template derivative<D>(0);
    auto d1 = poly.template derivative<D>(1);
    auto d2 = poly.template derivative<D>(2);

    auto da = (d1 - d0) / std::sqrt(2);

    auto db = (2 * d2 - d0 - d1) / std::sqrt(6);

    return {da, db};
}


template <typename TPBT,
          typename T,
          typename C,
          std::size_t... Degrees>
double abs_upper_bound(
        const TensorProductBezierTriangleBase<TPBT, T, C, Degrees...>& poly)
{
    return boost::accumulate(poly.coefficients(), 0., MaxAbs{});
}


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


template <std::size_t D,
          typename TPBT,
          typename T,
          typename C,
          std::size_t... Degrees>
double derivatives_upper_bound(
        const TensorProductBezierTriangleBase<TPBT, T, C, Degrees...>& poly)
{
    auto upper_bound = typename TPBT::Coeffs{};
    // estimate upper bound of gradient magnitude by L1 norm of control points
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

} // namespace pev

#endif

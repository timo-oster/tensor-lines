#include "TensorTopologyEvaluator.hh"

#include <Eigen/Geometry>

#include <boost/algorithm/cxx11/any_of.hpp>
#include <boost/range/algorithm/max_element.hpp>
#include <boost/range/algorithm/min_element.hpp>

#include <tuple>
#include <utility>
#include <vector>

using namespace cpp_utils;

namespace pev
{
template <typename T, std::size_t... Degrees>
using TPBT = TensorProductBezierTriangle<T, double, Degrees...>;

std::array<TPBT<double, 3, 0>, 7> tensorTopologyCoeffs(const TensorInterp& t)
{
    using Coords = TPBT<double, 3, 0>::Coords;
    // Constraint functions according to Zheng et al. 2004

    auto fx = [&](const Coords& coords) -> double {
        auto tv = t(coords.head<3>());
        return tv(0, 0) * ((tv(1, 1) * tv(1, 1) - tv(2, 2) * tv(2, 2))
                           + (tv(0, 1) * tv(0, 1) - tv(0, 2) * tv(0, 2)))
               + tv(1, 1) * ((tv(2, 2) * tv(2, 2) - tv(0, 0) * tv(0, 0))
                             + (tv(1, 2) * tv(1, 2) - tv(0, 1) * tv(0, 1)))
               + tv(2, 2) * ((tv(0, 0) * tv(0, 0) - tv(1, 1) * tv(1, 1))
                             + (tv(0, 2) * tv(0, 2) - tv(1, 2) * tv(1, 2)));
    };

    auto fy1 = [&](const Coords& coords) -> double {
        auto tv = t(coords.head<3>());
        return tv(1, 2) * (2 * (tv(1, 2) * tv(1, 2) - tv(0, 0) * tv(0, 0))
                           - (tv(0, 2) * tv(0, 2) + tv(0, 1) * tv(0, 1))
                           + 2 * (tv(1, 1) * tv(0, 0) + tv(2, 2) * tv(0, 0)
                                  - tv(1, 1) * tv(2, 2)))
               + tv(0, 1) * tv(0, 2) * (2 * tv(0, 0) - tv(2, 2) - tv(1, 1));
    };

    auto fy2 = [&](const Coords& coords) -> double {
        auto tv = t(coords.head<3>());
        return tv(0, 2) * (2 * (tv(0, 2) * tv(0, 2) - tv(1, 1) * tv(1, 1))
                           - (tv(0, 1) * tv(0, 1) + tv(1, 2) * tv(1, 2))
                           + 2 * (tv(2, 2) * tv(1, 1) + tv(0, 0) * tv(1, 1)
                                  - tv(2, 2) * tv(0, 0)))
               + tv(1, 2) * tv(0, 1) * (2 * tv(1, 1) - tv(0, 0) - tv(2, 2));
    };

    auto fy3 = [&](const Coords& coords) -> double {
        auto tv = t(coords.head<3>());
        return tv(0, 1) * (2 * (tv(0, 1) * tv(0, 1) - tv(2, 2) * tv(2, 2))
                           - (tv(1, 2) * tv(1, 2) + tv(0, 2) * tv(0, 2))
                           + 2 * (tv(0, 0) * tv(2, 2) + tv(1, 1) * tv(2, 2)
                                  - tv(0, 0) * tv(1, 1)))
               + tv(0, 2) * tv(1, 2) * (2 * tv(2, 2) - tv(1, 1) - tv(0, 0));
    };

    auto fz1 = [&](const Coords& coords) -> double {
        auto tv = t(coords.head<3>());
        return tv(1, 2) * (tv(0, 2) * tv(0, 2) - tv(0, 1) * tv(0, 1))
               + tv(0, 1) * tv(0, 2) * (tv(1, 1) - tv(2, 2));
    };

    auto fz2 = [&](const Coords& coords) -> double {
        auto tv = t(coords.head<3>());
        return tv(0, 2) * (tv(0, 1) * tv(0, 1) - tv(1, 2) * tv(1, 2))
               + tv(1, 2) * tv(0, 1) * (tv(2, 2) - tv(0, 0));
    };

    auto fz3 = [&](const Coords& coords) -> double {
        auto tv = t(coords.head<3>());
        return tv(0, 1) * (tv(1, 2) * tv(1, 2) - tv(0, 2) * tv(0, 2))
               + tv(0, 2) * tv(1, 2) * (tv(0, 0) - tv(1, 1));
    };

    return {TPBT<double, 3, 0>{fx},
            TPBT<double, 3, 0>{fy1},
            TPBT<double, 3, 0>{fy2},
            TPBT<double, 3, 0>{fy3},
            TPBT<double, 3, 0>{fz1},
            TPBT<double, 3, 0>{fz2},
            TPBT<double, 3, 0>{fz3}};
}


using TSHE = TensorTopologyEvaluator;

TSHE::TensorTopologyEvaluator(const DoubleTri& tri,
                              const TensorInterp& t,
                              const Options& opts)
        : _tri(tri), _target_funcs(tensorTopologyCoeffs(t)), _opts(opts)
{
}


std::array<TSHE, 4> TSHE::split() const
{
    auto part = [&](std::size_t i) {
        return TensorTopologyEvaluator(_tri.split<0>(i),
                                       {_target_funcs[0].split<0>(i),
                                        _target_funcs[1].split<0>(i),
                                        _target_funcs[2].split<0>(i),
                                        _target_funcs[3].split<0>(i),
                                        _target_funcs[4].split<0>(i),
                                        _target_funcs[5].split<0>(i),
                                        _target_funcs[6].split<0>(i)},
                                       _split_level + 1,
                                       _opts);
    };
    return {part(0), part(1), part(2), part(3)};
}


Result TSHE::eval()
{
    // Check if any of the error components can not become zero in the
    // current subdivision triangles
    auto has_nonzero =
            boost::algorithm::any_of(_target_funcs, [](const auto& c) {
                return sameSign(c.coefficients()) != 0;
            });

    // Discard triangles if no roots can occur inside
    if(has_nonzero)
    {
        return Result::Discard;
    }

    // Compute upper bound for target functions
    auto max_error = abs_max_upper_bound(_target_funcs);

    if(max_error < _opts.tolerance)
    {
        return Result::Accept;
    }

    return Result::Split;
}


double TSHE::error() const
{
    return upper_bound_norm(_target_funcs);
}


double distance(const TSHE& t1, const TSHE& t2)
{
    return distance(t1._tri, t2._tri);
}


bool operator==(const TSHE& t1, const TSHE& t2)
{
    return t1._tri == t2._tri && t1._target_funcs == t2._target_funcs
           && t1._split_level == t2._split_level && t1._opts == t2._opts;
}


bool operator!=(const TSHE& t1, const TSHE& t2)
{
    return !(t1 == t2);
}
}

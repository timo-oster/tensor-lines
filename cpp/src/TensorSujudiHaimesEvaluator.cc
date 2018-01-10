#include "TensorSujudiHaimesEvaluator.hh"

#include <Eigen/Geometry>

#include <boost/algorithm/cxx11/any_of.hpp>
#include <boost/range/algorithm/min_element.hpp>
#include <boost/range/algorithm/max_element.hpp>

#include <vector>
#include <tuple>
#include <utility>

using namespace cpp_utils;

namespace pev
{

template <typename T, std::size_t... Degrees>
using TPBT = TensorProductBezierTriangle<T, double, Degrees...>;

std::pair<std::array<TPBT<double, 1, 2>, 3>, std::array<TPBT<double, 0, 3>, 3>>
tensorSujudiHaimesCoeffs(const TensorInterp& t,
                         const std::array<TensorInterp, 3>& dt,
                         const Triangle& r)
{
    using Coords1 = TPBT<double, 1, 2>::Coords;
    using Coords2 = TPBT<double, 0, 3>::Coords;

    // (T * r) x r
    auto eval_ev =
            [&](const Coords1& coords, const TensorInterp& t, int i) -> double {
        auto rv = r(coords.tail<3>());
        auto tv = t(coords.head<3>());
        return (tv * rv).cross(rv)[i];
    };

    // ((\nabla T * r) * r) x r
    auto eval_deriv_ev = [&](const Coords2& coords,
                             const std::array<TensorInterp, 3>& dt,
                             int i) -> double {
        auto rv = r(coords.tail<3>());
        auto txv = dt[0](coords.head<3>());
        auto tyv = dt[1](coords.head<3>());
        auto tzv = dt[2](coords.head<3>());

        return ((txv * rv[0] + tyv * rv[1] + tzv * rv[2]) * rv).cross(rv)[i];
    };

    auto eval1 = [&](const Coords1& coords) { return eval_ev(coords, t, 0); };
    auto eval2 = [&](const Coords1& coords) { return eval_ev(coords, t, 1); };
    auto eval3 = [&](const Coords1& coords) { return eval_ev(coords, t, 2); };

    auto eval4 = [&](const Coords2& coords) {
        return eval_deriv_ev(coords, dt, 0);
    };
    auto eval5 = [&](const Coords2& coords) {
        return eval_deriv_ev(coords, dt, 1);
    };
    auto eval6 = [&](const Coords2& coords) {
        return eval_deriv_ev(coords, dt, 2);
    };

    return {{TPBT<double, 1, 2>{eval1},
             TPBT<double, 1, 2>{eval2},
             TPBT<double, 1, 2>{eval3}},
            {TPBT<double, 0, 3>{eval4},
             TPBT<double, 0, 3>{eval5},
             TPBT<double, 0, 3>{eval6}}};
}


using TSHE = TensorSujudiHaimesEvaluator;

TSHE::TensorSujudiHaimesEvaluator(const DoubleTri& tri,
                                  const TensorInterp& t,
                                  const std::array<TensorInterp, 3>& dt,
                                  const Options& opts)
        : _tri(tri),
          _opts(opts)
{
    std::tie(_target_funcs_t, _target_funcs_dt) =
            tensorSujudiHaimesCoeffs(t, dt, tri.dir_tri);
}


std::array<TSHE, 4> TSHE::split() const
{
    if(_last_split_dir)
    {
        return split<0>();
    }
    return split<1>();
}


Result TSHE::eval()
{
    // Check if any of the error components can not become zero in the
    // current subdivision triangles
    auto has_nonzero =
            boost::algorithm::any_of(_target_funcs_t, [](const auto& c) {
                    return sameSign(c.coefficients()) != 0;
            })
            || boost::algorithm::any_of(_target_funcs_dt, [](const auto& c) {
                   return sameSign(c.coefficients()) != 0;
               });

    // Discard triangles if no roots can occur inside
    if(has_nonzero)
    {
        return Result::Discard;
    }

    // Compute upper bound for target functions
    auto max_error = std::max(abs_max_upper_bound(_target_funcs_t),
                              abs_max_upper_bound(_target_funcs_dt));

    if(max_error < _opts.tolerance)
    {
        return Result::Accept;
    }

    return Result::Split;
}


double TSHE::error() const
{
    return std::max(upper_bound_norm(_target_funcs_t),
                    upper_bound_norm(_target_funcs_dt));
}


double distance(const TSHE& t1, const TSHE& t2)
{
    return distance(t1._tri, t2._tri);
}


bool operator==(const TSHE& t1, const TSHE& t2)
{
    return t1._tri == t2._tri && t1._target_funcs_t == t2._target_funcs_t
           && t1._target_funcs_dt == t2._target_funcs_dt
           && t1._last_split_dir == t2._last_split_dir
           && t1._split_level == t2._split_level && t1._opts == t2._opts;
}


bool operator!=(const TSHE& t1, const TSHE& t2)
{
    return !(t1 == t2);
}

}

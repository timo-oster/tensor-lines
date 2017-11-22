#include "TensorSujudiHaimesEvaluator.hh"

#include <Eigen/Geometry>

#include <boost/algorithm/cxx11/any_of.hpp>
#include <boost/range/algorithm/min_element.hpp>
#include <boost/range/algorithm/max_element.hpp>

#include <vector>
#include <tuple>
#include <utility>

namespace pev
{

template <typename T, std::size_t... Degrees>
using TPBT = TensorProductBezierTriangle<T, double, Degrees...>;

std::array<TPBT<double, 1, 3>, 6>
tensorSujudiHaimesCoeffs(const TensorInterp& t,
                         const std::array<TensorInterp, 3>& dt,
                         const Triangle& r)
{
    using Coords = TPBT<double, 1, 3>::Coords;

    // (T * r) x r
    auto eval_ev =
            [&](const Coords& coords, const TensorInterp& t, int i) -> double {
        auto rv = r(coords.tail<3>());
        auto tv = t(coords.head<3>());
        return (tv * rv).cross(rv)[i];
    };

    // ((\nabla T * r) * r) x r
    auto eval_deriv_ev = [&](const Coords& coords,
                             const std::array<TensorInterp, 3>& dt,
                             int i) -> double {
        auto rv = r(coords.tail<3>());
        auto txv = dt[0](coords.head<3>());
        auto tyv = dt[1](coords.head<3>());
        auto tzv = dt[2](coords.head<3>());

        return ((txv * rv[0] + tyv * rv[1] + tzv * rv[2]) * rv).cross(rv)[i];
    };

    auto eval1 = [&](const Coords& coords) { return eval_ev(coords, t, 0); };
    auto eval2 = [&](const Coords& coords) { return eval_ev(coords, t, 1); };
    auto eval3 = [&](const Coords& coords) { return eval_ev(coords, t, 2); };

    auto eval4 = [&](const Coords& coords) {
        return eval_deriv_ev(coords, dt, 0);
    };
    auto eval5 = [&](const Coords& coords) {
        return eval_deriv_ev(coords, dt, 1);
    };
    auto eval6 = [&](const Coords& coords) {
        return eval_deriv_ev(coords, dt, 2);
    };

    return {TPBT<double, 1, 3>{eval1}, //
            TPBT<double, 1, 3>{eval2}, // degree higher than necessary here
            TPBT<double, 1, 3>{eval3}, //
            TPBT<double, 1, 3>{eval4},
            TPBT<double, 1, 3>{eval5},
            TPBT<double, 1, 3>{eval6}};
}


std::pair<TPBT<double, 2, 2>, TPBT<double, 0, 2>>
tensorSujudiHaimesEstCoeffs(const TensorInterp& t, const Triangle& r)
{
    // ||T * r||^2
    auto eval_tr = [&](const TPBT<double, 2, 2>::Coords& coords) -> double {
        auto rv = r(coords.tail<3>());
        auto tv = t(coords.head<3>());
        return (tv * rv).norm();
    };

    // ||r||^2
    auto eval_r = [&](const TPBT<double, 0, 2>::Coords& coords) -> double {
        return r(coords.tail<3>()).norm();
    };

    return {TPBT<double, 2, 2>{eval_tr}, TPBT<double, 0, 2>{eval_r}};
}


using TSHE = TensorSujudiHaimesEvaluator;

TSHE::TensorSujudiHaimesEvaluator(const DoubleTri& tri,
                                  const TensorInterp& t,
                                  const std::array<TensorInterp, 3>& dt,
                                  const Options& opts)
        : _tri(tri),
          _target_funcs(tensorSujudiHaimesCoeffs(t, dt, tri.dir_tri)),
          _opts(opts)
{
    std::tie(_tr_length, _dir_length) =
            tensorSujudiHaimesEstCoeffs(t, tri.dir_tri);
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
    auto has_nonzero = boost::algorithm::any_of(
            _target_funcs,
            [](const auto& c) { return sameSign(c.coefficients()) != 0; });

    // Discard triangles if no roots can occur inside
    if(has_nonzero)
    {
        return Result::Discard;
    }

    // Discard if maximum possible eigenvalue is too small
    auto max_tr = *boost::max_element(_tr_length.coefficients());
    auto min_dir = *boost::min_element(_dir_length.coefficients());
    if(max_tr / min_dir < _opts.min_ev)
    {
        return Result::Discard;
    }

    // Compute upper bound for target function
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


double TSHE::condition() const
{
    auto gradients = std::vector<Eigen::Vector4d>{};
    gradients.reserve(_target_funcs.size());

    auto min_cos = 1.;

    auto center0 = (TensorProductDerivativeType_t<0, double, double, 1, 3>::
                            Coords::Ones()
                    / 3.).eval();
    auto center1 = (TensorProductDerivativeType_t<1, double, double, 1, 3>::
                            Coords::Ones()
                    / 3.).eval();

    for(const auto& poly: _target_funcs)
    {
        auto deriv0 = derivatives<0>(poly);
        auto deriv1 = derivatives<1>(poly);
        auto grad = Eigen::Vector4d(deriv0[0](center0),
                                    deriv0[1](center0),
                                    deriv1[0](center1),
                                    deriv1[1](center1));
        grad.normalize();
        for(const auto& g: gradients)
        {
            min_cos = std::min(min_cos, std::abs(grad.dot(g)));
        }
        gradients.push_back(grad);
    }

    auto angle = std::acos(min_cos);
    return min_cos / std::sin(angle);
}


double distance(const TSHE& t1, const TSHE& t2)
{
    return distance(t1._tri, t2._tri);
}


bool operator==(const TSHE& t1, const TSHE& t2)
{
    return t1._tri == t2._tri && t1._target_funcs == t2._target_funcs
           && t1._dir_length == t2._dir_length && t1._tr_length == t2._tr_length
           && t1._last_split_dir == t2._last_split_dir
           && t1._split_level == t2._split_level && t1._opts == t2._opts;
}


bool operator!=(const TSHE& t1, const TSHE& t2)
{
    return !(t1 == t2);
}

}

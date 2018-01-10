#include "ParallelEigenvectorsEvaluator.hh"

#include <Eigen/Geometry>

#include <boost/algorithm/cxx11/any_of.hpp>
#include <boost/range/algorithm/min_element.hpp>
#include <boost/range/algorithm/max_element.hpp>

using namespace cpp_utils;

namespace pev
{

template <typename T, std::size_t... Degrees>
using TPBT = TensorProductBezierTriangle<T, double, Degrees...>;

std::array<TPBT<double, 1, 2>, 6>
parallelEigenvectorsCoeffs(const TensorInterp& s,
                           const TensorInterp& t,
                           const Triangle& r)
{
    using Coords = TPBT<double, 1, 2>::Coords;

    // (T * r) x r
    auto eval_ev =
            [&](const Coords& coords, const TensorInterp& t, int i) -> double {
        auto rv = r(coords.tail<3>());
        auto tv = t(coords.head<3>());
        return (tv * rv).cross(rv)[i];
    };

    auto eval1 = [&](const Coords& coords) { return eval_ev(coords, s, 0); };
    auto eval2 = [&](const Coords& coords) { return eval_ev(coords, s, 1); };
    auto eval3 = [&](const Coords& coords) { return eval_ev(coords, s, 2); };

    auto eval4 = [&](const Coords& coords) { return eval_ev(coords, t, 0); };
    auto eval5 = [&](const Coords& coords) { return eval_ev(coords, t, 1); };
    auto eval6 = [&](const Coords& coords) { return eval_ev(coords, t, 2); };

    return {TPBT<double, 1, 2>{eval1},
            TPBT<double, 1, 2>{eval2},
            TPBT<double, 1, 2>{eval3},
            TPBT<double, 1, 2>{eval4},
            TPBT<double, 1, 2>{eval5},
            TPBT<double, 1, 2>{eval6}};
}


using PEVE = ParallelEigenvectorsEvaluator;

PEVE::ParallelEigenvectorsEvaluator(const DoubleTri& tri,
                                    const TensorInterp& s,
                                    const TensorInterp& t,
                                    const Options& opts)
        : _tri(tri),
          _target_funcs(parallelEigenvectorsCoeffs(s, t, tri.dir_tri)),
          _opts(opts)
{
}


std::array<PEVE, 4> PEVE::split() const
{
    if(_last_split_dir)
    {
        return split<0>();
    }
    return split<1>();
}


Result PEVE::eval()
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

    // Compute upper bound for target function
    auto max_error = abs_max_upper_bound(_target_funcs);

    if(max_error < _opts.tolerance)
    {
        return Result::Accept;
    }

    return Result::Split;
}


double PEVE::error() const
{
    return upper_bound_norm(_target_funcs);
}


double PEVE::condition() const
{
    auto gradients = std::vector<Eigen::Vector4d>{};
    gradients.reserve(_target_funcs.size());

    auto min_cos = 0.;

    auto center0 = (TensorProductDerivativeType_t<0, double, double, 1, 2>::
                            Coords::Ones()
                    / 3.).eval();
    auto center1 = (TensorProductDerivativeType_t<1, double, double, 1, 2>::
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


double distance(const PEVE& t1, const PEVE& t2)
{
    return distance(t1._tri, t2._tri);
}


bool operator==(const PEVE& t1, const PEVE& t2)
{
    return t1._tri == t2._tri && t1._target_funcs == t2._target_funcs
           && t1._last_split_dir == t2._last_split_dir
           && t1._split_level == t2._split_level && t1._opts == t2._opts;
}


bool operator!=(const PEVE& t1, const PEVE& t2)
{
    return !(t1 == t2);
}

}// namespace pev

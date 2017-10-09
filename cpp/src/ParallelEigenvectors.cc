#include "ParallelEigenvectors.hh"

#include "TensorProductBezierTriangle.hh"

#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include <boost/algorithm/cxx11/any_of.hpp>
#include <boost/algorithm/minmax_element.hpp>
#include <boost/range/algorithm/max_element.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/transform.hpp>
#include <boost/optional/optional.hpp>
#include <boost/range/join.hpp>
#include <boost/range/algorithm_ext/insert.hpp>

#include <algorithm>
#include <array>
#include <iostream>
#include <map>
#include <queue>
#include <stack>
#include <utility>
#include <vector>

namespace
{
using namespace pev;

/**
 * Tensor Product of linear and quadratic polynomials on barycentric coordinates
 */
using TPBT1_2 = TensorProductBezierTriangle<double, double, 1, 2>;

using TPBT1_3 = TensorProductBezierTriangle<double, double, 1, 3>;


/**
 * Aggregate representing a candidate solution for a parallel eigenvector point.
 *
 * Holds the triangle in direction space that contains the potential
 * eigenvector, and the triangle in barycentric coordinate space that contains
 * the potential eigenvector point.
 */
struct TriPair
{
    Triangle direction_tri;
    Triangle spatial_tri;

    template<int D>
    std::array<TriPair, 4> split()
    {
        static_assert(D >= 0 && D < 2,
                      "Split space must be 0 (position) or 1 (direction)");
        if(D == 0)
        {
            auto spatial_split = spatial_tri.split();
            return {TriPair{direction_tri, spatial_split[0]},
                    TriPair{direction_tri, spatial_split[1]},
                    TriPair{direction_tri, spatial_split[2]},
                    TriPair{direction_tri, spatial_split[3]}};
        }
        else
        {
            auto dir_split = direction_tri.split();
            return {TriPair{dir_split[0], spatial_tri},
                    TriPair{dir_split[1], spatial_tri},
                    TriPair{dir_split[2], spatial_tri},
                    TriPair{dir_split[3], spatial_tri}};
        }
    }

    friend bool operator==(const TriPair& tx1, const TriPair& tx2)
    {
        return tx1.direction_tri == tx2.direction_tri
               && tx1.spatial_tri == tx2.spatial_tri;
    }

    friend bool operator!=(const TriPair& tx1, const TriPair& tx2)
    {
        return !(tx1 == tx2);
    }
};


/**
 * List of parallel eigenvector point candidates
 */
using TriPairList = std::vector<TriPair>;


/**
 * Representative TriPair in a cluster of similar TriPairs
 */
template <typename TPBT>
struct ClusterRepr
{
    std::size_t cluster_size;
    Triangle direction_tri;
    Triangle spatial_tri;;
    std::array<TPBT, 6> poly_funcs;
};


/**
 * Structure for holding information needed during subdivision
 */
template<typename TPBT>
struct SubPackage
{
    TriPair trip;
    std::array<TPBT, 6> poly_funcs;
    bool last_split_dir;
    uint64_t split_level;

    template<std::size_t D>
    std::array<SubPackage, 4> split()
    {
        static_assert(D >= 0 && D < 2,
                      "Split space must be 0 (position) or 1 (direction)");

        auto poly_funcs_subs = std::array<std::array<TPBT, 4>, 6>{};
        for(auto i : range(poly_funcs.size()))
        {
            poly_funcs_subs[i] = poly_funcs[i].template split<D>();
        }

        auto tri_split = trip.split<D>();
        auto part = [&](std::size_t i) {
            return SubPackage{tri_split[i],
                              {poly_funcs_subs[0][i],
                               poly_funcs_subs[1][i],
                               poly_funcs_subs[2][i],
                               poly_funcs_subs[3][i],
                               poly_funcs_subs[4][i],
                               poly_funcs_subs[5][i]},
                              D == 1,
                              split_level + 1};
        };

        return {part(0), part(1), part(2), part(3)};
    }

    friend bool operator==(const SubPackage& s1, const SubPackage& s2)
    {
        return s1.trip == s2.trip && s1.last_split_dir == s2.last_split_dir
               && s1.split_level == s2.split_level
               && s1.poly_funcs == s2.poly_funcs;
    }
};


/**
 * List of parallel eigenvector point candidates
 */
template<typename TPBT>
using ResultList = std::vector<SubPackage<TPBT>>;


#ifdef DRAW_DEBUG

const double red[3] = {255, 0, 0};
const double yellow[3] = {255, 127, 0};
const double green[3] = {0, 255, 0};
const double blue[3] = {128, 255, 255};
const double dark_blue[3] = {0, 0, 255};
const double white[3] = {255, 255, 255};

using Vec2d = Eigen::Vector2d;

std::array<Vec2d, 3> project_tri(const Triangle& tri, bool topdown = true)
{
    auto proj = Eigen::Matrix<double, 2, 3>{};
    static const auto sqrt23 = std::sqrt(2.0) / std::sqrt(3.0);
    if(topdown)
    {
        proj << 1, 1, 0,
                1, -1, 0;
    }
    else
    {
        proj << 1, -1, 0,
                -sqrt23, -sqrt23, sqrt23;
    }
    auto result = std::array<Vec2d, 3>{
            proj * tri[0], proj * tri[1], proj * tri[2]};
    for(auto& p : result)
    {
        p = (p * pos_image.width() / 2)
            + Vec2d{pos_image.width() / 2, pos_image.height() / 2};
    }
    return result;
}


void draw_tri(CImg& image,
              const Triangle& tri,
              const double* color,
              bool fill = true,
              bool topdown = true)
{
    auto projected = project_tri(tri, topdown);
    if(fill)
    {
        image.draw_triangle(int(projected[0].x()),
                            int(projected[0].y()),
                            int(projected[1].x()),
                            int(projected[1].y()),
                            int(projected[2].x()),
                            int(projected[2].y()),
                            color);
    }
    else
    {
        image.draw_triangle(int(projected[0].x()),
                            int(projected[0].y()),
                            int(projected[1].x()),
                            int(projected[1].y()),
                            int(projected[2].x()),
                            int(projected[2].y()),
                            color,
                            1.0,
                            0xffffffff);
    }
}


void draw_cross(CImg& image,
                const Vec3d& pos,
                const double* color,
                bool topdown = true)
{
    static const auto size = 10;
    auto projected = project_tri(Triangle{{pos, pos, pos}}, topdown);
    image.draw_line(int(projected[0].x()) - size,
                    int(projected[0].y()),
                    int(projected[0].x()) + size,
                    int(projected[0].y()),
                    color);
    image.draw_line(int(projected[0].x()),
                    int(projected[0].y() - size),
                    int(projected[0].x()),
                    int(projected[0].y() + size),
                    color);
}

#endif // DRAW_DEBUG


/**
 * (Simplified) distance between two triangles
 */
double distance(const Triangle& t1, const Triangle& t2)
{
    return (t1({1. / 3., 1. / 3., 1. / 3.}) - t2({1. / 3., 1. / 3., 1. / 3.}))
            .norm();
}


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

    auto da = (d1 - d0)/std::sqrt(2);

    auto db = (2 * d2 - d0 - d1) / std::sqrt(6);

    return {da, db};
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
 * Cluster all triangles in a list that are closer than a given distance
 *
 * @param tris List of candidates generated by parallelEigenvectorSearch()
 * @param epsilon Maximum distance of triangles in a cluster
 *
 * @return List of clusters (each cluster is a list of candidates)
 */
template <typename TPBT>
std::vector<ResultList<TPBT>> clusterTris(const ResultList<TPBT>& tris,
                                          double epsilon)
{
    auto classes = std::vector<ResultList<TPBT>>{};
    for(const auto& t : tris)
    {
        classes.push_back({t});
    }

    auto has_close_elements = [&](const ResultList<TPBT>& c1,
                                  const ResultList<TPBT>& c2) {
        if(c1 == c2) return false;
        for(const auto& t1 : c1)
        {
            for(const auto& t2 : c2)
            {
                if(distance(t1.trip.spatial_tri, t2.trip.spatial_tri)
                   <= epsilon)
                {
                    return true;
                }
            }
        }
        return false;
    };

    auto changed = true;
    while(changed)
    {
        changed = false;
        for(auto it = std::begin(classes); it != std::end(classes); ++it)
        {
            auto jt = it;
            ++jt;
            for(; jt != std::end(classes); ++jt)
            {
                if(has_close_elements(*it, *jt))
                {
                    it->insert(std::end(*it), std::begin(*jt), std::end(*jt));
                    jt = classes.erase(jt);
                    --jt;
                    changed = true;
                }
            }
        }
    }
    return classes;
}


/**
 * @brief Select the candidate with the most parallel eigenvectors from each
 *     cluster
 * @details Discards any points that have eigenvectors which are less parallel
 *     than @a parallelity_epsilon
 *
 * @param clusters List of candidate clusters as produced by clusterTris()
 * @param s_interp First tensor field on the triangle
 * @param t_interp Second tensor field on the triangle
 * @param parallelity_epsilon Maximum parallelity error for a candidate to be
 *     considered valid
 *
 * @return List of candidates, each a representative of a cluster
 */
std::vector<ClusterRepr<TPBT1_2>>
findRepresentatives(const std::vector<ResultList<TPBT1_2>>& clusters,
                    const TensorInterp& s_interp,
                    const TensorInterp& t_interp)
{
    auto result = std::vector<ClusterRepr<TPBT1_2>>{};
    for(const auto& c : clusters)
    {
        const auto* min_angle_res = &*(c.cbegin());
        auto min_sin = 1.;
        for(const auto& res : c)
        {
            auto dir = res.trip.direction_tri({1. / 3., 1. / 3., 1. / 3.})
                               .normalized();
            auto center = res.trip.spatial_tri({1. / 3., 1. / 3., 1. / 3.});
            auto s = s_interp(center);
            auto t = t_interp(center);

            // compute error as sum of deviations from input direction
            // after multiplication with tensors
            auto ms = (s * dir).normalized().cross(dir.normalized()).norm()
                      + (t * dir).normalized().cross(dir.normalized()).norm();

            if(ms < min_sin)
            {
                min_sin = ms;
                min_angle_res = &res;
            }
        }

        result.push_back({c.size(),
                          min_angle_res->trip.direction_tri,
                          min_angle_res->trip.spatial_tri,
                          min_angle_res->poly_funcs});
    }
    return result;
}


std::vector<ClusterRepr<TPBT1_3>>
findRepresentativesSH(const std::vector<ResultList<TPBT1_3>>& clusters,
                      const TensorInterp& t_interp,
                      const TensorInterp& tx_interp,
                      const TensorInterp& ty_interp,
                      const TensorInterp& tz_interp)
{
    auto result = std::vector<ClusterRepr<TPBT1_3>>{};
    for(const auto& c : clusters)
    {
        const auto* min_angle_res = &*(c.cbegin());
        auto min_sin = 1.;
        for(const auto& res : c)
        {
            auto dir = res.trip.direction_tri({1. / 3., 1. / 3., 1. / 3.})
                               .normalized();
            auto center = res.trip.spatial_tri({1. / 3., 1. / 3., 1. / 3.});
            auto t = t_interp(center);
            auto tx = tx_interp(center);
            auto ty = ty_interp(center);
            auto tz = tz_interp(center);

            // compute error as sum of deviations from input direction
            // after multiplication with tensors
            auto ms = ((tx * dir[0] + ty * dir[1] + tz * dir[2]) * dir)
                              .normalized()
                              .cross(dir.normalized())
                              .norm()
                      + (t * dir).normalized().cross(dir.normalized()).norm();

            if(ms < min_sin)
            {
                min_sin = ms;
                min_angle_res = &res;
            }
        }

        result.push_back({c.size(),
                          min_angle_res->trip.direction_tri,
                          min_angle_res->trip.spatial_tri,
                          min_angle_res->poly_funcs});
    }
    return result;
}


/**
 * @brief Compute context info for representatives
 * @details Computes global point position, eigenvalue order, presence of other
 *      imaginary eigenvalues, and packs into result list together with point
 *      position, eigenvector direction, eigenvalues.
 *
 * @param representatives TriPairs selected by findRepresentatives()
 * @param s_interp First tensor field on the triangle
 * @param t_interp Second tensor field on the triangle
 * @param tri Spatial triangle
 * @return List of PEVPoints with context info
 */
PointList
computeContextInfo(const std::vector<ClusterRepr<TPBT1_2>>& representatives,
                   const TensorInterp& s_interp,
                   const TensorInterp& t_interp,
                   const Triangle& tri)
{
    auto points = PointList{};
    points.reserve(representatives.size());

    for(const auto& r : representatives)
    {
        auto result_center = r.spatial_tri({1. / 3., 1. / 3., 1. / 3.});
        auto result_dir =
                r.direction_tri({1. / 3., 1. / 3., 1. / 3.}).normalized();

        // We want to know which eigenvector of each tensor field we have
        // found (i.e. corresponding to largest, middle, or smallest
        // eigenvalue)
        // Therefore we explicitly compute the eigenvalues at the result
        // position and check which ones the found eigenvector direction
        // corresponds to.
        // @todo: make this step optional

        auto s = s_interp(result_center);
        auto t = t_interp(result_center);

        // Get eigenvalues from our computed direction
        auto s_real_eigv = (s * result_dir).dot(result_dir);
        auto t_real_eigv = (t * result_dir).dot(result_dir);

        // Compute all eigenvalues using Eigen
        auto s_eigvs = s.eigenvalues().eval();
        auto t_eigvs = t.eigenvalues().eval();

        // Find index of eigenvalue that is closest to the one we computed
        using Vec3c = decltype(s_eigvs);
        auto s_closest_index = Vec3d::Index{0};
        (s_eigvs - Vec3c::Ones() * s_real_eigv)
                .cwiseAbs()
                .minCoeff(&s_closest_index);

        auto t_closest_index = Vec3d::Index{0};
        (t_eigvs - Vec3c::Ones() * t_real_eigv)
                .cwiseAbs()
                .minCoeff(&t_closest_index);

        // Find which of the (real) eigenvalues ours is
        auto count_larger_real = [](double ref,
                                    const std::complex<double>& val) {
            if(val.imag() != 0) return 0;
            if(std::abs(ref) >= std::abs(val.real())) return 0;
            return 1;
        };
        auto s_order = s_eigvs.unaryExpr([&](const std::complex<double>& val) {
                                  return count_larger_real(
                                          s_eigvs[s_closest_index].real(), val);
                              })
                               .sum();
        auto t_order = t_eigvs.unaryExpr([&](const std::complex<double>& val) {
                                  return count_larger_real(
                                          t_eigvs[t_closest_index].real(), val);
                              })
                               .sum();

        // Compute angle between gradients of target functions to measure
        // numerical precision
        auto center0 = (TensorProductDerivativeType<0, double, double, 1, 2>::
                                type::Coords::Ones()
                        / 3.).eval();
        auto center1 = (TensorProductDerivativeType<1, double, double, 1, 2>::
                                type::Coords::Ones()
                        / 3.).eval();

        auto gradients = std::vector<Eigen::Vector4d>{};
        gradients.reserve(r.poly_funcs.size());

        auto max_cos = 0.;

        for(const auto& poly: r.poly_funcs)
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
                max_cos = std::max(max_cos, std::abs(grad.dot(g)));
            }
            gradients.push_back(grad);
        }

        points.push_back(
                PEVPoint{tri(result_center),
                         ERank(s_order),
                         ERank(t_order),
                         result_dir,
                         s_real_eigv,
                         t_real_eigv,
                         s_eigvs.sum().imag() != 0,
                         t_eigvs.sum().imag() != 0,
                         r.cluster_size,
                         (r.spatial_tri[1] - r.spatial_tri[0]).norm(),
                         (r.direction_tri[1] - r.direction_tri[0]).norm(),
                         std::acos(max_cos)});
    }
    return points;
}


PointList
computeContextInfoSH(const std::vector<ClusterRepr<TPBT1_3>>& representatives,
                     const TensorInterp& t_interp,
                     const TensorInterp& tx_interp,
                     const TensorInterp& ty_interp,
                     const TensorInterp& tz_interp,
                     const Triangle& tri)
{
    auto points = PointList{};
    points.reserve(representatives.size());

    for(const auto& r : representatives)
    {
        auto result_center = r.spatial_tri({1. / 3., 1. / 3., 1. / 3.});
        auto result_dir =
                r.direction_tri({1. / 3., 1. / 3., 1. / 3.}).normalized();

        // We want to know which eigenvector of each tensor field we have
        // found (i.e. corresponding to largest, middle, or smallest
        // eigenvalue)
        // Therefore we explicitly compute the eigenvalues at the result
        // position and check which ones the found eigenvector direction
        // corresponds to.
        // @todo: make this step optional

        auto t = t_interp(result_center);
        auto tx = tx_interp(result_center);
        auto ty = ty_interp(result_center);
        auto tz = tz_interp(result_center);
        auto dt = (tx * result_dir[0] + ty * result_dir[1] + tz * result_dir[2])
                          .eval();

        // Get eigenvalues from our computed direction
        auto t_real_eigv = (t * result_dir).dot(result_dir);
        auto dt_real_eigv = (dt * result_dir).dot(result_dir);

        // Compute all eigenvalues using Eigen
        auto t_eigvs = t.eigenvalues().eval();
        auto dt_eigvs = dt.eigenvalues().eval();

        // Find index of eigenvalue that is closest to the one we computed
        using Vec3c = decltype(t_eigvs);
        auto t_closest_index = Vec3d::Index{0};
        (t_eigvs - Vec3c::Ones() * t_real_eigv)
                .cwiseAbs()
                .minCoeff(&t_closest_index);

        auto dt_closest_index = Vec3d::Index{0};
        (dt_eigvs - Vec3c::Ones() * dt_real_eigv)
                .cwiseAbs()
                .minCoeff(&dt_closest_index);

        // Find which of the (real) eigenvalues ours is
        auto count_larger_real = [](double ref,
                                    const std::complex<double>& val) {
            if(val.imag() != 0) return 0;
            if(std::abs(ref) >= std::abs(val.real())) return 0;
            return 1;
        };
        auto t_order = t_eigvs.unaryExpr([&](const std::complex<double>& val) {
                                  return count_larger_real(
                                          t_eigvs[t_closest_index].real(), val);
                              })
                               .sum();
        auto dt_order =
                dt_eigvs.unaryExpr([&](const std::complex<double>& val) {
                            return count_larger_real(
                                    dt_eigvs[dt_closest_index].real(), val);
                        })
                        .sum();

        // Compute angle between gradients of target functions to measure
        // numerical precision
        auto center0 = (TensorProductDerivativeType<0, double, double, 1, 3>::
                                type::Coords::Ones()
                        / 3.).eval();
        auto center1 = (TensorProductDerivativeType<1, double, double, 1, 3>::
                                type::Coords::Ones()
                        / 3.).eval();

        auto gradients = std::vector<Eigen::Vector4d>{};
        gradients.reserve(r.poly_funcs.size());

        auto max_cos = 0.;

        for(const auto& poly: r.poly_funcs)
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
                max_cos = std::max(max_cos, std::abs(grad.dot(g)));
            }
            gradients.push_back(grad);
        }

        points.push_back(
                PEVPoint{tri(result_center),
                         ERank(t_order),
                         ERank(dt_order),
                         result_dir,
                         t_real_eigv,
                         dt_real_eigv,
                         t_eigvs.sum().imag() != 0,
                         dt_eigvs.sum().imag() != 0,
                         r.cluster_size,
                         (r.spatial_tri[1] - r.spatial_tri[0]).norm(),
                         (r.direction_tri[1] - r.direction_tri[0]).norm(),
                         std::acos(max_cos)});
    }
    return points;
}


/**
 * Check if all coefficients are positive or negative
 *
 * @return 1 for all positive, -1 for all negative, 0 otherwise
 */
template <typename TPBT>
int sameSign(const TPBT& coeffs)
{
    auto mima = std::minmax_element(std::begin(coeffs.coefficients()),
                                    std::end(coeffs.coefficients()));
    return *(mima.first) * *(mima.second) > 0 ? sgn(*(mima.second)) : 0;
}


/**
 * @brief Compute coefficients of BezierDoubleTriangle describing the error
 *     function of \a t over the direction triangle \a r
 *
 * @param t Tensor field in the spatial triangle
 * @param r Triangle in direction space
 *
 * @return One BezierDoubleTriangle for each barycentric coordinate
 */
std::array<TPBT1_2, 6> parallelEigenvectorCoeffs(const TensorInterp& s,
                                                    const TensorInterp& t,
                                                    const Triangle& r)
{
    using Coords = TPBT1_2::Coords;

    // T * r x r
    auto eval_func =
            [&](const Coords& coords, const TensorInterp& t, int i) -> double {
        return (t(coords.head<3>()) * r(coords.tail<3>()))
                .cross(r(coords.tail<3>()))[i];
    };
    auto eval1 = [&](const Coords& coords) { return eval_func(coords, s, 0); };
    auto eval2 = [&](const Coords& coords) { return eval_func(coords, s, 1); };
    auto eval3 = [&](const Coords& coords) { return eval_func(coords, s, 2); };
    auto eval4 = [&](const Coords& coords) { return eval_func(coords, t, 0); };
    auto eval5 = [&](const Coords& coords) { return eval_func(coords, t, 1); };
    auto eval6 = [&](const Coords& coords) { return eval_func(coords, t, 2); };

    return {TPBT1_2{eval1}, TPBT1_2{eval2}, TPBT1_2{eval3},
            TPBT1_2{eval4}, TPBT1_2{eval5}, TPBT1_2{eval6}};
}


std::array<TPBT1_3, 6> tensorSujudiHaimesCoeffs(const TensorInterp& t,
                                                   const TensorInterp& tx,
                                                   const TensorInterp& ty,
                                                   const TensorInterp& tz,
                                                   const Triangle& r)
{
    using Coords = TPBT1_3::Coords;

    // T * r x r
    auto eval_ev = [&](const Coords& coords,
                       const TensorInterp& t,
                       int i) -> double {
        auto rv = r(coords.tail<3>());
        return (t(coords.head<3>()) * rv).cross(rv)[i];
    };

    // (\nabla T * r) * r x r
    auto eval_deriv_ev = [&](const Coords& coords,
                             const TensorInterp& tx,
                             const TensorInterp& ty,
                             const TensorInterp& tz,
                             int i) -> double {
        auto rv = r(coords.tail<3>());

        return ((tx(coords.head<3>()) * rv[0] + ty(coords.head<3>()) * rv[1]
                 + tz(coords.head<3>()) * rv[2])
                * rv)
                .cross(rv)[i];
    };

    auto eval1 = [&](const Coords& coords) { return eval_ev(coords, t, 0); };
    auto eval2 = [&](const Coords& coords) { return eval_ev(coords, t, 1); };
    auto eval3 = [&](const Coords& coords) { return eval_ev(coords, t, 2); };

    auto eval4 = [&](const Coords& coords) {
        return eval_deriv_ev(coords, tx, ty, tz, 0);
    };
    auto eval5 = [&](const Coords& coords) {
        return eval_deriv_ev(coords, tx, ty, tz, 1);
    };
    auto eval6 = [&](const Coords& coords) {
        return eval_deriv_ev(coords, tx, ty, tz, 2);
    };

    return {TPBT1_3{eval1},
            TPBT1_3{eval2},
            TPBT1_3{eval3},
            TPBT1_3{eval4},
            TPBT1_3{eval5},
            TPBT1_3{eval6}};
}


template <typename TPBT>
ResultList<TPBT> rootSearch(const std::array<TPBT, 6>& polys,
                            const TriPair& trip,
                            double spatial_epsilon,
                            double direction_epsilon,
                            uint64_t* num_splits = nullptr,
                            uint64_t* max_level = nullptr)
{
    auto tstck = std::stack<SubPackage<TPBT>>{};
    tstck.push({trip, polys, true, 0});
    auto result = ResultList<TPBT>{};

    while(!tstck.empty())
    {
        auto pack = tstck.top();
        tstck.pop();
        if(num_splits) *num_splits += 1;
        if(max_level && *max_level < pack.split_level)
        {
            *max_level = pack.split_level;
        }

        // Check if any of the error components can not become zero in the
        // current subdivision triangles
        auto has_nonzero = boost::algorithm::any_of(
                pack.poly_funcs,
                [](const TPBT& c) { return sameSign(c) != 0; });

        // Discard triangles if no roots can occur inside
        if(has_nonzero)
        {
#ifdef DRAW_DEBUG
            draw_tri(pos_image, pack.trip.spatial_tri, red, false, false);
            draw_tri(dir_image, pack.trip.direction_tri, red, false, true);
            pos_frame.display(pos_image);
            dir_frame.display(dir_image);
#endif
            continue;
        }

        auto max_deriv_space = 0.;
        for(const auto& p: pack.poly_funcs)
        {
            auto upper_bound = derivatives_upper_bound<0>(p);
            max_deriv_space = std::max(upper_bound, max_deriv_space);
        }

        auto max_deriv_dir = 0.;
        for(const auto& p: pack.poly_funcs)
        {
            auto upper_bound = derivatives_upper_bound<1>(p);
            max_deriv_dir = std::max(upper_bound, max_deriv_dir);
        }

        // If maximum subdivision accuracy reached, accept point as solution
        auto dir_sub_reached = max_deriv_dir < direction_epsilon;
        auto pos_sub_reached = max_deriv_space < spatial_epsilon;

        if(pos_sub_reached && dir_sub_reached)
        {
#ifdef DRAW_DEBUG
            draw_tri(pos_image, pack.trip.spatial_tri, green, false, false);
            draw_tri(dir_image, pack.trip.direction_tri, green, false, true);
            pos_frame.display(pos_image);
            dir_frame.display(dir_image);
#endif
            result.push_back(pack);
            continue;
        }

        // Alternating subdivision in position and direction space
        if(pack.last_split_dir && !pos_sub_reached)
        {
            for(const auto& p : pack.template split<0>())
            {
                tstck.push(p);
            }
        }
        else
        {
            for(const auto& p : pack.template split<1>())
            {
                tstck.push(p);
            }
        }

        if(result.size() > 100000)
        {
            std::cerr << "Aborting search due to too many solutions"
                      << std::endl;
            break;
        }

#ifdef DRAW_DEBUG
        draw_tri(pos_image, pack.trip.spatial_tri, yellow, false, false);
        draw_tri(dir_image, pack.trip.direction_tri, yellow, false, true);
        pos_frame.display(pos_image);
        dir_frame.display(dir_image);
#endif
    }

    return result;
}


ResultList<TPBT1_2> parallelEigenvectorSearch(const TensorInterp& s,
                                              const TensorInterp& t,
                                              const Triangle& tri,
                                              double spatial_epsilon,
                                              double direction_epsilon,
                                              uint64_t* num_splits = nullptr,
                                              uint64_t* max_level = nullptr)
{
    auto result = ResultList<TPBT1_2>{};

    auto compute_tri =[&](const Triangle& r) {
        boost::insert(result,
                      result.end(),
                      rootSearch(parallelEigenvectorCoeffs(s, t, r),
                                 {r, tri},
                                 spatial_epsilon,
                                 direction_epsilon,
                                 num_splits,
                                 max_level));
#ifdef DRAW_DEBUG
    pos_image.fill(0);
    dir_image.fill(0);
#endif
    };

    // Four triangles covering hemisphere
    compute_tri(Triangle{{Vec3d{1, 0, 0}, Vec3d{0, 1, 0}, Vec3d{0, 0, 1}}});
    compute_tri(Triangle{{Vec3d{0, 1, 0}, Vec3d{-1, 0, 0}, Vec3d{0, 0, 1}}});
    compute_tri(Triangle{{Vec3d{-1, 0, 0}, Vec3d{0, -1, 0}, Vec3d{0, 0, 1}}});
    compute_tri(Triangle{{Vec3d{0, -1, 0}, Vec3d{1, 0, 0}, Vec3d{0, 0, 1}}});

    return result;
}


ResultList<TPBT1_3> tensorSujudiHaimesSearch(const TensorInterp& t,
                                             const TensorInterp& tx,
                                             const TensorInterp& ty,
                                             const TensorInterp& tz,
                                             const Triangle& tri,
                                             double spatial_epsilon,
                                             double direction_epsilon,
                                             uint64_t* num_splits = nullptr,
                                             uint64_t* max_level = nullptr)
{
    auto result = ResultList<TPBT1_3>{};

    auto compute_tri =[&](const Triangle& r) {
        boost::insert(result,
                      result.end(),
                      rootSearch(tensorSujudiHaimesCoeffs(t, tx, ty, tz, r),
                                 {r, tri},
                                 spatial_epsilon,
                                 direction_epsilon,
                                 num_splits,
                                 max_level));
#ifdef DRAW_DEBUG
    pos_image.fill(0);
    dir_image.fill(0);
#endif
    };

    // Four triangles covering hemisphere
    compute_tri(Triangle{{Vec3d{1, 0, 0}, Vec3d{0, 1, 0}, Vec3d{0, 0, 1}}});
    compute_tri(Triangle{{Vec3d{0, 1, 0}, Vec3d{-1, 0, 0}, Vec3d{0, 0, 1}}});
    compute_tri(Triangle{{Vec3d{-1, 0, 0}, Vec3d{0, -1, 0}, Vec3d{0, 0, 1}}});
    compute_tri(Triangle{{Vec3d{0, -1, 0}, Vec3d{1, 0, 0}, Vec3d{0, 0, 1}}});

    return result;
}

} // namespace


namespace pev
{
#ifdef DRAW_DEBUG
CImg pos_image(1024, 1024, 1, 3);
CImg dir_image(1024, 1024, 1, 3);
CImgDisplay pos_frame;
CImgDisplay dir_frame;
#endif

PointList findParallelEigenvectors(const TensorInterp& s,
                                   const TensorInterp& t,
                                   const Triangle& x,
                                   const PEVOptions& opts)
{
    auto start_tri =
            Triangle{{Vec3d{1., 0., 0.}, Vec3d{0., 1., 0.}, Vec3d{0., 0., 1.}}};

    auto num_splits = uint64_t{0};
    auto max_level = uint64_t{0};
    auto tris = parallelEigenvectorSearch(s,
                                          t,
                                          start_tri,
                                          opts.spatial_epsilon,
                                          opts.direction_epsilon,
                                          &num_splits,
                                          &max_level);

    auto clustered_tris = clusterTris(tris, opts.cluster_epsilon);

    auto representatives = findRepresentatives(clustered_tris, s, t);

    return computeContextInfo(representatives, s, t, x);
}


PointList findParallelEigenvectors(const TensorInterp& s,
                                   const TensorInterp& t,
                                   const PEVOptions& opts)
{
    return findParallelEigenvectors(
            s,
            t,
            Triangle{{Vec3d{1., 0., 0.}, Vec3d{0., 1., 0.}, Vec3d{0., 0., 1.}}},
            opts);
}


PointList findTensorSujudiHaimes(const TensorInterp& t,
                                 const TensorInterp& tx,
                                 const TensorInterp& ty,
                                 const TensorInterp& tz,
                                 const Triangle& x,
                                 const PEVOptions& opts)
{
    auto start_tri =
            Triangle{{Vec3d{1., 0., 0.}, Vec3d{0., 1., 0.}, Vec3d{0., 0., 1.}}};

    auto num_splits = uint64_t{0};
    auto max_level = uint64_t{0};

    auto tris = tensorSujudiHaimesSearch(t,
                                         tx,
                                         ty,
                                         tz,
                                         start_tri,
                                         opts.spatial_epsilon,
                                         opts.direction_epsilon,
                                         &num_splits,
                                         &max_level);

    auto clustered_tris = clusterTris(tris, opts.cluster_epsilon);

    auto representatives = findRepresentativesSH(clustered_tris, t, tx, ty, tz);

    return computeContextInfoSH(representatives, t, tx, ty, tz, x);
}


PointList findTensorSujudiHaimes(const TensorInterp& t,
                                 const TensorInterp& tx,
                                 const TensorInterp& ty,
                                 const TensorInterp& tz,
                                 const PEVOptions& opts)
{
    return findTensorSujudiHaimes(
            t,
            tx,
            ty,
            tz,
            Triangle{{Vec3d{1., 0., 0.}, Vec3d{0., 1., 0.}, Vec3d{0., 0., 1.}}},
            opts);
}

} // namespace pev

#include "ParallelEigenvectors.hh"

#include "TensorProductBezierTriangle.hh"

#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include <boost/algorithm/cxx11/any_of.hpp>
#include <boost/algorithm/minmax_element.hpp>
#include <boost/optional/optional.hpp>
#include <boost/range/join.hpp>

#include <algorithm>
#include <array>
#include <iostream>
#include <map>
#include <queue>
#include <stack>
#include <utility>
#include <vector>

template class pev::TensorProductBezierTriangle<double, double, 1>;

namespace
{
using namespace pev;

/**
 * Tensor Product of linear and quadratic polynomials on barycentric coordinates
 */
using BDoubleTri = TensorProductBezierTriangle<double, double, 1, 2>;


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
struct ClusterRepr
{
    std::size_t cluster_size;
    Triangle direction_tri;
    Triangle spatial_tri;
};

/**
 * Structure for holding information needed during subdivision
 */
struct SubPackage
{
    TriPair trip;
    std::array<BDoubleTri, 6> poly_funcs;
    bool last_split_dir;
    int split_level;

    template<int D>
    std::array<SubPackage, 4> split()
    {
        static_assert(D >= 0 && D < 2,
                      "Split space must be 0 (position) or 1 (direction)");

        auto poly_funcs_subs = std::array<std::array<BDoubleTri, 4>, 6>{};
        for(auto i : range(poly_funcs.size()))
        {
            poly_funcs_subs[i] = poly_funcs[i].split<D>();
        }

        auto tri_split = trip.split<D>();
        auto part = [&](int i) {
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
};


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
            proj * tri.v1(), proj * tri.v2(), proj * tri.v3()};
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
    auto projected = project_tri(Triangle{pos, pos, pos}, topdown);
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

#endif


/**
 * (Simplified) distance between two triangles
 */
double distance(const Triangle& t1, const Triangle& t2)
{
    return (t1({1. / 3., 1. / 3., 1. / 3.}) - t2({1. / 3., 1. / 3., 1. / 3.}))
            .norm();
}


/**
 * Cluster all triangles in a list that are closer than a given distance
 *
 * @param tris List of candidates generated by parallelEigenvectorSearch()
 * @param epsilon Maximum distance of triangles in a cluster
 *
 * @return List of clusters (each cluster is a list of candidates)
 */
std::vector<TriPairList> clusterTris(const TriPairList& tris, double epsilon)
{
    auto classes = std::vector<TriPairList>{};
    for(const auto& t : tris)
    {
        classes.push_back({t});
    }

    auto has_close_elements = [&](const TriPairList& c1,
                                  const TriPairList& c2) {
        if(c1 == c2) return false;
        for(const auto& t1 : c1)
        {
            for(const auto& t2 : c2)
            {
                if(distance(t1.spatial_tri, t2.spatial_tri) <= epsilon)
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
std::vector<ClusterRepr>
findRepresentatives(const std::vector<TriPairList>& clusters,
                    const TensorInterp& s_interp,
                    const TensorInterp& t_interp)
{
    auto result = std::vector<ClusterRepr>{};
    for(const auto& c : clusters)
    {
        const auto* min_angle_tri = &*(c.cbegin());
        auto min_sin = 1.;
        for(const auto& trip : c)
        {
            auto dir =
                    trip.direction_tri({1. / 3., 1. / 3., 1. / 3.}).normalized();
            auto center = trip.spatial_tri({1. / 3., 1. / 3., 1. / 3.});
            auto s = s_interp(center);
            auto t = t_interp(center);

            // compute error as sum of deviations from input direction
            // after multiplication with tensors
            auto ms = (s * dir).normalized().cross(dir.normalized()).norm()
                      + (t * dir).normalized().cross(dir.normalized()).norm();

            if(ms < min_sin)
            {
                min_sin = ms;
                min_angle_tri = &trip;
            }
        }

        result.push_back({c.size(),
                          min_angle_tri->direction_tri,
                          min_angle_tri->spatial_tri});
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
PointList computeContextInfo(const std::vector<ClusterRepr>& representatives,
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

        points.push_back(PEVPoint{tri(result_center),
                                  ERank(s_order),
                                  ERank(t_order),
                                  result_dir,
                                  s_real_eigv,
                                  t_real_eigv,
                                  s_eigvs.sum().imag() != 0,
                                  t_eigvs.sum().imag() != 0,
                                  r.cluster_size});
    }
    return points;
}


/**
 * Check if all coefficients are positive or negative
 *
 * @return 1 for all positive, -1 for all negative, 0 otherwise
 */
int sameSign(const BDoubleTri& coeffs)
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
std::array<BDoubleTri, 6> bezierDoubleCoeffs(const TensorInterp& s,
                                             const TensorInterp& t,
                                             const Triangle& r)
{
    using Coords = BDoubleTri::Coords;

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

    return {BDoubleTri{eval1}, BDoubleTri{eval2}, BDoubleTri{eval3},
            BDoubleTri{eval4}, BDoubleTri{eval5}, BDoubleTri{eval6}};
}


TriPairList parallelEigenvectorSearch(const TensorInterp& s,
                                      const TensorInterp& t,
                                      const Triangle& tri,
                                      double spatial_epsilon,
                                      double direction_epsilon,
                                      uint64_t* num_splits = nullptr,
                                      uint64_t* max_level = nullptr)
{
#ifdef DRAW_DEBUG
    pos_image.fill(0);
    dir_image.fill(0);
#endif

    auto tstck = std::stack<SubPackage>{};

    auto init_tri = [&](const Triangle& r) {
        auto poly_coeffs = bezierDoubleCoeffs(s, t, r);
        tstck.push({{r, tri}, poly_coeffs, true, 0});
    };

    // Start with four triangles covering hemisphere
    init_tri(Triangle{{Vec3d{1, 0, 0}, Vec3d{0, 1, 0}, Vec3d{0, 0, 1}}});
    init_tri(Triangle{{Vec3d{0, 1, 0}, Vec3d{-1, 0, 0}, Vec3d{0, 0, 1}}});
    init_tri(Triangle{{Vec3d{-1, 0, 0}, Vec3d{0, -1, 0}, Vec3d{0, 0, 1}}});
    init_tri(Triangle{{Vec3d{0, -1, 0}, Vec3d{1, 0, 0}, Vec3d{0, 0, 1}}});

    auto result = TriPairList{};

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
                [](const BDoubleTri& c) { return sameSign(c) != 0; });

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

        // If maximum subdivision accuracy reached, accept point as solution
        auto dir_sub_reached =
                (pack.trip.direction_tri[0] - pack.trip.direction_tri[1])
                        .norm()
                < direction_epsilon;
        auto pos_sub_reached =
                (pack.trip.spatial_tri[0] - pack.trip.spatial_tri[1]).norm()
                < spatial_epsilon;

        if(pos_sub_reached && dir_sub_reached)
        {
#ifdef DRAW_DEBUG
            draw_tri(pos_image, pack.trip.spatial_tri, green, false, false);
            draw_tri(dir_image, pack.trip.direction_tri, green, false, true);
            pos_frame.display(pos_image);
            dir_frame.display(dir_image);
#endif
            result.push_back(pack.trip);
            continue;
        }

        // Alternating subdivision in position and direction space
        if(pack.last_split_dir && !pos_sub_reached)
        {
            for(const auto& p : pack.split<0>())
            {
                tstck.push(p);
            }
        }
        else
        {
            for(const auto& p : pack.split<1>())
            {
                tstck.push(p);
            }
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

} // namespace pev

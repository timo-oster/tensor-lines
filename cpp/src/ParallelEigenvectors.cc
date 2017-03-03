#include "ParallelEigenvectors.hh"

#include "BarycentricInterpolator.hh"

#include <Eigen/LU>
#include <Eigen/Eigenvalues>

#include <boost/range.hpp>
#include <boost/range/adaptors.hpp>
#include <boost/algorithm/cxx11/any_of.hpp>
#include <boost/algorithm/cxx11/all_of.hpp>
#include <boost/range/algorithm_ext/insert.hpp>
#include <boost/range/algorithm/max_element.hpp>
#include <boost/range/algorithm/min_element.hpp>
#include <boost/optional/optional.hpp>

#include <iostream>
#include <utility>
#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <queue>
#include <stack>

#include <vtkPolyData.h>
#include <vtkPolyDataWriter.h>
#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkDoubleArray.h>
#include <vtkCellData.h>
#include <vtkIdList.h>
#include <vtkMergePoints.h>
#include <vtkCellArray.h>

namespace
{

using namespace peigv;
using vec2d = Eigen::Vector2d;
using bcoeffs = std::array<double, 10>;

enum BezierIndices : int
{
    i300 = 0,
    i210,
    i201,
    i030,
    i120,
    i021,
    i003,
    i102,
    i012,
    i111
};

using TensorInterp = BarycetricInterpolator<mat3d>;
using Triangle = BarycetricInterpolator<vec3d>;
using TriPair = std::pair<Triangle, Triangle>;

using tri_list = std::list<Triangle>;
using tri_pair_list = std::list<TriPair>;


struct VecComp
{
    bool operator()(const vec3d& v1, const vec3d& v2) const
    {
        if(v1.x() == v2.x())
        {
            if(v1.y() == v2.y())
            {
                if(v1.z() == v2.z())
                {
                    return false;
                }
                return v1.z() < v2.z();
            }
            return v1.y() < v2.y();
        }
        return v1.x() < v2.x();
    }
};


struct TriComp
{
    bool operator()(const Triangle& t1, const Triangle& t2) const
    {
        static auto vc = VecComp{};

        if(t1.v1() == t2.v1())
        {
            if(t1.v2() == t2.v2())
            {
                if(t1.v3() == t2.v3())
                {
                    return false;
                }
                return vc(t1.v3(), t2.v3());
            }
            return vc(t1.v2(), t2.v2());
        }
        return vc(t1.v1(), t2.v1());
    }
};


struct TriPairComp
{
    bool operator()(const TriPair& t1, const TriPair& t2) const
    {
        static auto tc = TriComp{};
        return tc(t1.second, t2.second);
    }
};


double distance(const Triangle& t1, const Triangle& t2)
{
    return std::min({
        (t1.v1()-t2.v1()).norm(),
        (t1.v2()-t2.v1()).norm(),
        (t1.v3()-t2.v1()).norm(),
        (t1.v1()-t2.v2()).norm(),
        (t1.v2()-t2.v2()).norm(),
        (t1.v3()-t2.v2()).norm(),
        (t1.v1()-t2.v3()).norm(),
        (t1.v2()-t2.v3()).norm(),
        (t1.v3()-t2.v3()).norm()
    });
}


// Cluster all points in a list that are closer than a given distance
std::vector<std::set<TriPair, TriPairComp> >
cluster_tris(const tri_pair_list& tris, double epsilon)
{
    using tri_dir_set = std::set<TriPair, TriPairComp>;

    auto classes = std::vector<tri_dir_set>{};
    for(const auto& t: tris)
    {
        classes.push_back({t});
    }

    auto has_close_elements = [&](const tri_dir_set& c1, const tri_dir_set& c2)
    {
        if(c1 == c2) return false;
        for(const auto& t1: c1)
        {
            for(const auto& t2: c2)
            {
                if(distance(t1.second, t2.second) <= epsilon)
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
        for(auto it = classes.begin(); it != classes.end(); ++it)
        {
            for(auto jt = it+1; jt != classes.end(); ++jt)
            {
                if(has_close_elements(*it, *jt))
                {
                    boost::range::insert(*it, *jt);
                    jt = classes.erase(jt)-1;
                    changed = true;
                }
            }
        }
    }
    return classes;
}


// Check if all coefficients are positive or negative
// @return 1 for all positive, -1 for all negative, 0 otherwise
int same_sign(const bcoeffs& coeffs)
{
    using namespace boost::algorithm;
    if(all_of(coeffs, [](double v){ return v > 0; }))
        return 1;
    if(all_of(coeffs, [](double v){ return v < 0; }))
        return -1;
    return 0;
}


bcoeffs bezierCoefficients(const mat3d& A,
                           const mat3d& B,
                           const mat3d& C)
{
    auto result = bcoeffs{};

    result[i300] = A.determinant();
    result[i030] = B.determinant();
    result[i003] = C.determinant();

    result[i210] = 1. / 3. * (
              (mat3d{} << B.col(0), A.col(1), A.col(2)).finished().determinant()
            + (mat3d{} << A.col(0), B.col(1), A.col(2)).finished().determinant()
            + (mat3d{} << A.col(0), A.col(1), B.col(2)).finished().determinant()
            );

    result[i201] = 1. / 3. * (
              (mat3d{} << C.col(0), A.col(1), A.col(2)).finished().determinant()
            + (mat3d{} << A.col(0), C.col(1), A.col(2)).finished().determinant()
            + (mat3d{} << A.col(0), A.col(1), C.col(2)).finished().determinant()
            );

    result[i120] = 1. / 3. * (
              (mat3d{} << A.col(0), B.col(1), B.col(2)).finished().determinant()
            + (mat3d{} << B.col(0), A.col(1), B.col(2)).finished().determinant()
            + (mat3d{} << B.col(0), B.col(1), A.col(2)).finished().determinant()
            );

    result[i021] = 1. / 3. * (
              (mat3d{} << C.col(0), B.col(1), B.col(2)).finished().determinant()
            + (mat3d{} << B.col(0), C.col(1), B.col(2)).finished().determinant()
            + (mat3d{} << B.col(0), B.col(1), C.col(2)).finished().determinant()
            );

    result[i102] = 1. / 3. * (
              (mat3d{} << A.col(0), C.col(1), C.col(2)).finished().determinant()
            + (mat3d{} << C.col(0), A.col(1), C.col(2)).finished().determinant()
            + (mat3d{} << C.col(0), C.col(1), A.col(2)).finished().determinant()
            );

    result[i012] = 1. / 3. * (
              (mat3d{} << B.col(0), C.col(1), C.col(2)).finished().determinant()
            + (mat3d{} << C.col(0), B.col(1), C.col(2)).finished().determinant()
            + (mat3d{} << C.col(0), C.col(1), B.col(2)).finished().determinant()
            );

    result[i111] = 1. / 6. * (
              (mat3d{} << A.col(0), B.col(1), C.col(2)).finished().determinant()
            + (mat3d{} << A.col(0), C.col(1), B.col(2)).finished().determinant()
            + (mat3d{} << B.col(0), A.col(1), C.col(2)).finished().determinant()
            + (mat3d{} << B.col(0), C.col(1), A.col(2)).finished().determinant()
            + (mat3d{} << C.col(0), A.col(1), B.col(2)).finished().determinant()
            + (mat3d{} << C.col(0), B.col(1), A.col(2)).finished().determinant()
            );

    return result;
}


std::array<bcoeffs, 3>
bezierCoefficients(const mat3d& t1, const mat3d& t2, const mat3d& t3,
                   const vec3d& r1, const vec3d& r2, const vec3d& r3)
{
    auto A = mat3d{};
    A << t1 * r1, t2 * r1, t3 * r1;

    auto B = mat3d{};
    B << t1 * r2, t2 * r2, t3 * r2;

    auto C = mat3d{};
    C << t1 * r3, t2 * r3, t3 * r3;

    auto alpha_coeffs = bezierCoefficients(
            (mat3d{} << r1, A.col(1), A.col(2)).finished(),
            (mat3d{} << r2, B.col(1), B.col(2)).finished(),
            (mat3d{} << r3, C.col(1), C.col(2)).finished());

    auto beta_coeffs = bezierCoefficients(
            (mat3d{} << A.col(0), r1, A.col(2)).finished(),
            (mat3d{} << B.col(0), r2, B.col(2)).finished(),
            (mat3d{} << C.col(0), r3, C.col(2)).finished());

    auto gamma_coeffs = bezierCoefficients(
            (mat3d{} << A.col(0), A.col(1), r1).finished(),
            (mat3d{} << B.col(0), B.col(1), r2).finished(),
            (mat3d{} << C.col(0), C.col(1), r3).finished());

    // auto denom_coeffs = bezierCoefficients(A, B, C);

    return {alpha_coeffs, beta_coeffs, gamma_coeffs};
}


boost::optional<Triangle> eigen_dir_stack(const TensorInterp& s,
                                          const TensorInterp& t,
                                          double epsilon)
{
    auto tstck = std::stack<Triangle>{};

    tstck.push(Triangle{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
    tstck.push(Triangle{{0, 1, 0}, {-1, 0, 0}, {0, 0, 1}});
    tstck.push(Triangle{{-1, 0, 0}, {0, -1, 0}, {0, 0, 1}});
    tstck.push(Triangle{{0, -1, 0}, {1, 0, 0}, {0, 0, 1}});

    while(!tstck.empty())
    {
        auto tri = tstck.top();
        tstck.pop();

        // @todo: further speedup:
        // compute coefficients for alpha, beta, gamma on-demand while checking
        // for different signs. Might skip gamma if alpha and beta already have
        // different signs.
        auto s_coeffs = bezierCoefficients(
                s.v1(), s.v2(), s.v3(),
                tri.v1(), tri.v2(), tri.v3());

        auto s_alpha_sign = same_sign(s_coeffs[0]);
        auto s_beta_sign = same_sign(s_coeffs[1]);
        auto s_gamma_sign = same_sign(s_coeffs[2]);

        // different signs: parallel vectors not possible
        if(std::max({s_alpha_sign, s_beta_sign, s_gamma_sign}) *
            std::min({s_alpha_sign, s_beta_sign, s_gamma_sign}) < 0)
        {
            continue;
        }

        auto t_coeffs = bezierCoefficients(
                t.v1(), t.v2(), t.v3(),
                tri.v1(), tri.v2(), tri.v3());

        auto t_alpha_sign = same_sign(t_coeffs[0]);
        auto t_beta_sign = same_sign(t_coeffs[1]);
        auto t_gamma_sign = same_sign(t_coeffs[2]);

        // different signs: parallel vectors not possible
        if(std::max({t_alpha_sign, t_beta_sign, t_gamma_sign}) *
            std::min({t_alpha_sign, t_beta_sign, t_gamma_sign}) < 0)
        {
            continue;
        }

        // All same signs: parallel vectors possible
        if(std::abs(s_alpha_sign + s_beta_sign + s_gamma_sign) == 3
           && std::abs(t_alpha_sign + t_beta_sign + t_gamma_sign) == 3)
        {
            return tri;
        }

        // Small triangle and still not sure if parallel eigenvectors possible
        // or impossible: assume yes
        if((tri.v1() - tri.v2()).norm() < epsilon)
        {
            return tri;
        }

        auto tri_subs = tri.split();
        for(const auto& tri_sub: tri_subs)
        {
            tstck.push(tri_sub);
        }
    }
    return boost::none;
}


tri_pair_list parallel_eigenvector_search_queue(const TensorInterp& s,
                                                const TensorInterp& t,
                                                const Triangle& tri,
                                                double spatial_epsilon,
                                                double direction_epsilon)
{
    static int file_counter = 0;

    struct SubPackage{
        TensorInterp s;
        TensorInterp t;
        Triangle tri;
    };

    // auto triangles = vtkSmartPointer<vtkPolyData>::New();
    // auto points = vtkSmartPointer<vtkPoints>::New();
    // triangles->SetPoints(points);
    // auto tag = vtkSmartPointer<vtkDoubleArray>::New();
    // tag->SetName("Class");
    // triangles->Allocate();
    // triangles->GetCellData()->SetScalars(tag);
    // // auto locator = vtkSmartPointer<vtkMergePoints>::New();
    // // auto bounds = std::array<double, 6>{0, 1, 0, 1, 0, 1};
    // // locator->InitPointInsertion(points, bounds.data());
    // auto tris = vtkSmartPointer<vtkCellArray>::New();
    // triangles->SetPolys(tris);

    // auto add_tri = [&](const Triangle& tri, double cl)
    // {
    //     auto pid1 = points->InsertNextPoint(tri.v1().data());
    //     auto pid2 = points->InsertNextPoint(tri.v2().data());
    //     auto pid3 = points->InsertNextPoint(tri.v3().data());
    //     auto cell = vtkSmartPointer<vtkIdList>::New();
    //     cell->InsertNextId(pid1);
    //     cell->InsertNextId(pid2);
    //     cell->InsertNextId(pid3);
    //     auto cid = triangles->InsertNextCell(VTK_TRIANGLE, cell);
    //     tag->InsertValue(cid, cl);
    // };

    auto pque = std::queue<SubPackage>{};
    pque.push({s, t, tri});

    auto result = tri_pair_list{};

    while(!pque.empty())
    {
        auto pack = pque.front();
        pque.pop();

        // std::cout << "Spatial Triangle: " << pack.tri << std::endl;

        // std::cout << "S Tensors: " << pack.s << std::endl;
        // std::cout << "T Tensors: " << pack.t << std::endl;

        auto dir = eigen_dir_stack(pack.s, pack.t, direction_epsilon);

        if(!dir)
        {
            // std::cout << "Found no eigenvector directions." << std::endl;
            // add_tri(pack.tri, 0.);
            continue;
        }

        if((pack.tri.v1() - pack.tri.v2()).norm() < spatial_epsilon)
        {
            // std::cout << "Triangle too small, assuming location "
                      // << print(pack.tri(1./3., 1./3., 1./3.))
                      // << " is valid." << std::endl;
            // add_tri(pack.tri, 1.);
            result.push_back(std::make_pair(dir.value(), pack.tri));
            continue;
        }

        // std::cout << "Found candidate point: "
                  // << print(pack.tri(1./3., 1./3., 1./3.))
                  // << ", splitting" << std::endl;

        auto s_subs = pack.s.split();
        auto t_subs = pack.t.split();
        auto tri_subs = pack.tri.split();

        for(auto i: range(s_subs.size()))
        {
            pque.push({s_subs[i], t_subs[i], tri_subs[i]});
        }
    }
    // if(result.size() > 10)
    // {
    //     std::cout << "Number of found candidate points: " << result.size() << std::endl;
    //     auto writer = vtkSmartPointer<vtkPolyDataWriter>::New();
    //     writer->SetFileName(std::string(make_string() << "Subdivision_" << file_counter++ << ".vtk").c_str());
    //     writer->SetInputData(triangles);
    //     writer->Write();
    // }

    return result;
}

} // namespace

namespace peigv
{
// Find parallel eigenvector points on a triangle given two tensors at each
// corner
point_list findParallelEigenvectors(
        const mat3d& s1, const mat3d& s2, const mat3d& s3,
        const mat3d& t1, const mat3d& t2, const mat3d& t3,
        const vec3d& p1, const vec3d& p2, const vec3d& p3,
        double spatial_epsilon, double direction_epsilon,
        double cluster_epsilon, double parallelity_epsilon)
{
    auto tri = Triangle{p1, p2, p3};

    auto start_tri = Triangle{vec3d{1., 0., 0.},
                              vec3d{0., 1., 0.},
                              vec3d{0., 0., 1.}};

    auto s_interp = TensorInterp{s1, s2, s3};
    auto t_interp = TensorInterp{t1, t2, t3};

    auto tris = parallel_eigenvector_search_queue(
            s_interp, t_interp,
            start_tri, spatial_epsilon, direction_epsilon);
    auto clustered_tris = cluster_tris(tris, cluster_epsilon);

    auto points = point_list{};

    for(const auto& c: clustered_tris)
    {
        const auto* min_angle_tri = &*(c.cbegin());
        auto min_sin = 1.;
        for(const auto& tri: c)
        {
            auto dir = tri.first(1./3., 1./3., 1./3.);
            auto center = tri.second(1./3., 1./3., 1./3.);
            auto s = s_interp(center);
            auto t = t_interp(center);

            std::cout << "Checking candidate point " << print(center)
                      << "...";

            // Project direction to a pyramid
            auto norm_dir = [](const vec3d& dir)
            {
                return dir.z() > 0 ? (dir/dir.cwiseAbs().sum()).eval() :
                                     (-dir/dir.cwiseAbs().sum()).eval();
            };

            auto sr = norm_dir(s * dir);
            auto tr = norm_dir(t * dir);

            // compute error as sum of deviations from input direction
            // after multiplication with tensors
            auto ms = sr.normalized().cross(dir.normalized()).norm()
                      + tr.normalized().cross(dir.normalized()).norm();

            auto tri_s = Triangle{norm_dir(s_interp(tri.second.v1()) * dir),
                                  norm_dir(s_interp(tri.second.v2()) * dir),
                                  norm_dir(s_interp(tri.second.v3()) * dir)};

            auto tri_t = Triangle{norm_dir(t_interp(tri.second.v1()) * dir),
                                  norm_dir(t_interp(tri.second.v2()) * dir),
                                  norm_dir(t_interp(tri.second.v3()) * dir)};

            // check if the error measure is low enough to consider the point a
            // parallel eigenvector point

            if(ms > parallelity_epsilon)
            {
                std::cout << "rejected with error " << ms << std::endl;
                continue;
            }

            // check if the original direction is inside the triangle of
            // possible results

            // auto mat_s = mat3d{};
            // mat_s << tri_s.v1(), tri_s.v2(), tri_s.v3();
            // auto coords_s = mat_s
            //         .jacobiSvd(Eigen::ComputeFullU|Eigen::ComputeFullV)
            //         .solve(dir).eval();

            // if(coords_s.cwiseSign().sum() != 3)
            // {
            //     std::cout << "rejected" << std::endl;
            //     continue;
            // }

            // auto mat_t = mat3d{};
            // mat_t << tri_t.v1(), tri_t.v2(), tri_t.v3();
            // auto coords_t = mat_t
            //         .jacobiSvd(Eigen::ComputeFullU|Eigen::ComputeFullV)
            //         .solve(dir).eval();

            // if(coords_t.cwiseSign().sum() != 3)
            // {
            //     std::cout << "rejected" << std::endl;
            //     continue;
            // }

            std::cout << "accepted with error " << ms << std::endl;

            if(ms < min_sin)
            {
                min_sin = ms;
                min_angle_tri = &tri;
            }
        }
        if(min_sin < 1.)
        {
            // We want to know which eigenvector of each tensor field we have
            // found (i.e. corresponding to largest, middle, or smallest
            // eigenvalue)
            // Therefore we explicitly compute the eigenvalues at the result
            // position and check which ones the found eigenvector direction
            // corresponds to.
            auto result_center = (min_angle_tri->second)(1./3., 1./3., 1./3.);
            auto result_dir = (min_angle_tri->first)(1./3., 1./3., 1./3.)
                                    .normalized();

            auto s = s_interp(result_center);
            auto t = t_interp(result_center);

            // Get eigenvalues from our computed direction
            auto s_real_eigv = (s * result_dir).dot(result_dir);
            auto t_real_eigv = (t * result_dir).dot(result_dir);

            // Compute all eigenvalues using Eigen
            auto s_eigvs = s.eigenvalues().eval();
            auto t_eigvs = t.eigenvalues().eval();

            // Find index of eigenvalue that is closest to the one we computed
            using vec3c = decltype(s_eigvs);
            auto s_closest_index = vec3d::Index{0};
            (s_eigvs - vec3c::Ones() * s_real_eigv)
                    .cwiseAbs().minCoeff(&s_closest_index);

            auto t_closest_index = vec3d::Index{0};
            (t_eigvs - vec3c::Ones() * t_real_eigv)
                    .cwiseAbs().minCoeff(&t_closest_index);

            // Find which of the (real) eigenvalues ours is
            auto larger_not_complex = [](double ref,
                                         const std::complex<double>& val)
            {
                if(val.imag() != 0) return 0;
                if(std::abs(ref) >= std::abs(val.real())) return 0;
                return 1;
            };
            auto s_order = s_eigvs.unaryExpr(
                    [&](const std::complex<double>& val)
                    {
                        return larger_not_complex(
                                s_eigvs[s_closest_index].real(), val);
                    }).sum();
            auto t_order = t_eigvs.unaryExpr(
                    [&](const std::complex<double>& val)
                    {
                        return larger_not_complex(
                                t_eigvs[t_closest_index].real(), val);
                    }).sum();

            std::cout << "Final Eigenvector point: "
                      << print(result_center) << std::endl;
            std::cout << "Final Eigenvector direction: "
                      << print(result_dir) << std::endl;
            std::cout << "Final S Eigenvalue: "
                      << s_real_eigv << std::endl;
            std::cout << "Final T Eigenvalue: "
                      << t_real_eigv << std::endl;
            std::cout << "Final S Matrix: \n" << s << std::endl;
            std::cout << "Final T Matrix: \n" << t << std::endl;
            auto errvec = Eigen::Matrix<double, 6, 1>{};
            errvec.topRows<3>() = (s*result_dir).cross(result_dir);
            errvec.bottomRows<3>() = (t*result_dir).cross(result_dir);
            std::cout << "Error vector: \n" << errvec << std::endl;

            points.push_back({tri(result_center),
                              ERank(s_order),
                              ERank(t_order),
                              result_dir,
                              s_real_eigv,
                              t_real_eigv});
        }
    }
    return points;
}


// Compute parallel eigenvector points in barycentric coordinates for a general
// triangle given two tensors at each corner
point_list findParallelEigenvectors(
        const mat3d& s1, const mat3d& s2, const mat3d& s3,
        const mat3d& t1, const mat3d& t2, const mat3d& t3,
        double spatial_epsilon, double direction_epsilon,
        double cluster_epsilon, double parallelity_epsilon)
{
    return findParallelEigenvectors(
            s1, s2, s3, t1, t2, t3,
            vec3d{1, 0, 0}, vec3d{0, 1, 0}, vec3d{0, 0, 1},
            spatial_epsilon, direction_epsilon,
            cluster_epsilon, parallelity_epsilon);
}

} // namespace peigv

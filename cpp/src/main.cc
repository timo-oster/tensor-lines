#include "utils.hh"
#include "BarycentricInterpolator.hh"

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Eigenvalues>

#include <boost/program_options.hpp>
#include <boost/range.hpp>
#include <boost/range/adaptors.hpp>
#include <boost/algorithm/cxx11/any_of.hpp>
#include <boost/algorithm/cxx11/all_of.hpp>
#include <boost/range/algorithm_ext/insert.hpp>
#include <boost/range/algorithm/max_element.hpp>
#include <boost/range/algorithm/min_element.hpp>

#include <utility>
#include <algorithm>
#include <numeric>
#include <array>
#include <list>
#include <map>
#include <set>
#include <queue>
#include <stack>
#include <iostream>
#include <random>

#ifdef DRAW_DEBUG
#include <CImg.h>
#endif

namespace peigv
{

using vec2d = Eigen::Vector2d;
using mat3d = Eigen::Matrix3d;

using bcoeffs = std::array<double, 10>;

using point_list = std::list<vec3d>;

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

using tri_list = std::list<Triangle>;

struct VecComp
{
    const bool operator()(const vec3d& v1, const vec3d& v2)
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
    const bool operator()(const Triangle& t1, const Triangle& t2)
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

#ifdef DRAW_DEBUG
using CImg = cimg_library::CImg<double>;
using CImgDisplay = cimg_library::CImgDisplay;

CImg image(1024, 1024, 1, 3);
CImg image2(1024, 1024, 1, 3);
CImg alpha_image(1024, 1024, 1, 3);
CImgDisplay frame{};
CImgDisplay frame2{};
CImgDisplay frame3{};

const double red[3] = {255, 0, 0};
const double yellow[3] = {255, 255, 0};
const double green[3] = {0, 255, 0};
const double blue[3] = {0.5, 255, 255};


class ColorMap
{
public:
    using Color = std::array<double, 3>;

    ColorMap(const std::string& name = "diff")
    {
        if(name == "diff")
        {
            colors[0.0] = {255., 0., 0.};
            colors[0.5] = {0., 0., 0.};
            colors[1.0] = {0., 255., 0.};
        }
        else
        {
            throw std::invalid_argument("invalid colormap name");
        }
    }

    Color operator()(double pos)
    {
        auto f = colors.upper_bound(pos);
        if(f == colors.begin()) return f->second;
        if(f == colors.end()) return colors.rbegin()->second;
        auto c2 = f->second;
        auto p2 = f->first;
        auto c1 = (--f)->second;
        auto p1 = f->first;
        auto c = Color{};
        auto w = (pos - p1)/(p2 - p1);
        for(auto i: range(c.size()))
        {
            c[i] = (1-w)*c1[i] + w*c2[i];
        }
        return c;
    }

private:
    std::map<double, Color> colors;
};


std::array<vec2d, 3> project_tri(const Triangle& tri, bool topdown = true)
{
    auto proj = Eigen::Matrix<double, 2, 3>{};
    static const auto sqrt23 = std::sqrt(2.0)/std::sqrt(3.0);
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
    auto result = std::array<vec2d, 3>{
        proj*tri.v1(),
        proj*tri.v2(),
        proj*tri.v3()
    };
    for(auto& p: result)
    {
        p = (p * 511) + vec2d(image.width()/2, image.height()/2);
    }
    return result;
}


void draw_tri(CImg& image, CImgDisplay& /*frame*/,
              const Triangle& tri, const double* color,
              bool fill = true, bool topdown=true)
{
    auto projected = project_tri(tri, topdown);
    if(fill)
    {
        image.draw_triangle(int(projected[0].x()), int(projected[0].y()),
                            int(projected[1].x()), int(projected[1].y()),
                            int(projected[2].x()), int(projected[2].y()),
                            color);
    }
    else
    {
        image.draw_triangle(int(projected[0].x()), int(projected[0].y()),
                            int(projected[1].x()), int(projected[1].y()),
                            int(projected[2].x()), int(projected[2].y()),
                            color, 1.0, 0xffffffff);
    }
    // frame.display(image);
}


void draw_interp_tri(CImg& image, const Triangle& tri,
                     double v1, double v2, double v3,
                     double vmin, double vmax,
                     bool topdown=true)
{
    auto projected = project_tri(tri, topdown);
    auto cmap = ColorMap{};
    v1 = (v1-vmin)/(vmax-vmin);
    v2 = (v2-vmin)/(vmax-vmin);
    v3 = (v3-vmin)/(vmax-vmin);

    auto c1 = cmap(v1);
    auto c2 = cmap(v2);
    auto c3 = cmap(v3);

    image.draw_triangle(int(projected[0].x()), int(projected[0].y()),
                        int(projected[1].x()), int(projected[1].y()),
                        int(projected[2].x()), int(projected[2].y()),
                        c1.data(), c2.data(), c3.data());
}

#endif

uint32_t call_count = 0;

// Cluster all points in a list that are closer than a given distance
std::vector<std::set<Triangle, TriComp> > cluster_tris(const tri_list& tris,
                                                  double epsilon)
{
    using tri_set = std::set<Triangle, TriComp>;

    auto classes = std::vector<tri_set>{};
    for(const auto& t: tris)
    {
        classes.push_back({t});
    }

    auto has_close_elements = [&](const tri_set& c1, const tri_set& c2)
    {
        if(c1 == c2) return false;
        for(const auto& t1: c1)
        {
            for(const auto& t2: c2)
            {
                if(distance(t1, t2) <= epsilon)
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

// compute the minimum dot product of (normalized) eigenvectors of s and t
// double min_eigenvector_cos(const mat3d& s, const mat3d& t)
// {
//     auto eigens_s = Eigen::EigenSolver<mat3d>{s};
//     auto eigens_t = Eigen::EigenSolver<mat3d>{t};

//     auto ev_s = eigens_s.eigenvalues().eval();
//     auto ev_t = eigens_t.eigenvalues().eval();

//     auto min_cos = std::numeric_limits<double>::infinity();
//     for(auto i: range(s.cols()))
//     {
//         for(auto j: range(t.cols()))
//         {
//             if(ev_s.col(i).sum().imag() < std::numeric_limits<double>::epsilon()
//                && ev_t.col(i).sum().imag() < std::numeric_limits<double>::epsilon())
//             {
//                 min_cos = std::min(min_cos, std::abs(ev_s.col(i).dot(ev_t.col(j)).real()));
//             }
//         }
//     }
//     return min_cos;
// }

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
    ++call_count;
    auto result = bcoeffs{};

    result[i300] = A.determinant();
    result[i030] = B.determinant();
    result[i003] = C.determinant();

    result[i210] = 1.0 / 3.0 * (
              (mat3d{} << B.col(0), A.col(1), A.col(2)).finished().determinant()
            + (mat3d{} << A.col(0), B.col(1), A.col(2)).finished().determinant()
            + (mat3d{} << A.col(0), A.col(1), B.col(2)).finished().determinant()
            );

    result[i201] = 1.0 / 3.0 * (
              (mat3d{} << C.col(0), A.col(1), A.col(2)).finished().determinant()
            + (mat3d{} << A.col(0), C.col(1), A.col(2)).finished().determinant()
            + (mat3d{} << A.col(0), A.col(1), C.col(2)).finished().determinant()
            );

    result[i120] = 1.0 / 3.0 * (
              (mat3d{} << A.col(0), B.col(1), B.col(2)).finished().determinant()
            + (mat3d{} << B.col(0), A.col(1), B.col(2)).finished().determinant()
            + (mat3d{} << B.col(0), B.col(1), A.col(2)).finished().determinant()
            );

    result[i021] = 1.0 / 3.0 * (
              (mat3d{} << C.col(0), B.col(1), B.col(2)).finished().determinant()
            + (mat3d{} << B.col(0), C.col(1), B.col(2)).finished().determinant()
            + (mat3d{} << B.col(0), B.col(1), C.col(2)).finished().determinant()
            );

    result[i102] = 1.0 / 3.0 * (
              (mat3d{} << A.col(0), C.col(1), C.col(2)).finished().determinant()
            + (mat3d{} << C.col(0), A.col(1), C.col(2)).finished().determinant()
            + (mat3d{} << C.col(0), C.col(1), A.col(2)).finished().determinant()
            );

    result[i012] = 1.0 / 3.0 * (
              (mat3d{} << B.col(0), C.col(1), C.col(2)).finished().determinant()
            + (mat3d{} << C.col(0), B.col(1), C.col(2)).finished().determinant()
            + (mat3d{} << C.col(0), C.col(1), B.col(2)).finished().determinant()
            );

    result[i111] = 1.0 / 6.0 * (
              (mat3d{} << A.col(0), B.col(1), C.col(2)).finished().determinant()
            + (mat3d{} << A.col(0), C.col(1), B.col(2)).finished().determinant()
            + (mat3d{} << B.col(0), A.col(1), C.col(2)).finished().determinant()
            + (mat3d{} << B.col(0), C.col(1), A.col(2)).finished().determinant()
            + (mat3d{} << C.col(0), A.col(1), B.col(2)).finished().determinant()
            + (mat3d{} << C.col(0), B.col(1), A.col(2)).finished().determinant()
            );

    return result;
}

std::array<bcoeffs, 4>
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

    return {alpha_coeffs, beta_coeffs, gamma_coeffs}; //, denom_coeffs};
}


bool eigen_dirs_stack(const TensorInterp& s,
                      const TensorInterp& t,
                      double epsilon)
{
    auto tstck = std::stack<Triangle>{};

    tstck.push(Triangle{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
    tstck.push(Triangle{{0, 1, 0}, {-1, 0, 0}, {0, 0, 1}});
    tstck.push(Triangle{{-1, 0, 0}, {0, -1, 0}, {0, 0, 1}});
    tstck.push(Triangle{{0, -1, 0}, {1, 0, 0}, {0, 0, 1}});

#ifdef DRAW_DEBUG
    // auto found = false;
#endif
    // auto vmin = 0.0;
    // auto vmax = 0.0;
    // auto vin = false;

    while(!tstck.empty())
    {
#ifdef DRAW_DEBUG
        // frame2.display(image2);
#endif
        auto tri = tstck.top();
        tstck.pop();

        // std::cout << "Checking directions between " << print(tri.v1()) <<
        //           ", " << print(tri.v2()) << ", " << print(tri.v3()) << std::endl;

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
        // auto s_denom_sign = same_sign(s_coeffs[3]);

        // different signs: parallel vectors not possible
        if(std::max({s_alpha_sign, s_beta_sign, s_gamma_sign}) *
            std::min({s_alpha_sign, s_beta_sign, s_gamma_sign}) < 0)
        {
#ifdef DRAW_DEBUG
            // draw_tri(image2, frame2, tri, red, false);
#endif
            continue;
        }

        auto t_coeffs = bezierCoefficients(
                t.v1(), t.v2(), t.v3(),
                tri.v1(), tri.v2(), tri.v3());

        auto t_alpha_sign = same_sign(t_coeffs[0]);
        auto t_beta_sign = same_sign(t_coeffs[1]);
        auto t_gamma_sign = same_sign(t_coeffs[2]);
        // auto t_denom_sign = same_sign(t_coeffs[3]);

        // different signs: parallel vectors not possible
        if(std::max({t_alpha_sign, t_beta_sign, t_gamma_sign}) *
            std::min({t_alpha_sign, t_beta_sign, t_gamma_sign}) < 0)
        {
#ifdef DRAW_DEBUG
            // draw_tri(image2, frame2, tri, red, false);
#endif
            continue;
        }

        // All same signs: parallel vectors possible
        if(true //s_denom_sign != 0 && t_denom_sign != 0
           && std::abs(s_alpha_sign + s_beta_sign + s_gamma_sign) == 3
           && std::abs(t_alpha_sign + t_beta_sign + t_gamma_sign) == 3)
        {
#ifdef DRAW_DEBUG
//             draw_tri(image2, frame2, tri, green, false);
//             found = true;
//             continue;
            return true;
#else
            return true;
#endif
        }


        // Small triangle and still not sure if parallel eigenvectors possible
        // or impossible: assume yes
        if((tri.v1() - tri.v2()).norm() < epsilon)
        {
#ifdef DRAW_DEBUG
//             draw_tri(image2, frame2, tri, yellow, false);
//             found = true;
//             continue;
            return true;
#else
            return true;
#endif
        }

        auto tri_subs = tri.split();
        for(const auto& tri_sub: tri_subs)
        {
#ifdef DRAW_DEBUG
            // draw_tri(image2, frame2, tri_sub, blue, false);
#endif
            tstck.push(tri_sub);
        }
    }

#ifdef DRAW_DEBUG
//     return found;
    return false;
#else
    return false;
#endif
}


tri_list parallel_eigenvector_search_queue(const TensorInterp& s,
                                           const TensorInterp& t,
                                           const Triangle& tri,
                                           double spatial_epsilon,
                                           double direction_epsilon)
{
    struct SubPackage{
        TensorInterp t;
        TensorInterp s;
        Triangle tri;
    };

    auto pque = std::queue<SubPackage>{};
    pque.push({s, t, tri});

    auto result_tris = tri_list{};

    // auto count = 1;
    while(!pque.empty())
    {
#ifdef DRAW_DEBUG
        frame.display(image);
#endif
        auto pack = pque.front();
        pque.pop();

#ifdef DRAW_DEBUG
        // image2.fill(0);
#endif
        auto has_dirs = eigen_dirs_stack(pack.s, pack.t, direction_epsilon);
        // if(count == 1)
        // {
        //     image2.save("directions_nodenom.png", count, 6);
        // }
#ifdef DRAW_DEBUG
        // frame2.display(image2);
#endif
        // ++count;
        if(!has_dirs)
        {
#ifdef DRAW_DEBUG
            draw_tri(image, frame, pack.tri, red, false, false);
#endif
            continue;
        }

        if((pack.tri.v1() - pack.tri.v2()).norm() < spatial_epsilon)
        {
#ifdef DRAW_DEBUG
            draw_tri(image, frame, pack.tri, yellow, false, false);
#endif
            result_tris.push_back(pack.tri);
            continue;
        }

        auto s_subs = pack.s.split();
        auto t_subs = pack.t.split();
        auto tri_subs = pack.tri.split();

        for(auto i: range(s_subs.size()))
        {
#ifdef DRAW_DEBUG
            // draw_tri(image, frame, tri_subs[i], blue, false, false);
#endif
            pque.push({s_subs[i], t_subs[i], tri_subs[i]});
        }
    }
    return result_tris;
}


point_list find_parallel_eigenvectors(
        const mat3d& s1, const mat3d& s2, const mat3d& s3,
        const mat3d& t1, const mat3d& t2, const mat3d& t3,
        double spatial_epsilon, double direction_epsilon)
{
    auto start_tri = Triangle{
        vec3d{1, 0, 0},
        vec3d{0, 1, 0},
        vec3d{0, 0, 1}
    };

    auto s_interp = TensorInterp{s1, s2, s3};
    auto t_interp = TensorInterp{t1, t2, t3};

    auto tris = parallel_eigenvector_search_queue(
            s_interp, t_interp,
            start_tri, spatial_epsilon, direction_epsilon);
    auto clustered_tris = cluster_tris(tris, spatial_epsilon/std::sqrt(3));

    auto points = point_list{};

    for(const auto& c: clustered_tris)
    {
        // const auto* min_angle_tri = &*(c.cbegin());
        // auto min_cos = std::numeric_limits<double>::infinity();
        auto avg_point = vec3d{0., 0., 0.};
        for(const auto& tri: c)
        {
            auto center = tri(1.0/3.0, 1.0/3.0, 1.0/3.0);
            auto s = s_interp(center.x(), center.y(), center.z());
            auto t = t_interp(center.x(), center.y(), center.z());

            auto es = Eigen::ComplexEigenSolver<mat3d>(s).eigenvectors().eval();
            auto et = Eigen::ComplexEigenSolver<mat3d>(t).eigenvectors().eval();

            std::cout << "Position: " << print(center) << std::endl;
            std::cout << "S: \n" << s << std::endl;
            std::cout << "T: \n" << t << std::endl;
            std::cout << "Eigenvectors(S): " << es << std::endl;
            std::cout << "Eigenvectors(T): " << et << std::endl;

            // compute eigenvectors of s and t and compute minimum angle
            // auto mc = min_eigenvector_cos(s, t);
            // if(mc < min_cos)
            // {
            //     min_cos = mc;
            //     min_angle_tri = &tri;
            // }
            avg_point = avg_point + center;
        }
#ifdef DRAW_DEBUG
        // draw_tri(image, frame, *min_angle_tri, green, true, false);
#endif
        // points.push_back((*min_angle_tri)(1.0/3.0, 1.0/3.0, 1.0/3.0));
        points.push_back(avg_point/c.size());
    }
    return points;
}

} // namespace peigv

namespace po = boost::program_options;

int main(int argc, char const *argv[])
{
    using namespace peigv;

    auto random_seed = uint32_t{7};
    auto spatial_epsilon = 1e-3;
    auto direction_epsilon = 1e-3;

    try
    {
        po::options_description desc("Allowed options");
        desc.add_options()
        ("help,h", "produce help message")
        ("seed,s", po::value<uint32_t>()->required()->default_value(7),
            "Random seed for tensor generation")
        ("spatial-epsilon,e", po::value<double>()->required()->default_value(1e-3),
            "epsilon for spatial subdivision")
        ("direction-epsilon,d", po::value<double>()->required()->default_value(1e-3),
            "epsilon for directional subdivision")
        ;

        auto vm = po::variables_map{};
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if(vm.empty() || vm.count("help"))
        {
            std::cout << desc << "\n";
            return 0;
        }

        po::notify(vm);

        random_seed = vm["seed"].as<uint32_t>();
        spatial_epsilon = vm["spatial-epsilon"].as<double>();
        direction_epsilon = vm["direction-epsilon"].as<double>();
    }
    catch (std::exception& e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch (...)
    {
        std::cerr << "Exception of unknown type!\n";
    }
    // good seeds:
    // 7 (3 solutions)
    // 9 (1 solution)
    // 10 (3 solutions, two badly conditioned)
    // 11 (4 solutions, all weirdly shaped)
    // 16 (5 solutions, one a near curv shape)
    // 20 (2 solutions, one badly conditioned)
    auto gen = std::mt19937{random_seed};
    auto rnd = std::uniform_real_distribution<>{-1.0, 1.0};

    auto s1 = mat3d{};
    auto s2 = mat3d{};
    auto s3 = mat3d{};

    s1 << rnd(gen), rnd(gen), rnd(gen),
          rnd(gen), rnd(gen), rnd(gen),
          rnd(gen), rnd(gen), rnd(gen);
    std::cout << "Matrix S1:\n"
              << s1 << std::endl;

    s2 << rnd(gen), rnd(gen), rnd(gen),
          rnd(gen), rnd(gen), rnd(gen),
          rnd(gen), rnd(gen), rnd(gen);
    std::cout << "Matrix S2:\n"
              << s2 << std::endl;

    s3 << rnd(gen), rnd(gen), rnd(gen),
          rnd(gen), rnd(gen), rnd(gen),
          rnd(gen), rnd(gen), rnd(gen);
    std::cout << "Matrix S3:\n"
              << s3 << std::endl;

    // std::cout << "Enter Matrix S1:\n";
    // std::cin >> s1(0, 0) >> s1(0, 1) >> s1(0, 2)
    //          >> s1(1, 0) >> s1(1, 1) >> s1(1, 2)
    //          >> s1(2, 0) >> s1(2, 1) >> s1(2, 2);
    // std::cout << s1 << std::endl;

    // std::cout << "Enter Matrix S2:\n";
    // std::cin >> s2(0, 0) >> s2(0, 1) >> s2(0, 2)
    //          >> s2(1, 0) >> s2(1, 1) >> s2(1, 2)
    //          >> s2(2, 0) >> s2(2, 1) >> s2(2, 2);
    // std::cout << s2 << std::endl;

    // std::cout << "Enter Matrix S3:\n";
    // std::cin >> s3(0, 0) >> s3(0, 1) >> s3(0, 2)
    //          >> s3(1, 0) >> s3(1, 1) >> s3(1, 2)
    //          >> s3(2, 0) >> s3(2, 1) >> s3(2, 2);
    // std::cout << s3 << std::endl;

    auto t1 = mat3d{};
    auto t2 = mat3d{};
    auto t3 = mat3d{};

    t1 << rnd(gen), rnd(gen), rnd(gen),
          rnd(gen), rnd(gen), rnd(gen),
          rnd(gen), rnd(gen), rnd(gen);
    std::cout << "Matrix T1:\n"
              << t1 << std::endl;

    t2 << rnd(gen), rnd(gen), rnd(gen),
          rnd(gen), rnd(gen), rnd(gen),
          rnd(gen), rnd(gen), rnd(gen);
    std::cout << "Matrix T2:\n"
              << t2 << std::endl;

    t3 << rnd(gen), rnd(gen), rnd(gen),
          rnd(gen), rnd(gen), rnd(gen),
          rnd(gen), rnd(gen), rnd(gen);
    std::cout << "Matrix T3:\n"
              << t3 << std::endl;


    // std::cout << "Enter Matrix T1:\n";
    // std::cin >> t1(0, 0) >> t1(0, 1) >> t1(0, 2)
    //          >> t1(1, 0) >> t1(1, 1) >> t1(1, 2)
    //          >> t1(2, 0) >> t1(2, 1) >> t1(2, 2);
    // std::cout << t1 << std::endl;

    // std::cout << "Enter Matrix T2:\n";
    // std::cin >> t2(0, 0) >> t2(0, 1) >> t2(0, 2)
    //          >> t2(1, 0) >> t2(1, 1) >> t2(1, 2)
    //          >> t2(2, 0) >> t2(2, 1) >> t2(2, 2);
    // std::cout << t2 << std::endl;

    // std::cout << "Enter Matrix T3:\n";
    // std::cin >> t3(0, 0) >> t3(0, 1) >> t3(0, 2)
    //          >> t3(1, 0) >> t3(1, 1) >> t3(1, 2)
    //          >> t3(2, 0) >> t3(2, 1) >> t3(2, 2);
    // std::cout << t3 << std::endl;

#ifdef DRAW_DEBUG
    frame.display(image);
    // frame2.display(image2);
#endif

    auto result = find_parallel_eigenvectors(s1, s2, s3, t1, t2, t3,
                                             spatial_epsilon, direction_epsilon);

    // std::cout << "Number of calls to bezierCoefficients: " << call_count << std::endl;

    std::cout << "Found Parallel Eigenvector points:" << std::endl;

    for(const auto& p: result)
    {
        std::cout << print(p) << std::endl;
    }

#ifdef DRAW_DEBUG
    while(!frame.is_closed())
    {
        frame.wait();
    }
#endif

    // auto call_avg = 0.0;
    // constexpr auto n_runs = 200;

    // for(auto i: range(n_runs))
    // {
    //     call_count = 0;
    //     find_parallel_eigenvectors(mat3d::Random(), mat3d::Random(), mat3d::Random(),
    //                                mat3d::Random(), mat3d::Random(), mat3d::Random(),
    //                                spatial_epsilon, direction_epsilon);
    //     call_avg += double(call_count);
    // }

    // std::cout << "Average calls to bezierCoefficients: "
    //           << (call_avg/n_runs) << std::endl;

    // image.save_png("subdivided.png");
    // frame.display(image);
    // while(!frame.is_closed())
    // {
    //     frame.wait();
    // }

    // auto result = bezierCoefficients(t1, t2, t3, r1, r2, r3);

    // std::cout << "Bezier coefficients for \u03B1:\n";
    // std::cout << "\u03B1300 = " << result[0][i300] << std::endl;
    // std::cout << "\u03B1210 = " << result[0][i210] << std::endl;
    // std::cout << "\u03B1201 = " << result[0][i201] << std::endl;
    // std::cout << "\u03B1030 = " << result[0][i030] << std::endl;
    // std::cout << "\u03B1120 = " << result[0][i120] << std::endl;
    // std::cout << "\u03B1021 = " << result[0][i021] << std::endl;
    // std::cout << "\u03B1003 = " << result[0][i003] << std::endl;
    // std::cout << "\u03B1102 = " << result[0][i102] << std::endl;
    // std::cout << "\u03B1012 = " << result[0][i012] << std::endl;
    // std::cout << "\u03B1111 = " << result[0][i111] << std::endl;

    // std::cout << "Bezier coefficients for \u03B2:\n";
    // std::cout << "\u03B2300 = " << result[1][i300] << std::endl;
    // std::cout << "\u03B2210 = " << result[1][i210] << std::endl;
    // std::cout << "\u03B2201 = " << result[1][i201] << std::endl;
    // std::cout << "\u03B2030 = " << result[1][i030] << std::endl;
    // std::cout << "\u03B2120 = " << result[1][i120] << std::endl;
    // std::cout << "\u03B2021 = " << result[1][i021] << std::endl;
    // std::cout << "\u03B2003 = " << result[1][i003] << std::endl;
    // std::cout << "\u03B2102 = " << result[1][i102] << std::endl;
    // std::cout << "\u03B2012 = " << result[1][i012] << std::endl;
    // std::cout << "\u03B2111 = " << result[1][i111] << std::endl;

    // std::cout << "Bezier coefficients for \u03B3:\n";
    // std::cout << "\u03B3300 = " << result[2][i300] << std::endl;
    // std::cout << "\u03B3210 = " << result[2][i210] << std::endl;
    // std::cout << "\u03B3201 = " << result[2][i201] << std::endl;
    // std::cout << "\u03B3030 = " << result[2][i030] << std::endl;
    // std::cout << "\u03B3120 = " << result[2][i120] << std::endl;
    // std::cout << "\u03B3021 = " << result[2][i021] << std::endl;
    // std::cout << "\u03B3003 = " << result[2][i003] << std::endl;
    // std::cout << "\u03B3102 = " << result[2][i102] << std::endl;
    // std::cout << "\u03B3012 = " << result[2][i012] << std::endl;
    // std::cout << "\u03B3111 = " << result[2][i111] << std::endl;

    // std::cout << "Bezier coefficients for d:\n";
    // std::cout << "d300 = " << result[3][i300] << std::endl;
    // std::cout << "d210 = " << result[3][i210] << std::endl;
    // std::cout << "d201 = " << result[3][i201] << std::endl;
    // std::cout << "d030 = " << result[3][i030] << std::endl;
    // std::cout << "d120 = " << result[3][i120] << std::endl;
    // std::cout << "d021 = " << result[3][i021] << std::endl;
    // std::cout << "d003 = " << result[3][i003] << std::endl;
    // std::cout << "d102 = " << result[3][i102] << std::endl;
    // std::cout << "d012 = " << result[3][i012] << std::endl;
    // std::cout << "d111 = " << result[3][i111] << std::endl;

    return 0;
}

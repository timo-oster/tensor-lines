#include <Eigen/Core>
#include <Eigen/LU>

#include <utility>
#include <algorithm>
#include <numeric>
#include <array>
#include <list>
#include <map>
#include <queue>
#include <iostream>
#include <random>

#include <boost/range/adaptors.hpp>
#include <boost/algorithm/cxx11/any_of.hpp>
#include <boost/algorithm/cxx11/all_of.hpp>
#include <boost/range/algorithm_ext/insert.hpp>
#include <boost/range/algorithm/max_element.hpp>
#include <boost/range/algorithm/min_element.hpp>

#include <CImg.h>

#include "utils.hh"

namespace peigv
{

using vec2d = Eigen::Vector2d;
using mat3d = Eigen::Matrix3d;

using bcoeffs = std::array<double, 10>;

using point_list = std::list<vec3d>;

using CImg = cimg_library::CImg<double>;
using CImgDisplay = cimg_library::CImgDisplay;

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

template<typename T>
class BarycetricInterpolator
{
public:

    using Self = BarycetricInterpolator;

    explicit BarycetricInterpolator(const T& v1 = T{},
                                    const T& v2 = T{},
                                    const T& v3 = T{}):
            _v1{v1}, _v2{v2}, _v3{v3}
    { }

    /**
     * @brief Evaluate the interpolator at a given barycentric coordinate
     *
     * It is assumed that pos.sum() == 1.0
     *
     * @param pos barycentric coordinate with three components
     * @return Value and position on the triangle at the given barycentric coordinates
     */
    T operator()(const vec3d& pos) const
    {
        return pos[0]*_v1 + pos[1]*_v2 + pos[2]*_v3;
    }

    T& v1() { return _v1; }
    const T& v1() const { return _v1; }
    T& v2() { return _v2; }
    const T& v2() const { return _v2; }
    T& v3() { return _v3; }
    const T& v3() const { return _v3; }

    /**
     * @brief Split the interpolator into 4 new ones representing the parts of
     *        a subdivided triangle
     * @return Array of four new interpolators
     */
    std::array<Self, 4> split() const
    {
        auto v12 = T{(_v1+_v2)/2};
        auto v13 = T{(_v1+_v3)/2};
        auto v23 = T{(_v2+_v3)/2};
        return {
            Self{_v1, v12, v13},
            Self{v12, _v2, v23},
            Self{v13, v23, _v3},
            Self{v12, v23, v13}
        };
    }

private:
    T _v1;
    T _v2;
    T _v3;
};

using TensorInterp = BarycetricInterpolator<mat3d>;
using Triangle = BarycetricInterpolator<vec3d>;

struct SubPackage{
    TensorInterp t;
    TensorInterp s;
    Triangle tri;
};

using PackQueue = std::queue<SubPackage>;

std::array<vec2d, 3> project_tri(const Triangle& tri, bool topdown = true)
{
    auto proj = Eigen::Matrix<double, 2, 3>{};
    if(topdown)
    {
        proj << 1, 1, 0,
                1, -1, 0;
    }
    else
    {
        proj << 1, -1, 0,
                -1, -1, 1;
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

void draw_tri(CImg& image, CImgDisplay& frame,
              const Triangle& tri, const double* color,
              bool fill = true, bool topdown=true)
{
    auto projected = project_tri(tri, topdown);
    if(fill)
    {
        image.draw_triangle(projected[0].x(), projected[0].y(),
                            projected[1].x(), projected[1].y(),
                            projected[2].x(), projected[2].y(),
                            color);
    }
    else
    {
        image.draw_triangle(projected[0].x(), projected[0].y(),
                            projected[1].x(), projected[1].y(),
                            projected[2].x(), projected[2].y(),
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

    image.draw_triangle(projected[0].x(), projected[0].y(),
                        projected[1].x(), projected[1].y(),
                        projected[2].x(), projected[2].y(),
                        c1.data(), c2.data(), c3.data());
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

    auto denom_coeffs = bezierCoefficients(A, B, C);

    return {alpha_coeffs, beta_coeffs, gamma_coeffs, denom_coeffs};
}


bool eigen_dirs_queue(const TensorInterp& s,
                      const TensorInterp& t,
                      double epsilon)
{
    auto tque = std::queue<Triangle>{};

    tque.push(Triangle{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
    tque.push(Triangle{{0, 1, 0}, {-1, 0, 0}, {0, 0, 1}});
    tque.push(Triangle{{-1, 0, 0}, {0, -1, 0}, {0, 0, 1}});
    tque.push(Triangle{{0, -1, 0}, {1, 0, 0}, {0, 0, 1}});

    auto found = false;
    // auto vmin = 0.0;
    // auto vmax = 0.0;
    // auto vin = false;

    while(!tque.empty())
    {
        auto tri = tque.front();
        tque.pop();

        auto s_coeffs = bezierCoefficients(
                s.v1(), s.v2(), s.v3(),
                tri.v1(), tri.v2(), tri.v3());

        auto t_coeffs = bezierCoefficients(
                t.v1(), t.v2(), t.v3(),
                tri.v1(), tri.v2(), tri.v3());

        // auto& draw_var = t_coeffs[0];

        // if(!vin)
        // {
        //     auto ma = boost::range::max_element(draw_var);
        //     auto mi = boost::range::min_element(draw_var);
        //     vmax = std::max(std::abs(*ma), std::abs(*mi));
        //     vmin = -vmax;
        //     vin = true;
        // }


        // draw_interp_tri(alpha_image, tri,
        //                 draw_var[i300], draw_var[i030], draw_var[i003],
        //                 vmin*0.01, vmax*0.01);
        // frame3.display(alpha_image);
        // frame2.display(image2);

        auto s_alpha_sign = same_sign(s_coeffs[0]);
        auto s_beta_sign = same_sign(s_coeffs[1]);
        auto s_gamma_sign = same_sign(s_coeffs[2]);
        auto s_denom_sign = same_sign(s_coeffs[3]);

        auto t_alpha_sign = same_sign(t_coeffs[0]);
        auto t_beta_sign = same_sign(t_coeffs[1]);
        auto t_gamma_sign = same_sign(t_coeffs[2]);
        auto t_denom_sign = same_sign(t_coeffs[3]);

        if(true //s_denom_sign != 0 && t_denom_sign != 0
           && std::abs(s_alpha_sign + s_beta_sign + s_gamma_sign) == 3
           && std::abs(t_alpha_sign + t_beta_sign + t_gamma_sign) == 3)
        {
            draw_tri(image2, frame2, tri, green, false);
            // return true;
            found = true;
            continue;
        }

        if(( true //s_denom_sign != 0
                 && std::max({s_alpha_sign, s_beta_sign, s_gamma_sign}) *
                    std::min({s_alpha_sign, s_beta_sign, s_gamma_sign}) < 0)
            ||
                (true // t_denom_sign != 0
                 && std::max({t_alpha_sign, t_beta_sign, t_gamma_sign}) *
                    std::min({t_alpha_sign, t_beta_sign, t_gamma_sign}) < 0)
            )
        {
            draw_tri(image2, frame2, tri, red, false);
            continue;
        }

        // recursion ends when triangle is too small
        if((tri.v1() - tri.v2()).norm() < epsilon)
        {
            // Small triangle and still not sure if parallel eigenvectors
            // possible: assume yes
            draw_tri(image2, frame2, tri, yellow, false);
            found = true;
            continue;
        }

        auto tri_subs = tri.split();
        for(const auto& tri_sub: tri_subs)
        {
            tque.push(tri_sub);
        }
    }

    // return false;
    return found;
}


point_list parallel_eigenvector_search_queue(const TensorInterp& s,
                                             const TensorInterp& t,
                                             const Triangle& tri,
                                             double epsilon)
{
    auto pque = PackQueue{};
    pque.push(SubPackage{s, t, tri});

    auto result_points = point_list{};

    auto count = 1;
    while(!pque.empty())
    {
        frame.display(image);
        auto pack = pque.front();
        pque.pop();

        image2.fill(0);
        auto has_dirs = eigen_dirs_queue(pack.s, pack.t, epsilon);
        // image2.display();
        if(count == 1)
        {
            image2.save("directions_nodenom.png", count, 6);
        }

        // frame2.display(image2);
        ++count;
        if(!has_dirs)
        {
            draw_tri(image, frame, pack.tri, red, true, false);
            continue;
        }

        if((pack.tri.v1() - pack.tri.v2()).norm() < epsilon)
        {
            draw_tri(image, frame, pack.tri, green, true, false);
            result_points.push_back(tri({1.0/3.0, 1.0/3.0, 1.0/3.0}));
            continue;
        }

        auto s_subs = pack.s.split();
        auto t_subs = pack.t.split();
        auto tri_subs = pack.tri.split();

        for(auto i: range(s_subs.size()))
        {
            draw_tri(image, frame, tri_subs[i], blue, false, false);
            pque.push({s_subs[i], t_subs[i], tri_subs[i]});
        }
    }

    return result_points;
}


point_list find_parallel_eigenvectors(
        const mat3d& s1, const mat3d& s2, const mat3d& s3,
        const mat3d& t1, const mat3d& t2, const mat3d& t3,
        double epsilon)
{
    auto start_tri = Triangle{
        vec3d{1, 0, 0},
        vec3d{0, 1, 0},
        vec3d{0, 0, 1}
    };

    return parallel_eigenvector_search_queue(TensorInterp{s1, s2, s3},
                                             TensorInterp{t1, t2, t3},
                                             start_tri, epsilon);
}

} // namespace peigv

int main(int argc, char const *argv[])
{
    using namespace peigv;
    auto gen = std::mt19937{7};
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

    // frame.display(image);
    // frame2.display(image2);

    find_parallel_eigenvectors(s1, s2, s3, t1, t2, t3, 0.01);

    image.save_png("subdivided.png");

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

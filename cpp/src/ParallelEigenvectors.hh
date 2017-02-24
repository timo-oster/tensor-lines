#ifndef CPP_PARALLEL_EIGENVECTORS_HH
#define CPP_PARALLEL_EIGENVECTORS_HH

#include "utils.hh"

#include <list>

#ifdef DRAW_DEBUG
#include <CImg.h>
#endif

namespace peigv
{

#ifdef DRAW_DEBUG
using CImg = cimg_library::CImg<double>;
using CImgDisplay = cimg_library::CImgDisplay;

extern CImg image;
// extern CImg image2;
extern CImgDisplay frame;
// extern CImgDisplay frame2;
#endif

using point_list = std::list<vec3d>;

point_list findParallelEigenvectors(
        const mat3d& s1, const mat3d& s2, const mat3d& s3,
        const mat3d& t1, const mat3d& t2, const mat3d& t3,
        double spatial_epsilon, double direction_epsilon);

point_list findParallelEigenvectors(
        const mat3d& s1, const mat3d& s2, const mat3d& s3,
        const mat3d& t1, const mat3d& t2, const mat3d& t3,
        const vec3d& p1, const vec3d& p2, const vec3d& p3,
        double spatial_epsilon, double direction_epsilon);

} // namespace peigv

#endif

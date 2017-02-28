#ifndef CPP_PARALLEL_EIGENVECTORS_HH
#define CPP_PARALLEL_EIGENVECTORS_HH

#include "utils.hh"

#include <list>

#ifdef DRAW_DEBUG
#include <CImg.h>
#endif

namespace peigv
{

/**
 * Rank of an eigenvector (when ordering by absolute eigenvalue)
 */
enum class ERank: int
{
    First = 0,
    Second = 1,
    Third = 2
};

/**
 * Parallel eigenvector point.
 */
struct peigvPoint
{
    vec3d pos; ///< position
    ERank s_rank; ///< rank of eigenvector of tensor field S
    ERank t_rank; ///< rank of eigenvector of tensor field T
    vec3d eivec; ///< Eigenvector direction
    double s_eival; ///< Eigenvalue for tensor field S
    double t_eival; ///< Eigenvalue for tensor field T
};

using point_list = std::list<peigvPoint>;

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

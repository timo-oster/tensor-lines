#ifndef CPP_PARALLEL_EIGENVECTORS_HH
#define CPP_PARALLEL_EIGENVECTORS_HH

#include "utils.hh"

#include <list>


namespace peigv
{

/**
 * Rank/order of an eigenvalue of a 3x3 matrix
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
struct PeigvPoint
{
    Vec3d pos; ///< position
    ERank s_rank; ///< rank of eigenvector of tensor field S
    ERank t_rank; ///< rank of eigenvector of tensor field T
    Vec3d eivec; ///< Eigenvector direction
    double s_eival; ///< Eigenvalue for tensor field S
    double t_eival; ///< Eigenvalue for tensor field T
};

using PointList = std::list<PeigvPoint>;

/**
 * @brief Find parallel eigenvector points on a triangle given two tensors at
 *     each corner
 *
 * @param s1 Value of first tensor field at first triangle corner
 * @param s2 Value of first tensor field at second triangle corner
 * @param s3 Value of first tensor field at third triangle corner
 * @param t1 Value of second tensor field at first triangle corner
 * @param t2 Value of second tensor field at second triangle corner
 * @param t3 Value of second tensor field at third triangle corner
 * @param p1 Position of first triangle corner
 * @param p2 Position of second triangle corner
 * @param p3 Position of third triangle corner
 * @param spatial_epsilon Lower limit for subdivision in space
 * @param direction_epsilon Lower limit for subdivision in eigenvector direction
 *     space
 * @param cluster_epsilon Minimum distance of two solutions to be considered
 *     separate from each other
 * @param parallelity_epsilon Maximum angle error for a solution to be
 *     considered a parallel eigenvector point
 * @return A list of found parallel eigenvector points on the triangle
 */
PointList findParallelEigenvectors(
        const Mat3d& s1, const Mat3d& s2, const Mat3d& s3,
        const Mat3d& t1, const Mat3d& t2, const Mat3d& t3,
        double spatial_epsilon, double direction_epsilon,
        double cluster_epsilon, double parallelity_epsilon);

/**
 * @brief Find parallel eigenvector points in barycentric coordinates for a
 *     general triangle given two tensors at each corner
 *
 * @param s1 Value of first tensor field at first triangle corner
 * @param s2 Value of first tensor field at second triangle corner
 * @param s3 Value of first tensor field at third triangle corner
 * @param t1 Value of second tensor field at first triangle corner
 * @param t2 Value of second tensor field at second triangle corner
 * @param t3 Value of second tensor field at third triangle corner
 * @param spatial_epsilon Lower limit for subdivision in space
 * @param direction_epsilon Lower limit for subdivision in eigenvector direction
 *     space
 * @param cluster_epsilon Minimum distance of two solutions to be considered
 *     separate from each other
 * @param parallelity_epsilon Maximum angle error for a solution to be
 *     considered a parallel eigenvector point
 * @return A list of found parallel eigenvector points on the triangle in
 *     barycentric coordinates
 */
PointList findParallelEigenvectors(
        const Mat3d& s1, const Mat3d& s2, const Mat3d& s3,
        const Mat3d& t1, const Mat3d& t2, const Mat3d& t3,
        const Vec3d& p1, const Vec3d& p2, const Vec3d& p3,
        double spatial_epsilon, double direction_epsilon,
        double cluster_epsilon, double parallelity_epsilon);

} // namespace peigv

#endif

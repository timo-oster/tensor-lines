#ifndef CPP_PARALLEL_EIGENVECTORS_HH
#define CPP_PARALLEL_EIGENVECTORS_HH

#include "ParallelEigenvectorDefinitions.hh"

#include <vector>
#include <array>

namespace pev
{

/**
 * List of found parallel eigenvector points
 */
using PointList = std::vector<PEVPoint>;


/**
 * @brief Options for the PEV point search
 */
struct PEVOptions
{
    double spatial_epsilon = 1e-6;
    double direction_epsilon = 1e-6;
    double cluster_epsilon = 5e-6;
};


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
 * @param x1 Position of first triangle corner
 * @param x2 Position of second triangle corner
 * @param x3 Position of third triangle corner
 * @param spatial_epsilon Lower limit for subdivision in space
 * @param direction_epsilon Lower limit for subdivision in eigenvector direction
 *     space
 * @param cluster_epsilon Minimum distance of two solutions to be considered
 *     separate from each other
 * @param parallelity_epsilon Maximum angle error for a solution to be
 *     considered a parallel eigenvector point
 * @return The number of false positives and a list of found parallel
 *     eigenvector points on the triangle
 */
PointList findParallelEigenvectors(const std::array<Mat3d, 3>& s,
                                   const std::array<Mat3d, 3>& t,
                                   const std::array<Vec3d, 3>& x,
                                   const PEVOptions& opts = PEVOptions{});

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
 * @return The number of false positives and a list of found parallel
 *     eigenvector points on the triangle in barycentric coordinates
 */
PointList findParallelEigenvectors(const std::array<Mat3d, 3>& s,
                                   const std::array<Mat3d, 3>& t,
                                   const PEVOptions& opts = PEVOptions{});


PointList findTensorSujudiHaimes(const std::array<Mat3d, 3>& t,
                                 const std::array<std::array<Mat3d, 3>, 3>& dt,
                                 const std::array<Vec3d, 3>& x,
                                 const PEVOptions& opts = PEVOptions{});

PointList findTensorSujudiHaimes(const std::array<Mat3d, 3>& t,
                                 const std::array<std::array<Mat3d, 3>, 3>& dt,
                                 const PEVOptions& opts = PEVOptions{});

} // namespace pev

#endif

#ifndef CPP_TENSOR_LINES_HH
#define CPP_TENSOR_LINES_HH

#include "TensorLineDefinitions.hh"

#include <vector>
#include <array>

namespace tl
{

/**
 * List of found parallel eigenvector points
 */
using PointList = std::vector<TLPoint>;

struct TLResult
{
    // List of parallel eigenvector points
    PointList points;
    // List of (approximate) eigenvector directions for which
    // a planar or volume structure might exist
    std::vector<Vec3d> non_line_dirs;
};


/**
 * @brief Options for the tensor line point search
 */
struct TLOptions
{
    double tolerance = 1e-6;
    double cluster_epsilon = 5e-6;
    std::size_t max_candidates = 100;
};


/**
 * Find intersections of parallel eigenvector lines with a triangle defining
 * two linear tensor fields.
 *
 * @param s The first tensor field (given by the tensors at the triangle corners)
 * @param t The second tensor field (given by the tensors at the triangle corners)
 * @param x The triangle on which they are defined (given by the three corners)
 * @param opts Options for the search algorithm
 * @return The result of the search with points in cartesian coordinates in the
 *      original 3D space
 */
TLResult findParallelEigenvectors(const std::array<Mat3d, 3>& s,
                                  const std::array<Mat3d, 3>& t,
                                  const std::array<Vec3d, 3>& x,
                                  const TLOptions& opts = TLOptions{});

/**
 * @brief Find intersections of parallel eigenvector lines with a standard triangle
 *      defining two linear tensor fields.
 *
 *
 * @param s The first tensor field (given by the tensors at the triangle corners)
 * @param t The second tensor field (given by the tensors at the triangle corners)
 * @param opts Options for the search algorithm
 * @return The result of the search with points in barycentric coordinates of
 *     the triangle
 */
TLResult findParallelEigenvectors(const std::array<Mat3d, 3>& s,
                                  const std::array<Mat3d, 3>& t,
                                  const TLOptions& opts = TLOptions{});


/**
 * Find intersections of tensor core lines with a triangle defining a linear
 * tensor field
 *
 * @param t The tensor field (given by the tensors at the triangle corners)
 * @param dt The derivatives of the tensor field (in x, y, and z direction,
 *     constant on the triangle)
 * @param x The triangle on which the tensor field is defined (given by the
 *     three corners)
 * @param opts Options for the search algorithm
 * @return The result of the search with points in cartesian coordinates in the
 *      original 3D space
 */
TLResult findTensorCoreLines(const std::array<Mat3d, 3>& t,
                             const std::array<Mat3d, 3>& dt,
                             const std::array<Vec3d, 3>& x,
                             const TLOptions& opts = TLOptions{});

/**
 * Find intersections of tensor core lines with a triangle defining a linear
 * tensor field
 *
 * @param t The tensor field (given by the tensors at the triangle corners)
 * @param dt The derivatives of the tensor field (in x, y, and z direction,
 *     constant on the triangle)
 * @param opts Options for the search algorithm
 * @return The result of the search with points in barycentric coordinates of
 *     the triangle
 */
TLResult findTensorCoreLines(const std::array<Mat3d, 3>& t,
                             const std::array<Mat3d, 3>& dt,
                             const TLOptions& opts = TLOptions{});


/**
 * Find intersections of degenerate lines (where two eigenvalues are equal) with
 * a triangle defining a linear tensor field
 *
 * @param t The tensor field (given by the tensors at the triangle corners)
 * @param x The triangle on which the tensor field is defined (given by the
 *     three corners)
 * @param opts Options for the search algorithm
 * @return The result of the search with points in cartesian coordinates in the
 *      original 3D space
 */
TLResult findTensorTopology(const std::array<Mat3d, 3>& t,
                            const std::array<Vec3d, 3>& x,
                            const TLOptions& opts = TLOptions{});

/**
 * Find intersections of degenerate lines (where two eigenvalues are equal) with
 * a triangle defining a linear tensor field
 *
 * @param t The tensor field (given by the tensors at the triangle corners)
 * @param x The triangle on which the tensor field is defined (given by the
 *     three corners)
 * @param opts Options for the search algorithm
 * @return The result of the search with points in barycentric coordinates of
 *     the triangle
 */
TLResult findTensorTopology(const std::array<Mat3d, 3>& t,
                            const TLOptions& opts = TLOptions{});

} // namespace tl

#endif

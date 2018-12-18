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


TLResult findParallelEigenvectors(const std::array<Mat3d, 3>& s,
                                   const std::array<Mat3d, 3>& t,
                                   const std::array<Vec3d, 3>& x,
                                   const TLOptions& opts = TLOptions{});

TLResult findParallelEigenvectors(const std::array<Mat3d, 3>& s,
                                   const std::array<Mat3d, 3>& t,
                                   const TLOptions& opts = TLOptions{});


TLResult findTensorCoreLines(const std::array<Mat3d, 3>& t,
                              const std::array<Mat3d, 3>& dt,
                              const std::array<Vec3d, 3>& x,
                              const TLOptions& opts = TLOptions{});

TLResult findTensorCoreLines(const std::array<Mat3d, 3>& t,
                              const std::array<Mat3d, 3>& dt,
                              const TLOptions& opts = TLOptions{});

TLResult findTensorTopology(const std::array<Mat3d, 3>& t,
                             const std::array<Vec3d, 3>& x,
                             const TLOptions& opts = TLOptions{});

TLResult findTensorTopology(const std::array<Mat3d, 3>& t,
                             const TLOptions& opts = TLOptions{});

} // namespace tl

#endif

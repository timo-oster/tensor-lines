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
    double tolerance = 1e-6;
    double min_ev = 1e-6;
    double cluster_epsilon = 5e-6;
};


PointList findParallelEigenvectors(const std::array<Mat3d, 3>& s,
                                   const std::array<Mat3d, 3>& t,
                                   const std::array<Vec3d, 3>& x,
                                   const PEVOptions& opts = PEVOptions{});

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

#ifndef CPP_PARALLEL_EIGENVECTOR_DEFINITIONS_HH
#define CPP_PARALLEL_EIGENVECTOR_DEFINITIONS_HH

#include <Eigen/Core>

namespace pev
{

using Vec3d = Eigen::Vector3d;
using Mat3d = Eigen::Matrix3d;


/**
 * Rank/order of an eigenvalue of a 3x3 matrix
 */
enum class ERank : int
{
    First = 0,
    Second = 1,
    Third = 2
};


/**
 * Parallel eigenvector point.
 */
struct PEVPoint
{
    Vec3d pos; ///< position
    ERank s_rank; ///< rank of eigenvector of tensor field S
    ERank t_rank; ///< rank of eigenvector of tensor field T
    Vec3d eivec; ///< Eigenvector direction
    double s_eival; ///< Eigenvalue for tensor field S
    double t_eival; ///< Eigenvalue for tensor field T
    bool s_has_imaginary; ///< S has any imaginary eigenvalues at the position
    bool t_has_imaginary; ///< T has any imaginary eigenvalues at the position
    std::size_t cluster_size; ///< Number of candidate points that contributed
    double pos_uncertainty; ///< Size of the last subdivision cell in
                            /// position space
    double dir_uncertainty; ///< Size of the last subdivision cell in
                            /// direction space
    double line_stability; ///< Measure of numeric stability of the solution
};

} // namespace pev

#endif

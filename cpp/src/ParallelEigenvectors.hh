#ifndef CPP_PARALLEL_EIGENVECTORS_HH
#define CPP_PARALLEL_EIGENVECTORS_HH

#include "TensorProductBezierTriangle.hh"
#include "utils.hh"

#ifdef DRAW_DEBUG
#include <CImg.h>
#ifdef Success
#undef Success
#endif
#endif

#include <vector>


namespace pev
{
#ifdef DRAW_DEBUG
using CImg = cimg_library::CImg<double>;
using CImgDisplay = cimg_library::CImgDisplay;

extern CImg pos_image;
extern CImg dir_image;
extern CImgDisplay pos_frame;
extern CImgDisplay dir_frame;
#endif


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
    bool s_has_imaginary; //< S has any imaginary eigenvalues at the position
    bool t_has_imaginary; //< T has any imaginary eigenvalues at the position
    std::size_t cluster_size; //< Number of candidate points that contributed
};

/**
 * List of found parallel eigenvector points
 */
using PointList = std::vector<PEVPoint>;


/**
 * Linear tensor field expressed in barycentric coordinates
 */
using TensorInterp = TensorProductBezierTriangle<Mat3d, double, 1>;


/**
 * Triangle in 3d expressed in barycentric coordinates
 */
using Triangle = TensorProductBezierTriangle<Vec3d, double, 1>;


/**
 * @brief Options for the PEV point search
 */
struct PEVOptions
{
    PEVOptions(double s = 1e-6, double d = 1e-6, double c = 5e-6)
            : spatial_epsilon(s), direction_epsilon(d), cluster_epsilon(c)
    {
    }

    double spatial_epsilon;
    double direction_epsilon;
    double cluster_epsilon;
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
PointList findParallelEigenvectors(const TensorInterp& s,
                                   const TensorInterp& t,
                                   const Triangle& x,
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
PointList findParallelEigenvectors(const TensorInterp& s,
                                   const TensorInterp& t,
                                   const PEVOptions& opts = PEVOptions{});

} // namespace pev

#endif

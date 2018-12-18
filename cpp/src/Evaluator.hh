#ifndef CPP_EVALUATOR_HH
#define CPP_EVALUATOR_HH

#include "TensorLineDefinitions.hh"
#include "TensorProductBezierTriangles.hh"

#include <iterator>
#include <type_traits>
#include <utility>

namespace tl
{

using Triangle = TensorProductBezierTriangle<Vec3d, double, 1>;


/// Pair of triangles in position and direction space
struct DoubleTri
{
    Triangle pos_tri = Triangle{};
    Triangle dir_tri = Triangle{};

    /**
     * @brief Split one of the triangles into a smaller one.
     * @details Depending on @a D, either the triangle in position (0) or
     *      direction (1) space is split. The result is a new DoubleTri with
     *      the selected @a part (0-3) of the split triangle.
     *
     * @param part The part of the split triangle to return (0-3)
     * @tparam D Which triangle to split? 0: position, 1: direction
     * @return New DoubleTri with the part of the split triangle
     */
    template <std::size_t D>
    DoubleTri split(std::size_t part) const
    {
        static_assert(D < 2, "D must be smaller than 2");
        if constexpr(D == 0)
            return {pos_tri.split(part), dir_tri};
        else if constexpr(D == 1)
            return {pos_tri, dir_tri.split(part)};
    }

    friend bool operator==(const DoubleTri& d1, const DoubleTri& d2)
    {
        return d1.pos_tri == d2.pos_tri && d1.dir_tri == d2.dir_tri;
    }

    friend bool operator!=(const DoubleTri& d1, const DoubleTri& d2)
    {
        return !(d1 == d2);
    }
};


/// Compute the distance between the centers of two triangles for clustering
/// purposes.
inline double distance(const Triangle& t1, const Triangle& t2)
{
    return (t1({1./3, 1./3, 1./3}) - t2({1./3, 1./3, 1./3}))
            .norm();
}


/// Compute the distance between two triangle pairs as the maximum of the
/// distances in position and direction space.
inline double distance(const DoubleTri& t1, const DoubleTri& t2)
{
    return std::max(distance(t1.pos_tri, t2.pos_tri),
                    distance(t1.dir_tri, t2.dir_tri));
}


/// Result of evaluating an evaluator by calling the @c eval() function.
enum class Result
{
    Accept,
    Discard,
    Split
};


/// Concept check for the @c tris() function of an evaluator
template <typename E>
constexpr bool has_tris =
        std::is_convertible<decltype(std::declval<E>().tris()),
                            DoubleTri>::value;

/// Concept check for the @c split() function of an evaluator
template <typename E>
constexpr bool is_splittable =
        std::is_convertible<decltype(*std::begin(std::declval<E>().split())),
                            E>::value;


/// Concept check for the @c splitLevel() function of an evaluator
template <typename E>
constexpr bool has_splitlevel =
        std::is_convertible<decltype(std::declval<E>().splitLevel()),
                            std::size_t>::value;


/// Concept check for the @c eval() function of an evaluator
template <typename E>
constexpr bool is_evaluatable =
        std::is_convertible<decltype(std::declval<E>().eval()), Result>::value;


/// Concept check for the @c distance() function between evaluators
template <typename E>
constexpr bool has_distance =
        std::is_convertible<decltype(distance(std::declval<E>(),
                                              std::declval<E>())),
                            double>::value;

template <typename, typename = void>
struct is_evaluator : std::false_type {};

template <typename E>
struct is_evaluator<E,
                    std::enable_if_t<has_tris<E>
                                     && is_splittable<E>
                                     && has_splitlevel<E>
                                     && is_evaluatable<E>
                                     && has_distance<E>
                                     && cpp_utils::is_equality_comparable<E>>>
        : std::true_type
{
};

/// Concept check for an evaluator
template<typename E>
inline constexpr bool is_evaluator_v = is_evaluator<E>::value;

} // namespace tl

#endif

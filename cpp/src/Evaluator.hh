#ifndef CPP_EVALUATOR_HH
#define CPP_EVALUATOR_HH

#include "ParallelEigenvectorDefinitions.hh"
#include "TensorProductBezierTriangle.hh"

#include <iterator>
#include <type_traits>

namespace pev
{

using Triangle = TensorProductBezierTriangle<Vec3d, double, 1>;


inline double distance(const Triangle& t1, const Triangle& t2)
{
    return (t1({1./3, 1./3, 1./3}) - t2({1./3, 1./3, 1./3}))
            .norm();
}


struct DoubleTri
{
    Triangle pos_tri = Triangle{};
    Triangle dir_tri = Triangle{};

    template <std::size_t D>
    DoubleTri split(std::size_t part) const
    {
        static_assert(D < 2, "D must be smaller than 2");
        if(D == 0)
            return {pos_tri.split(part), dir_tri};
        else if(D == 1)
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


inline double distance(const DoubleTri& t1, const DoubleTri& t2)
{
    return std::max(distance(t1.pos_tri, t2.pos_tri),
                    distance(t1.dir_tri, t2.dir_tri));
}


enum class Result
{
    Accept,
    Discard,
    Split
};


template <typename E>
constexpr bool has_tris =
        std::is_convertible<decltype(std::declval<E>().tris()),
                            DoubleTri>::value;

template <typename E>
constexpr bool is_splittable =
        std::is_convertible<decltype(*std::begin(std::declval<E>().split())),
                            E>::value;

template <typename E>
constexpr bool has_splitlevel =
        std::is_convertible<decltype(std::declval<E>().splitLevel()),
                            std::size_t>::value;

template <typename E>
constexpr bool is_evaluatable =
        std::is_convertible<decltype(std::declval<E>().eval()), Result>::value;

template <typename E>
constexpr bool has_condition =
        std::is_convertible<decltype(std::declval<E>().condition()),
                            double>::value;

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
                                     && has_condition<E>
                                     && has_distance<E>
                                     && is_equality_comparable<E>>>
        : std::true_type
{
};

} // namespace pev

#endif

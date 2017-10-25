#ifndef CPP_UTILS_HH
#define CPP_UTILS_HH

#include <Eigen/Core>
#include <boost/range/irange.hpp>

#include <array>
#include <cstddef>
#include <ostream>
#include <sstream>
#include <type_traits>

#define BOOST_RESULT_OF_USE_DECLTYPE

namespace pev
{
using Vec3d = Eigen::Vector3d;
using Mat3d = Eigen::Matrix3d;


struct MinNotZero
{
    template <typename T>
    const T& operator()(const T& x, const T& y) const
    {
        if(x == T{0}) return y;
        if(y == T{0}) return x;
        return std::min(x, y);
    }
};

struct MinAbs
{
    template <typename T>
    T operator()(const T& x, const T& y) const
    {
        if((x*y) < T{0}) return T{0};
        return std::min(std::abs(x), std::abs(y));
    }
};

struct MaxAbs
{
    template <typename T>
    T operator()(const T& x, const T& y) const
    {
        return std::max(std::abs(x), std::abs(y));
    }
};


/**
 * @brief An adaptor class to negate a unary predicate functor
 * @details We could use std::not1 instead, but it requires the Predicate to
 *          have a member typedef @c argument_type. This is unnecessarily
 *          restrictive as it prevents the predicate from overloading its
 *          operator() for multiple types.
 *
 * @tparam Predicate A predicate functor returning a type with boolean semantics
 */
template <class Predicate>
struct negator
{
    Predicate pred;

    template <typename T>
    decltype(auto) operator()(const T& arg) const
    {
        return !pred(arg);
    }
};


template <class Predicate>
negator<Predicate> negate(Predicate predicate)
{
    return negator<Predicate>{predicate};
}


template <typename T>
int sgn(T val)
{
    return (T{0} < val) - (val < T{0});
}


template <class T>
inline typename std::make_signed_t<T> as_signed(T t)
{
    return typename std::make_signed_t<T>(t);
}


template <class T>
inline typename std::make_unsigned_t<T> as_unsigned(T t)
{
    return typename std::make_unsigned_t<T>(t);
}


template <typename T, typename U, typename V = int>
inline auto range(T start, U end, V step = 1)
{
    using D = typename std::decay_t<decltype(true ? start : end)>;
    return boost::irange<D>(start, end, step);
}


template <typename T>
inline auto range(T end)
{
    return boost::irange(T{0}, end);
}


/// erase_if for (unordered) containers
template <typename Container,
          typename Predicate>
std::size_t erase_if(Container& container, Predicate pred)
{
    for(auto it = std::begin(container); it != std::end(container);)
    {
        if(pred(*it))
        {
            it = container.erase(it);
        }
        else
        {
            ++it;
        }
    }
}


/// pretty print an Eigen::Vector3* without line breaks
template <typename T,
          typename = typename std::enable_if_t<T::SizeAtCompileTime == 3>>
std::string print(const T& in)
{
    std::stringstream out;
    out << "(" << in[0] << ", " << in[1] << ", " << in[2] << ")";
    return out.str();
}


struct make_string
{
    std::stringstream ss;
    template <typename T>
    make_string& operator<<(const T& data)
    {
        ss << data;
        return *this;
    }
    operator std::string()
    {
        return ss.str();
    }
};
}

#endif // CPP_UTILS_HH

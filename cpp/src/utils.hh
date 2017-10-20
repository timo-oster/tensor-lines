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
    auto operator()(const T& arg) const -> decltype(!pred(arg))
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
inline typename std::make_signed<T>::type as_signed(T t)
{
    return typename std::make_signed<T>::type(t);
}


template <class T>
inline typename std::make_unsigned<T>::type as_unsigned(T t)
{
    return typename std::make_unsigned<T>::type(t);
}


template <typename T, typename U, typename V = int>
inline auto range(T start, U end, V step = 1)
        -> decltype(
            boost::irange<
                typename std::decay<
                    decltype(true ? std::declval<T>() : std::declval<U>())
                >::type
            >(
                std::declval<T>(),
                std::declval<U>(),
                std::declval<V>()))
{
    using D = typename std::decay<decltype(true ? start : end)>::type;
    return boost::irange<D>(start, end, step);
}


template <typename T>
inline auto range(T end)
        -> decltype(boost::irange(std::declval<T>(), std::declval<T>()))
{
    return boost::irange(T{0}, end);
}


/// erase_if for associative (unordered) containers
template <typename AssocContainer,
          typename Predicate,
          typename std::enable_if<
              std::is_same<
                  typename std::decay<
                      typename AssocContainer::value_type::first_type
                  >::type,
                  typename AssocContainer::key_type
              >::value,
              int
          >::type = 0>
std::size_t erase_if(AssocContainer& container, Predicate pred)
{
    auto to_erase = std::vector<typename AssocContainer::key_type>{};
    for(const auto& e : container)
    {
        if(pred(e)) to_erase.push_back(e.first);
    }
    auto erased = std::size_t{0};
    for(const auto& k : to_erase)
    {
        erased += container.erase(k);
    }
    return erased;
}


/// erase_if for (unordered) sets
template <typename Container,
          typename Predicate,
          typename std::enable_if<
              std::is_same<
                  typename Container::value_type,
                  typename Container::key_type
              >::value,
              int
          >::type = 0>
std::size_t erase_if(Container& container, Predicate pred)
{
    auto to_erase = std::vector<typename Container::key_type>{};
    for(const auto& e : container)
    {
        if(pred(e)) to_erase.push_back(e);
    }
    auto erased = std::size_t{0};
    for(const auto& e : to_erase)
    {
        erased = container.erase(e);
    }
    return erased;
}


/// pretty print an Eigen::Vector3* without line breaks
template <typename T,
          typename = typename std::enable_if<T::SizeAtCompileTime == 3>::type>
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

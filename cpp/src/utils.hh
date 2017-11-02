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

struct Max
{
    template <typename T>
    T operator()(const T& x, const T& y) const
    {
        return std::max(x, y);
    }
};

struct Min
{
    template <typename T>
    T operator()(const T& x, const T& y) const
    {
        return std::min(x, y);
    }
};


template <typename T, typename = std::enable_if_t<std::is_signed<T>::value>>
int sgn(T val)
{
    return (T{0} < val) - (val < T{0});
}


struct SameSign
{
    template <typename T1,
              typename T2,
              typename = std::enable_if_t<std::is_signed<T1>::value
                                          && std::is_signed<T2>::value>>
    int operator()(const T1& x, const T2& y) const
    {
        return x * y > 0 ? sgn(x) : 0;
    }
};

/**
 * Check if all numbers of a sequence are positive or negative
 *
 * @return 1 for all positive, -1 for all negative, 0 otherwise
 */
template <typename Sequence>
int sameSign(const Sequence& numbers)
{
    auto nelem = std::distance(std::begin(numbers), std::end(numbers));
    if(nelem == 0) return 0;
    auto first = *std::begin(numbers);
    if(nelem == 1) return sgn(first);
    return std::accumulate(std::next(std::begin(numbers)),
                           std::end(numbers),
                           sgn(first),
                           SameSign{});
}

template<typename... Ts> struct make_void { typedef void type;};
template<typename... Ts> using void_t = typename make_void<Ts...>::type;

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

    template <typename... Args>
    decltype(auto) operator()(Args&&... args) const
    {
        return !pred(std::forward<Args>(args)...);
    }
};


template <class Predicate>
negator<Predicate> negate(Predicate predicate)
{
    return negator<Predicate>{predicate};
}

template <class T>
inline std::make_signed_t<T> as_signed(T t)
{
    return std::make_signed_t<T>(t);
}


template <class T>
inline std::make_unsigned_t<T> as_unsigned(T t)
{
    return std::make_unsigned_t<T>(t);
}


template <typename T>
constexpr bool is_equality_comparable =
        std::is_convertible<decltype(std::declval<T>() == std::declval<T>()),
                            bool>::value &&
                std::is_convertible<decltype(std::declval<T>()
                                             != std::declval<T>()),
                                    bool>::value;

template <typename T>
constexpr bool is_comparable =
        std::is_convertible<decltype(std::declval<T>() < std::declval<T>()),
                            bool>::value &&
                std::is_convertible<decltype(std::declval<T>()
                                             > std::declval<T>()),
                                    bool>::value;


template <typename T, typename U, typename V = int>
inline auto range(T start, U end, V step = 1)
{
    using D = std::decay_t<decltype(true ? start : end)>;
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
void erase_if(Container& container, Predicate pred)
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
} // namespace pev

#endif // CPP_UTILS_HH

#ifndef PEV_UTILS_HH
#define PEV_UTILS_HH

#include <cpp_utils/cpp_utils.h>

#include <Eigen/Core>

#include <array>
#include <cstddef>
#include <ostream>
#include <sstream>
#include <iterator>
#include <type_traits>
#include <utility>
#include <numeric>

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


struct SameSign
{
    template <typename T1,
              typename T2,
              typename = std::enable_if_t<std::is_signed<T1>::value
                                          && std::is_signed<T2>::value>>
    int operator()(const T1& x, const T2& y) const
    {
        using cpp_utils::sgn;
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
    using cpp_utils::sgn;
    auto nelem = std::distance(std::begin(numbers), std::end(numbers));
    if(nelem == 0) return 0;
    auto first = *std::begin(numbers);
    if(nelem == 1) return sgn(first);
    using std::begin;
    using std::end;
    return std::accumulate(std::next(begin(numbers)),
                           end(numbers),
                           sgn(first),
                           SameSign{});
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

} // namespace pev

#endif // PEV_UTILS_HH

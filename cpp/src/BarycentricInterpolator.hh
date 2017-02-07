#ifndef CPP_BARYCENTRIC_INTERPOLATOR_HH
#define CPP_BARYCENTRIC_INTERPOLATOR_HH

#include <array>

#include <utils.hh>

namespace peigv
{

/**
 * Class representing the linear interpolation between three values using
 * barycentric coordinates.
 *
 * The type T is assumed to be an arithmetic type supporting addition and
 * multiplication with a double.
 */
template<typename T>
class BarycetricInterpolator
{
public:

    using Self = BarycetricInterpolator;

    explicit BarycetricInterpolator(const T& v1 = T{},
                                    const T& v2 = T{},
                                    const T& v3 = T{}):
            _v1{v1}, _v2{v2}, _v3{v3}
    { }

    /**
     * @brief Evaluate the interpolator at a given barycentric coordinate
     *
     * It is assumed that a + b + c == 1.0
     *
     * @param a, b, c barycentric coordinates
     * @return Value at the given barycentric coordinates
     */
    T operator()(double a, double b, double c) const
    {
        return a*_v1 + b*_v2 + c*_v3;
    }

    T& v1() { return _v1; }
    const T& v1() const { return _v1; }
    T& v2() { return _v2; }
    const T& v2() const { return _v2; }
    T& v3() { return _v3; }
    const T& v3() const { return _v3; }

    /**
     * @brief Split the interpolator into 4 new ones representing the parts of
     *        a subdivided triangle
     * @return Array of four new interpolators
     */
    std::array<Self, 4> split() const
    {
        auto v12 = T{(_v1+_v2)/2};
        auto v13 = T{(_v1+_v3)/2};
        auto v23 = T{(_v2+_v3)/2};
        return {
            Self{_v1, v12, v13},
            Self{v12, _v2, v23},
            Self{v13, v23, _v3},
            Self{v12, v23, v13}
        };
    }

private:
    T _v1;
    T _v2;
    T _v3;
};

template<typename T>
bool operator==(const BarycetricInterpolator<T>& b1,
                const BarycetricInterpolator<T>& b2)
{
    return b1.v1() == b2.v1() && b1.v2() == b2.v2() && b1.v3() == b2.v3();
}

}

#endif
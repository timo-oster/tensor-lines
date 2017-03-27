#ifndef CPP_BEZIER_TRIANGLE_HH
#define CPP_BEZIER_TRIANGLE_HH

#include "utils.hh"

#include <Eigen/Core>

namespace pev
{

template<typename T, typename C=double>
class BezierTriangle
{
public:
    using Self = BezierTriangle;
    using Coeffs = Eigen::Matrix<T, 10, 1>;
    using Coords = Eigen::Matrix<C, 3, 1>;

    enum Indices: int
    {
        i300 = 0,
        i030,
        i003,
        i210,
        i120,
        i021,
        i012,
        i201,
        i102,
        i111
    };

    explicit BezierTriangle(const Coeffs& coefficients):
            _coeffs(coefficients)
    {}

    T operator()(const C& u, const C& v, const C& w) const
    {
        return *this({u, v, w});
    }

    T operator()(const C& u, const C& v) const
    {
        return *this({u, v, 1.-u-v});
    }

    T operator()(const Coords& pos) const
    {
        return makeBasis(pos).dot(_coeffs);
    }

    C& operator[](int i)
    {
        return _coeffs[i];
    }

    const C& operator[](int i) const
    {
        return _coeffs[i];
    }

    Coeffs& coefficients()
    {
        return _coeffs;
    }

    const Coeffs& coefficients() const
    {
        return _coeffs;
    }

    std::array<Self, 4> split() const
    {
        return {Self{m0()*_coeffs}, Self{m1()*_coeffs},
                Self{m2()*_coeffs}, Self{m3()*_coeffs}};
    }


private:

    using Matrix10 = Eigen::Matrix<T, 10, 10>;
    using Basis = Eigen::Matrix<C, 10, 1>;

    static Basis makeBasis(const Coords& pos)
    {
        return (Basis{} << 1 * pos[0]*pos[0]*pos[0],
                           1 * pos[1]*pos[1]*pos[1],
                           1 * pos[2]*pos[2]*pos[2],
                           3 * pos[0]*pos[0]*pos[1],
                           3 * pos[0]*pos[1]*pos[1],
                           3 * pos[1]*pos[1]*pos[2],
                           3 * pos[1]*pos[2]*pos[2],
                           3 * pos[0]*pos[0]*pos[2],
                           3 * pos[0]*pos[2]*pos[2],
                           6 * pos[0]*pos[1]*pos[2]).finished();
    }

    // Coefficients of the Bezier Triangle
    Coeffs _coeffs;

    // Splitting operator matrices
    static Matrix10 m0()
    {
        static Matrix10 result = (Matrix10{} <<
                1,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            1./8., 1./8.,     0, 3./8., 3./8.,     0,     0,     0,     0,     0,
            1./8.,     0, 1./8.,     0,     0,     0,     0, 3./8., 3./8.,     0,
            1./2.,     0,     0, 1./2.,     0,     0,     0,     0,     0,     0,
            1./4.,     0,     0, 1./2., 1./4.,     0,     0,     0,     0,     0,
            1./8.,     0,     0, 1./4., 1./8., 1./8.,     0, 1./8.,     0, 1./4.,
            1./8.,     0,     0, 1./8.,     0,     0, 1./8., 1./4., 1./8., 1./4.,
            1./2.,     0,     0,     0,     0,     0,     0, 1./2.,     0,     0,
            1./4.,     0,     0,     0,     0,     0,     0, 1./2., 1./4.,     0,
            1./4.,     0,     0, 1./4.,     0,     0,     0, 1./4.,     0, 1./4.
        ).finished();
        return result;
    };

    static Matrix10 m1()
    {
        static Matrix10 result = (Matrix10{} <<
            1./8., 1./8.,     0, 3./8., 3./8.,     0,     0,     0,     0,     0,
                0,     1,     0,     0,     0,     0,     0,     0,     0,     0,
                0, 1./8., 1./8.,     0,     0, 3./8., 3./8.,     0,     0,     0,
                0, 1./4.,     0, 1./4., 1./2.,     0,     0,     0,     0,     0,
                0, 1./2.,     0,     0, 1./2.,     0,     0,     0,     0,     0,
                0, 1./2.,     0,     0,     0, 1./2.,     0,     0,     0,     0,
                0, 1./4.,     0,     0,     0, 1./2., 1./4.,     0,     0,     0,
                0, 1./8.,     0, 1./8., 1./4., 1./8.,     0, 1./8.,     0, 1./4.,
                0, 1./8.,     0,     0, 1./8., 1./4., 1./8.,     0, 1./8., 1./4.,
                0, 1./4.,     0,     0, 1./4., 1./4.,     0,     0,     0, 1./4.
        ).finished();
        return result;
    };

    static Matrix10 m2()
    {
        static Matrix10 result = (Matrix10{} <<
            1./8.,     0, 1./8.,     0,     0,     0,     0, 3./8., 3./8.,     0,
                0, 1./8., 1./8.,     0,     0, 3./8., 3./8.,     0,     0,     0,
                0,     0,   1,     0,     0,     0,     0,     0,     0,     0,
                0,     0, 1./8., 1./8.,     0,     0, 1./8., 1./8., 1./4., 1./4.,
                0,     0, 1./8.,     0, 1./8., 1./8., 1./4.,     0, 1./8., 1./4.,
                0,     0, 1./4.,     0,     0, 1./4., 1./2.,     0,     0,     0,
                0,     0, 1./2.,     0,     0,     0, 1./2.,     0,     0,     0,
                0,     0, 1./4.,     0,     0,     0,     0, 1./4., 1./2.,     0,
                0,     0, 1./2.,     0,     0,     0,     0,     0, 1./2.,     0,
                0,     0, 1./4.,     0,     0,     0, 1./4.,     0, 1./4., 1./4.
        ).finished();
        return result;
    };

    static Matrix10 m3()
    {
        static Matrix10 result = (Matrix10{} <<
                0, 1./8., 1./8.,     0,     0, 3./8., 3./8.,     0,     0,     0,
            1./8.,     0, 1./8.,     0,     0,     0,     0, 3./8., 3./8.,     0,
            1./8., 1./8.,     0, 3./8., 3./8.,     0,     0,     0,     0,     0,
                0,     0, 1./8.,     0, 1./8., 1./8., 1./4.,     0, 1./8., 1./4.,
                0,     0, 1./8., 1./8.,     0,     0, 1./8., 1./8., 1./4., 1./4.,
            1./8.,     0,     0, 1./8.,     0,     0, 1./8., 1./4., 1./8., 1./4.,
            1./8.,     0,     0, 1./4., 1./8., 1./8.,     0, 1./8.,     0, 1./4.,
                0, 1./8.,     0,     0, 1./8., 1./4., 1./8.,     0, 1./8., 1./4.,
                0, 1./8.,     0, 1./8., 1./4., 1./8.,     0, 1./8.,     0, 1./4.,
                0,     0,     0, 1./8., 1./8., 1./8., 1./8., 1./8., 1./8., 1./4.
        ).finished();
        return result;
    };
};

}

#endif

#include "TensorProductBezierTriangle3.hh"

#include "utils.hh"

namespace pev
{

template<typename T, typename C>
using TPBT3 = TensorProductBezierTriangle<T, C, 3>;

template<typename T, typename C>
template<int D>
std::array<TPBT3<T, C>, 4> TPBT3<T, C>::split() const
{
    static_assert(D == 0, "D must be 0");
    return {Self{splitCoeffs<0, D>(_coeffs)},
            Self{splitCoeffs<1, D>(_coeffs)},
            Self{splitCoeffs<2, D>(_coeffs)},
            Self{splitCoeffs<3, D>(_coeffs)}};
}


template<typename T, typename C>
const typename TPBT3<T, C>::DomainPoints& TPBT3<T, C>::domainPoints()
{
    static const auto result = (DomainPoints{} <<
          1.,   0.,   0.,
        2./3, 1./3,   0.,
        2./3,   0., 1./3,
        1./3, 2./3,   0.,
        1./3, 1./3, 1./3,
        1./3,   0., 2./3,
          0.,   1.,   0.,
          0., 2./3, 1./3,
          0., 1./3, 2./3,
          0.,   0.,   1.
    ).finished();
    return result;
}


template<typename T, typename C>
typename TPBT3<T, C>::Basis TPBT3<T, C>::makeBasis(const Coords& pos)
{
    return (Basis{} <<
            1 * pos[0] * pos[0] * pos[0],
            3 * pos[0] * pos[0] * pos[1],
            3 * pos[0] * pos[0] * pos[2],
            3 * pos[0] * pos[1] * pos[1],
            6 * pos[0] * pos[1] * pos[2],
            3 * pos[0] * pos[2] * pos[2],
            1 * pos[1] * pos[1] * pos[1],
            3 * pos[1] * pos[1] * pos[2],
            3 * pos[1] * pos[2] * pos[2],
            1 * pos[2] * pos[2] * pos[2]
            ).finished();
}


template<typename T, typename C>
template<int I, int D>
typename TPBT3<T, C>::Coeffs
TPBT3<T, C>::splitCoeffs(const Coeffs& in)
{
    static_assert(D == 0, "Split dimension D must be 0");
    static_assert(I >= 0 && I < 4, "Subdivision index must be between 0 and 3");
    auto out = Coeffs{};
    if(I == 0)
    {
        out[0] = in[0];
        out[1] = 1./2 * in[0] + 1./2 * in[1];
        out[2] = 1./2 * in[0] + 1./2 * in[2];
        out[3] = 1./4 * in[0] + 1./2 * in[1] + 1./4 * in[3];
        out[4] = 1./4 * in[0] + 1./4 * in[1] + 1./4 * in[2] + 1./4 * in[4];
        out[5] = 1./4 * in[0] + 1./2 * in[2] + 1./4 * in[5];
        out[6] = 1./8 * in[0] + 3./8 * in[1] + 3./8 * in[3] + 1./8 * in[6];
        out[7] = 1./8 * in[0] + 1./4 * in[1] + 1./8 * in[2] + 1./8 * in[3]
                 + 1./4 * in[4] + 1./8 * in[7];
        out[8] = 1./8 * in[0] + 1./8 * in[1] + 1./4 * in[2] + 1./4 * in[4]
                 + 1./8 * in[5] + 1./8 * in[8];
        out[9] = 1./8 * in[0] + 3./8 * in[2] + 3./8 * in[5] + 1./8 * in[9];
    }
    else if(I == 1)
    {
        out[0] = 1./8 * in[0] + 3./8 * in[1] + 3./8 * in[3] + 1./8 * in[6];
        out[1] = 1./4 * in[1] + 1./2 * in[3] + 1./4 * in[6];
        out[2] = 1./8 * in[1] + 1./8 * in[2] + 1./4 * in[3] + 1./4 * in[4]
                 + 1./8 * in[6] + 1./8 * in[7];
        out[3] = 1./2 * in[3] + 1./2 * in[6];
        out[4] = 1./4 * in[3] + 1./4 * in[4] + 1./4 * in[6] + 1./4 * in[7];
        out[5] = 1./8 * in[3] + 1./4 * in[4] + 1./8 * in[5] + 1./8 * in[6]
                 + 1./4 * in[7] + 1./8 * in[8];
        out[6] = in[6];
        out[7] = 1./2 * in[6] + 1./2 * in[7];
        out[8] = 1./4 * in[6] + 1./2 * in[7] + 1./4 * in[8];
        out[9] = 1./8 * in[6] + 3./8 * in[7] + 3./8 * in[8] + 1./8 * in[9];
    }
    else if(I == 2)
    {
        out[0] = 1./8 * in[0] + 3./8 * in[2] + 3./8 * in[5] + 1./8 * in[9];
        out[1] = 1./8 * in[1] + 1./8 * in[2] + 1./4 * in[4] + 1./4 * in[5]
                 + 1./8 * in[8] + 1./8 * in[9];
        out[2] = 1./4 * in[2] + 1./2 * in[5] + 1./4 * in[9];
        out[3] = 1./8 * in[3] + 1./4 * in[4] + 1./8 * in[5] + 1./8 * in[7]
                 + 1./4 * in[8] + 1./8 * in[9];
        out[4] = 1./4 * in[4] + 1./4 * in[5] + 1./4 * in[8] + 1./4 * in[9];
        out[5] = 1./2 * in[5] + 1./2 * in[9];
        out[6] = 1./8 * in[6] + 3./8 * in[7] + 3./8 * in[8] + 1./8 * in[9];
        out[7] = 1./4 * in[7] + 1./2 * in[8] + 1./4 * in[9];
        out[8] = 1./2 * in[8] + 1./2 * in[9];
        out[9] = in[9];
    }
    else
    {
        out[0] = 1./8 * in[0] + 3./8 * in[1] + 3./8 * in[3] + 1./8 * in[6];
        out[1] = 1./8 * in[1] + 1./8 * in[2] + 1./4 * in[3] + 1./4 * in[4]
                 + 1./8 * in[6] + 1./8 * in[7];
        out[2] = 1./8 * in[0] + 1./4 * in[1] + 1./8 * in[2] + 1./8 * in[3]
                 + 1./4 * in[4] + 1./8 * in[7];
        out[3] = 1./8 * in[3] + 1./4 * in[4] + 1./8 * in[5] + 1./8 * in[6]
                 + 1./4 * in[7] + 1./8 * in[8];
        out[4] = 1./8 * in[1] + 1./8 * in[2] + 1./8 * in[3] + 1./4 * in[4]
                 + 1./8 * in[5] + 1./8 * in[7] + 1./8 * in[8];
        out[5] = 1./8 * in[0] + 1./8 * in[1] + 1./4 * in[2] + 1./4 * in[4]
                 + 1./8 * in[5] + 1./8 * in[8];
        out[6] = 1./8 * in[6] + 3./8 * in[7] + 3./8 * in[8] + 1./8 * in[9];
        out[7] = 1./8 * in[3] + 1./4 * in[4] + 1./8 * in[5] + 1./8 * in[7]
                 + 1./4 * in[8] + 1./8 * in[9];
        out[8] = 1./8 * in[1] + 1./8 * in[2] + 1./4 * in[4] + 1./4 * in[5]
                 + 1./8 * in[8] + 1./8 * in[9];
        out[9] = 1./8 * in[0] + 3./8 * in[2] + 3./8 * in[5] + 1./8 * in[9];
    }
    return out;
}

template<typename T, typename C>
typename TPBT3<T, C>::Coeffs
TPBT3<T, C>::computeCoeffs(const Coeffs& in)
{
    auto out = Coeffs{};
    out[0] = in[0];
    out[1] = -5./6 * in[0] + 3. * in[1] + -3./2 * in[3] + 1./3 * in[6];
    out[2] = -5./6 * in[0] + 3. * in[2] + -3./2 * in[5] + 1./3 * in[9];
    out[3] = 1./3 * in[0] + -3./2 * in[1] + 3. * in[3] + -5./6 * in[6];
    out[4] = 1./3 * in[0] + -3./4 * in[1] + -3./4 * in[2] + -3./4 * in[3]
             + 9./2 * in[4] + -3./4 * in[5] + 1./3 * in[6] + -3./4 * in[7]
             + -3./4 * in[8] + 1./3 * in[9];
    out[5] = 1./3 * in[0] + -3./2 * in[2] + 3. * in[5] + -5./6 * in[9];
    out[6] = in[6];
    out[7] = -5./6 * in[6] + 3. * in[7] + -3./2 * in[8] + 1./3 * in[9];
    out[8] = 1./3 * in[6] + -3./2 * in[7] + 3. * in[8] + -5./6 * in[9];
    out[9] = in[9];
    return out;
}

}

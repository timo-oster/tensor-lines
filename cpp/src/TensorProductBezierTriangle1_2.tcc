#include "TensorProductBezierTriangle1_2.hh"

#include "utils.hh"

namespace pev
{

template<typename T, typename C>
using TPBT1_2 = TensorProductBezierTriangle<T, C, 1, 2>;


template<typename T, typename C>
const typename TPBT1_2<T, C>::DomainPoints& TPBT1_2<T, C>::domainPoints()
{
    static const auto result = (DomainPoints{} <<
        1.,   0.,   0.,   1.,   0.,   0.,
        1.,   0.,   0., 1./2, 1./2,   0.,
        1.,   0.,   0., 1./2,   0., 1./2,
        1.,   0.,   0.,   0.,   1.,   0.,
        1.,   0.,   0.,   0., 1./2, 1./2,
        1.,   0.,   0.,   0.,   0.,   1.,
        0.,   1.,   0.,   1.,   0.,   0.,
        0.,   1.,   0., 1./2, 1./2,   0.,
        0.,   1.,   0., 1./2,   0., 1./2,
        0.,   1.,   0.,   0.,   1.,   0.,
        0.,   1.,   0.,   0., 1./2, 1./2,
        0.,   1.,   0.,   0.,   0.,   1.,
        0.,   0.,   1.,   1.,   0.,   0.,
        0.,   0.,   1., 1./2, 1./2,   0.,
        0.,   0.,   1., 1./2,   0., 1./2,
        0.,   0.,   1.,   0.,   1.,   0.,
        0.,   0.,   1.,   0., 1./2, 1./2,
        0.,   0.,   1.,   0.,   0.,   1.
    ).finished();
    return result;
}


template<typename T, typename C>
typename TPBT1_2<T, C>::Basis TPBT1_2<T, C>::makeBasis(const Coords& pos)
{
    return (Basis{} <<
            1 * pos[0] * pos[3] * pos[3],
            2 * pos[0] * pos[3] * pos[4],
            2 * pos[0] * pos[3] * pos[5],
            1 * pos[0] * pos[4] * pos[4],
            2 * pos[0] * pos[4] * pos[5],
            1 * pos[0] * pos[5] * pos[5],
            1 * pos[1] * pos[3] * pos[3],
            2 * pos[1] * pos[3] * pos[4],
            2 * pos[1] * pos[3] * pos[5],
            1 * pos[1] * pos[4] * pos[4],
            2 * pos[1] * pos[4] * pos[5],
            1 * pos[1] * pos[5] * pos[5],
            1 * pos[2] * pos[3] * pos[3],
            2 * pos[2] * pos[3] * pos[4],
            2 * pos[2] * pos[3] * pos[5],
            1 * pos[2] * pos[4] * pos[4],
            2 * pos[2] * pos[4] * pos[5],
            1 * pos[2] * pos[5] * pos[5]
            ).finished();
}


template<typename T, typename C>
template<int I, int D>
typename TPBT1_2<T, C>::Coeffs
TPBT1_2<T, C>::splitCoeffs(const Coeffs& in)
{
    static_assert(D >= 0 && D < 2, "Split dimension D must be 0 or 1");
    static_assert(I >= 0 && I < 4, "Subdivision index must be between 0 and 3");
    auto out = Coeffs{};
    if(D == 0 && I == 0)
    {
        out[0] = in[0];
        out[1] = in[1];
        out[2] = in[2];
        out[3] = in[3];
        out[4] = in[4];
        out[5] = in[5];
        out[6] = 1./2 * in[0] + 1./2 * in[6];
        out[7] = 1./2 * in[1] + 1./2 * in[7];
        out[8] = 1./2 * in[2] + 1./2 * in[8];
        out[9] = 1./2 * in[3] + 1./2 * in[9];
        out[10] = 1./2 * in[4] + 1./2 * in[10];
        out[11] = 1./2 * in[5] + 1./2 * in[11];
        out[12] = 1./2 * in[0] + 1./2 * in[12];
        out[13] = 1./2 * in[1] + 1./2 * in[13];
        out[14] = 1./2 * in[2] + 1./2 * in[14];
        out[15] = 1./2 * in[3] + 1./2 * in[15];
        out[16] = 1./2 * in[4] + 1./2 * in[16];
        out[17] = 1./2 * in[5] + 1./2 * in[17];
    }
    else if(D == 0 && I == 1)
    {
        out[0] = 1./2 * in[0] + 1./2 * in[6];
        out[1] = 1./2 * in[1] + 1./2 * in[7];
        out[2] = 1./2 * in[2] + 1./2 * in[8];
        out[3] = 1./2 * in[3] + 1./2 * in[9];
        out[4] = 1./2 * in[4] + 1./2 * in[10];
        out[5] = 1./2 * in[5] + 1./2 * in[11];
        out[6] = in[6];
        out[7] = in[7];
        out[8] = in[8];
        out[9] = in[9];
        out[10] = in[10];
        out[11] = in[11];
        out[12] = 1./2 * in[6] + 1./2 * in[12];
        out[13] = 1./2 * in[7] + 1./2 * in[13];
        out[14] = 1./2 * in[8] + 1./2 * in[14];
        out[15] = 1./2 * in[9] + 1./2 * in[15];
        out[16] = 1./2 * in[10] + 1./2 * in[16];
        out[17] = 1./2 * in[11] + 1./2 * in[17];
    }
    else if(D == 0 && I == 2)
    {
        out[0] = 1./2 * in[0] + 1./2 * in[12];
        out[1] = 1./2 * in[1] + 1./2 * in[13];
        out[2] = 1./2 * in[2] + 1./2 * in[14];
        out[3] = 1./2 * in[3] + 1./2 * in[15];
        out[4] = 1./2 * in[4] + 1./2 * in[16];
        out[5] = 1./2 * in[5] + 1./2 * in[17];
        out[6] = 1./2 * in[6] + 1./2 * in[12];
        out[7] = 1./2 * in[7] + 1./2 * in[13];
        out[8] = 1./2 * in[8] + 1./2 * in[14];
        out[9] = 1./2 * in[9] + 1./2 * in[15];
        out[10] = 1./2 * in[10] + 1./2 * in[16];
        out[11] = 1./2 * in[11] + 1./2 * in[17];
        out[12] = in[12];
        out[13] = in[13];
        out[14] = in[14];
        out[15] = in[15];
        out[16] = in[16];
        out[17] = in[17];
        out[17] = in[17];
    }
    else if(D == 0 && I == 3)
    {
        out[0] = 1./2 * in[0] + 1./2 * in[6];
        out[1] = 1./2 * in[1] + 1./2 * in[7];
        out[2] = 1./2 * in[2] + 1./2 * in[8];
        out[3] = 1./2 * in[3] + 1./2 * in[9];
        out[4] = 1./2 * in[4] + 1./2 * in[10];
        out[5] = 1./2 * in[5] + 1./2 * in[11];
        out[6] = 1./2 * in[6] + 1./2 * in[12];
        out[7] = 1./2 * in[7] + 1./2 * in[13];
        out[8] = 1./2 * in[8] + 1./2 * in[14];
        out[9] = 1./2 * in[9] + 1./2 * in[15];
        out[10] = 1./2 * in[10] + 1./2 * in[16];
        out[11] = 1./2 * in[11] + 1./2 * in[17];
        out[12] = 1./2 * in[0] + 1./2 * in[12];
        out[13] = 1./2 * in[1] + 1./2 * in[13];
        out[14] = 1./2 * in[2] + 1./2 * in[14];
        out[15] = 1./2 * in[3] + 1./2 * in[15];
        out[16] = 1./2 * in[4] + 1./2 * in[16];
        out[17] = 1./2 * in[5] + 1./2 * in[17];
    }
    else if(D == 1 && I == 0)
    {
        out[0] = in[0];
        out[1] = 1./2 * in[0] + 1./2 * in[1];
        out[2] = 1./2 * in[0] + 1./2 * in[2];
        out[3] = 1./4 * in[0] + 1./2 * in[1] + 1./4 * in[3];
        out[4] = 1./4 * in[0] + 1./4 * in[1] + 1./4 * in[2] + 1./4 * in[4];
        out[5] = 1./4 * in[0] + 1./2 * in[2] + 1./4 * in[5];
        out[6] = in[6];
        out[7] = 1./2 * in[6] + 1./2 * in[7];
        out[8] = 1./2 * in[6] + 1./2 * in[8];
        out[9] = 1./4 * in[6] + 1./2 * in[7] + 1./4 * in[9];
        out[10] = 1./4 * in[6] + 1./4 * in[7] + 1./4 * in[8] + 1./4 * in[10];
        out[11] = 1./4 * in[6] + 1./2 * in[8] + 1./4 * in[11];
        out[12] = in[12];
        out[13] = 1./2 * in[12] + 1./2 * in[13];
        out[14] = 1./2 * in[12] + 1./2 * in[14];
        out[15] = 1./4 * in[12] + 1./2 * in[13] + 1./4 * in[15];
        out[16] = 1./4 * in[12] + 1./4 * in[13] + 1./4 * in[14] + 1./4 * in[16];
        out[17] = 1./4 * in[12] + 1./2 * in[14] + 1./4 * in[17];
    }
    else if(D == 1 && I == 1)
    {
        out[0] = 1./4 * in[0] + 1./2 * in[1] + 1./4 * in[3];
        out[1] = 1./2 * in[1] + 1./2 * in[3];
        out[2] = 1./4 * in[1] + 1./4 * in[2] + 1./4 * in[3] + 1./4 * in[4];
        out[3] = in[3];
        out[4] = 1./2 * in[3] + 1./2 * in[4];
        out[5] = 1./4 * in[3] + 1./2 * in[4] + 1./4 * in[5];
        out[6] = 1./4 * in[6] + 1./2 * in[7] + 1./4 * in[9];
        out[7] = 1./2 * in[7] + 1./2 * in[9];
        out[8] = 1./4 * in[7] + 1./4 * in[8] + 1./4 * in[9] + 1./4 * in[10];
        out[9] = in[9];
        out[10] = 1./2 * in[9] + 1./2 * in[10];
        out[11] = 1./4 * in[9] + 1./2 * in[10] + 1./4 * in[11];
        out[12] = 1./4 * in[12] + 1./2 * in[13] + 1./4 * in[15];
        out[13] = 1./2 * in[13] + 1./2 * in[15];
        out[14] = 1./4 * in[13] + 1./4 * in[14] + 1./4 * in[15] + 1./4 * in[16];
        out[15] = in[15];
        out[16] = 1./2 * in[15] + 1./2 * in[16];
        out[17] = 1./4 * in[15] + 1./2 * in[16] + 1./4 * in[17];
    }
    else if(D == 1 && I == 2)
    {
        out[0] = 1./4 * in[0] + 1./2 * in[2] + 1./4 * in[5];
        out[1] = 1./4 * in[1] + 1./4 * in[2] + 1./4 * in[4] + 1./4 * in[5];
        out[2] = 1./2 * in[2] + 1./2 * in[5];
        out[3] = 1./4 * in[3] + 1./2 * in[4] + 1./4 * in[5];
        out[4] = 1./2 * in[4] + 1./2 * in[5];
        out[5] = in[5];
        out[6] = 1./4 * in[6] + 1./2 * in[8] + 1./4 * in[11];
        out[7] = 1./4 * in[7] + 1./4 * in[8] + 1./4 * in[10] + 1./4 * in[11];
        out[8] = 1./2 * in[8] + 1./2 * in[11];
        out[9] = 1./4 * in[9] + 1./2 * in[10] + 1./4 * in[11];
        out[10] = 1./2 * in[10] + 1./2 * in[11];
        out[11] = in[11];
        out[12] = 1./4 * in[12] + 1./2 * in[14] + 1./4 * in[17];
        out[13] = 1./4 * in[13] + 1./4 * in[14] + 1./4 * in[16] + 1./4 * in[17];
        out[14] = 1./2 * in[14] + 1./2 * in[17];
        out[15] = 1./4 * in[15] + 1./2 * in[16] + 1./4 * in[17];
        out[16] = 1./2 * in[16] + 1./2 * in[17];
        out[17] = in[17];
    }
    else
    {
        out[0] = 1./4 * in[0] + 1./2 * in[1] + 1./4 * in[3];
        out[1] = 1./4 * in[1] + 1./4 * in[2] + 1./4 * in[3] + 1./4 * in[4];
        out[2] = 1./4 * in[0] + 1./4 * in[1] + 1./4 * in[2] + 1./4 * in[4];
        out[3] = 1./4 * in[3] + 1./2 * in[4] + 1./4 * in[5];
        out[4] = 1./4 * in[1] + 1./4 * in[2] + 1./4 * in[4] + 1./4 * in[5];
        out[5] = 1./4 * in[0] + 1./2 * in[2] + 1./4 * in[5];
        out[6] = 1./4 * in[6] + 1./2 * in[7] + 1./4 * in[9];
        out[7] = 1./4 * in[7] + 1./4 * in[8] + 1./4 * in[9] + 1./4 * in[10];
        out[8] = 1./4 * in[6] + 1./4 * in[7] + 1./4 * in[8] + 1./4 * in[10];
        out[9] = 1./4 * in[9] + 1./2 * in[10] + 1./4 * in[11];
        out[10] = 1./4 * in[7] + 1./4 * in[8] + 1./4 * in[10] + 1./4 * in[11];
        out[11] = 1./4 * in[6] + 1./2 * in[8] + 1./4 * in[11];
        out[12] = 1./4 * in[12] + 1./2 * in[13] + 1./4 * in[15];
        out[13] = 1./4 * in[13] + 1./4 * in[14] + 1./4 * in[15] + 1./4 * in[16];
        out[14] = 1./4 * in[12] + 1./4 * in[13] + 1./4 * in[14] + 1./4 * in[16];
        out[15] = 1./4 * in[15] + 1./2 * in[16] + 1./4 * in[17];
        out[16] = 1./4 * in[13] + 1./4 * in[14] + 1./4 * in[16] + 1./4 * in[17];
        out[17] = 1./4 * in[12] + 1./2 * in[14] + 1./4 * in[17];
    }
    return out;
}


template<typename T, typename C>
typename TPBT1_2<T, C>::Coeffs
TPBT1_2<T, C>::computeCoeffs(const Coeffs& in)
{
    auto out = Coeffs{};
    out[0] = in[0];
    out[1] = -1./2 * in[0] + 2. * in[1] + -1./2 * in[3];
    out[2] = -1./2 * in[0] + 2. * in[2] + -1./2 * in[5];
    out[3] = in[3];
    out[4] = -1./2 * in[3] + 2. * in[4] + -1./2 * in[5];
    out[5] = in[5];
    out[6] = in[6];
    out[7] = -1./2 * in[6] + 2. * in[7] + -1./2 * in[9];
    out[8] = -1./2 * in[6] + 2. * in[8] + -1./2 * in[11];
    out[9] = in[9];
    out[10] = -1./2 * in[9] + 2. * in[10] + -1./2 * in[11];
    out[11] = in[11];
    out[12] = in[12];
    out[13] = -1./2 * in[12] + 2. * in[13] + -1./2 * in[15];
    out[14] = -1./2 * in[12] + 2. * in[14] + -1./2 * in[17];
    out[15] = in[15];
    out[16] = -1./2 * in[15] + 2. * in[16] + -1./2 * in[17];
    out[17] = in[17];
    return out;
}


template<typename T, typename C>
typename TensorProductDerivative<0, T, C, 1, 2>::DerivCoeffs
TensorProductDerivative<0, T, C, 1, 2>::deriv_op(const Coeffs& in, int dir)
{
    assert(dir >= 0 && dir < 3);
    switch(dir)
    {
        case 0:
        {
            auto out = DerivCoeffs{};
            out[0] = in[0];
            out[1] = in[1];
            out[2] = in[2];
            out[3] = in[3];
            out[4] = in[4];
            out[5] = in[5];
            return out;
        }
        case 1:
        {
            auto out = DerivCoeffs{};
            out[0] = in[6];
            out[1] = in[7];
            out[2] = in[8];
            out[3] = in[9];
            out[4] = in[10];
            out[5] = in[11];
            return out;
        }
        case 2:
        {
            auto out = DerivCoeffs{};
            out[0] = in[12];
            out[1] = in[13];
            out[2] = in[14];
            out[3] = in[15];
            out[4] = in[16];
            out[5] = in[17];
            return out;
        }
        default:
            assert(false);
    }
}

template<typename T, typename C>
typename TensorProductDerivative<1, T, C, 1, 2>::DerivCoeffs
TensorProductDerivative<1, T, C, 1, 2>::deriv_op(const Coeffs& in, int dir)
{
    assert(dir >= 0 && dir < 3);
    switch(dir)
    {
        case 0:
        {
            auto out = DerivCoeffs{};
            out[0] = 2. * in[0];
            out[1] = 2. * in[1];
            out[2] = 2. * in[2];
            out[3] = 2. * in[6];
            out[4] = 2. * in[7];
            out[5] = 2. * in[8];
            out[6] = 2. * in[12];
            out[7] = 2. * in[13];
            out[8] = 2. * in[14];
            return out;
        }
        case 1:
        {
            auto out = DerivCoeffs{};
            out[0] = 2. * in[1];
            out[1] = 2. * in[3];
            out[2] = 2. * in[4];
            out[3] = 2. * in[7];
            out[4] = 2. * in[9];
            out[5] = 2. * in[10];
            out[6] = 2. * in[13];
            out[7] = 2. * in[15];
            out[8] = 2. * in[16];
            return out;
        }
        case 2:
        {
            auto out = DerivCoeffs{};
            out[0] = 2. * in[2];
            out[1] = 2. * in[4];
            out[2] = 2. * in[5];
            out[3] = 2. * in[8];
            out[4] = 2. * in[10];
            out[5] = 2. * in[11];
            out[6] = 2. * in[14];
            out[7] = 2. * in[16];
            out[8] = 2. * in[17];
            return out;
        }
        default:
            assert(false);
    }
}

} // namespace pev

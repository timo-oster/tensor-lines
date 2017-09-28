#include "TensorProductBezierTriangle2.hh"

#include "utils.hh"

namespace pev
{

template<typename T, typename C>
using TPBT2 = TensorProductBezierTriangle<T, C, 2>;


template<typename T, typename C>
const typename TPBT2<T, C>::DomainPoints& TPBT2<T, C>::domainPoints()
{
    static const auto result = (DomainPoints{} <<
          1.,   0.,   0.,
        1./2, 1./2,   0.,
        1./2,   0., 1./2,
          0.,   1.,   0.,
          0., 1./2, 1./2,
          0.,   0.,   1.
    ).finished();
    return result;
}


template<typename T, typename C>
typename TPBT2<T, C>::Basis TPBT2<T, C>::makeBasis(const Coords& pos)
{
    return (Basis{} <<
                1 * pos[0] * pos[0],
                2 * pos[0] * pos[1],
                2 * pos[0] * pos[2],
                1 * pos[1] * pos[1],
                2 * pos[1] * pos[2],
                1 * pos[2] * pos[2]
                ).finished();
}


template<typename T, typename C>
template<std::size_t I, std::size_t D>
typename TPBT2<T, C>::Coeffs
TPBT2<T, C>::splitCoeffs(const Coeffs& in)
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
    }
    else if(I == 1)
    {
        out[0] = 1./4 * in[0] + 1./2 * in[1] + 1./4 * in[3];
        out[1] = 1./2 * in[1] + 1./2 * in[3];
        out[2] = 1./4 * in[1] + 1./4 * in[2] + 1./4 * in[3] + 1./4 * in[4];
        out[3] = in[3];
        out[4] = 1./2 * in[3] + 1./2 * in[4];
        out[5] = 1./4 * in[3] + 1./2 * in[4] + 1./4 * in[5];
    }
    else if(I == 2)
    {
        out[0] = 1./4 * in[0] + 1./2 * in[2] + 1./4 * in[5];
        out[1] = 1./4 * in[1] + 1./4 * in[2] + 1./4 * in[4] + 1./4 * in[5];
        out[2] = 1./2 * in[2] + 1./2 * in[5];
        out[3] = 1./4 * in[3] + 1./2 * in[4] + 1./4 * in[5];
        out[4] = 1./2 * in[4] + 1./2 * in[5];
        out[5] = in[5];
    }
    else
    {
        out[0] = 1./4 * in[0] + 1./2 * in[1] + 1./4 * in[3];
        out[1] = 1./4 * in[1] + 1./4 * in[2] + 1./4 * in[3] + 1./4 * in[4];
        out[2] = 1./4 * in[0] + 1./4 * in[1] + 1./4 * in[2] + 1./4 * in[4];
        out[3] = 1./4 * in[3] + 1./2 * in[4] + 1./4 * in[5];
        out[4] = 1./4 * in[1] + 1./4 * in[2] + 1./4 * in[4] + 1./4 * in[5];
        out[5] = 1./4 * in[0] + 1./2 * in[2] + 1./4 * in[5];
    }
    return out;
}

template <typename T, typename C>
typename TPBT2<T, C>::Coeffs
TPBT2<T, C>::computeCoeffs(const Coeffs& in)
{
    auto out = Coeffs{};
    out[0] = in[0];
    out[1] = -1./2 * in[0] + 2. * in[1] + -1./2 * in[3];
    out[2] = -1./2 * in[0] + 2. * in[2] + -1./2 * in[5];
    out[3] = in[3];
    out[4] = -1./2 * in[3] + 2. * in[4] + -1./2 * in[5];
    out[5] = in[5];
    return out;
}

} // namespace pev

#include "TensorProductBezierTriangle1.hh"

#include "utils.hh"

namespace pev
{

template<typename T, typename C>
using TPBT1 = TensorProductBezierTriangle<T, C, 1>;


template<typename T, typename C>
const typename TPBT1<T, C>::DomainPoints& TPBT1<T, C>::domainPoints()
{
    static const auto result = (DomainPoints{} <<
        1., 0., 0.,
        0., 1., 0.,
        0., 0., 1.
    ).finished();
    return result;
}


template<typename T, typename C>
typename TPBT1<T, C>::Basis TPBT1<T, C>::makeBasis(const Coords& pos)
{
    return pos;
}


template<typename T, typename C>
template<std::size_t I, std::size_t D>
typename TPBT1<T, C>::Coeffs
TPBT1<T, C>::splitCoeffs(const Coeffs& in)
{
    static_assert(D == 0, "Split dimension D must be 0");
    static_assert(I >= 0 && I < 4, "Subdivision index must be between 0 and 3");
    auto out = Coeffs{};
    if(I == 0)
    {
        out[0] = in[0];
        out[1] = 1./2 * in[0] + 1./2 * in[1];
        out[2] = 1./2 * in[0] + 1./2 * in[2];
    }
    else if(I == 1)
    {
        out[0] = 1./2 * in[0] + 1./2 * in[1];
        out[1] = in[1];
        out[2] = 1./2 * in[1] + 1./2 * in[2];
    }
    else if(I == 2)
    {
        out[0] = 1./2 * in[0] + 1./2 * in[2];
        out[1] = 1./2 * in[1] + 1./2 * in[2];
        out[2] = in[2];
    }
    else
    {
        out[0] = 1./2 * in[0] + 1./2 * in[1];
        out[1] = 1./2 * in[1] + 1./2 * in[2];
        out[2] = 1./2 * in[0] + 1./2 * in[2];
    }
    return out;
}

template <typename T, typename C>
typename TPBT1<T, C>::Coeffs
TPBT1<T, C>::computeCoeffs(const Coeffs& in)
{
    return in;
}

} // namespace pev

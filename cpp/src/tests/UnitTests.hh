#include "../BezierDoubleTriangle.hh"
#include "../utils.hh"
#include <cxxtest/TestSuite.h>

using namespace pev;
typedef BezierDoubleTriangle<double> BDT;

class UnitTests : public CxxTest::TestSuite
{
public:
    void testSplits()
    {
        auto coeffs = BDT::Coeffs::Random().eval();

        auto orig = BDT{coeffs};
        auto splits = orig.split();

        for(auto i : range(4))
        {
            for(auto j : range(4))
            {
                auto si = 4 * i + j;
                const auto& part = splits[si];

                for(auto k : range(10))
                {
                    auto pos = ((BDT::Coords::Random() + BDT::Coords::Ones())
                                / 2).eval();
                    pos.head<3>() /= pos.head<3>().sum();
                    pos.tail<3>() /= pos.tail<3>().sum();

                    auto origpos = pos;
                    switch(i)
                    {
                        case 0:
                            origpos[0] += (origpos[1] + origpos[2]) / 2;
                            origpos[1] /= 2;
                            origpos[2] /= 2;
                            break;
                        case 1:
                            origpos[1] += (origpos[0] + origpos[2]) / 2;
                            origpos[0] /= 2;
                            origpos[2] /= 2;
                            break;
                        case 2:
                            origpos[2] += (origpos[0] + origpos[1]) / 2;
                            origpos[0] /= 2;
                            origpos[1] /= 2;
                            break;
                        case 3:
                            origpos[0] = (pos[0] + pos[2]) / 2;
                            origpos[1] = (pos[1] + pos[0]) / 2;
                            origpos[2] = (pos[2] + pos[1]) / 2;
                            break;
                        default:
                            break;
                    }
                    switch(j)
                    {
                        case 0:
                            origpos[3] += (origpos[4] + origpos[5]) / 2;
                            origpos[4] /= 2;
                            origpos[5] /= 2;
                            break;
                        case 1:
                            origpos[4] += (origpos[3] + origpos[5]) / 2;
                            origpos[3] /= 2;
                            origpos[5] /= 2;
                            break;
                        case 2:
                            origpos[5] += (origpos[3] + origpos[4]) / 2;
                            origpos[3] /= 2;
                            origpos[4] /= 2;
                            break;
                        case 3:
                            origpos[3] = (pos[3] + pos[5]) / 2;
                            origpos[4] = (pos[4] + pos[3]) / 2;
                            origpos[5] = (pos[5] + pos[4]) / 2;
                            break;
                        default:
                            break;
                    }

                    TS_ASSERT_DELTA(part(pos), orig(origpos), 1e-9);
                }
            }
        }
    }
};

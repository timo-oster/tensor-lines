#include "../TensorProductBezierTriangle.hh"
#include "../utils.hh"

#include <cxxtest/TestSuite.h>

#include <iostream>

typedef pev::TensorProductBezierTriangle<double, double, 1, 2> TPBT1_2;
typedef pev::TensorProductBezierTriangle<double, double, 1, 3> TPBT1_3;
typedef pev::TensorProductBezierTriangle<double, double, 3> TPBT3;
typedef pev::TensorProductBezierTriangle<pev::Vec3d, double, 1> Triangle;
// todo: test other variants of the class

class UnitTests : public CxxTest::TestSuite
{
    typedef TPBT1_2::Coeffs Coeffs1_2;
    typedef TPBT1_2::Coords Coords1_2;
    typedef TPBT1_3::Coeffs Coeffs1_3;
    typedef TPBT1_3::Coords Coords1_3;
    typedef TPBT3::Coeffs Coeffs3;
    typedef TPBT3::Coords Coords3;

    std::function<double(const Coords1_2&)> _poly1_2;
    std::function<double(const Coords1_3&)> _poly1_3;
    std::function<double(const Coords1_3&)> _poly1_3_d00;
    std::function<double(const Coords1_3&)> _poly1_3_d01;
    std::function<double(const Coords1_3&)> _poly1_3_d02;
    std::function<double(const Coords1_3&)> _poly1_3_d10;
    std::function<double(const Coords1_3&)> _poly1_3_d11;
    std::function<double(const Coords1_3&)> _poly1_3_d12;
    std::function<double(const Coords3&)> _poly3;

public:
    void setUp()
    {
        _poly1_2 = [](const Coords1_2& pos) {
            return 5.1 * 1 * pos[0] * pos[3] * pos[3]
                   + 8.1 * 2 * pos[0] * pos[3] * pos[4]
                   + 2.4 * 2 * pos[0] * pos[3] * pos[5]
                   + 6.5 * 1 * pos[0] * pos[4] * pos[4]
                   + 7.1 * 2 * pos[0] * pos[4] * pos[5]
                   + 8.4 * 1 * pos[0] * pos[5] * pos[5]
                   + 6.2 * 1 * pos[1] * pos[3] * pos[3]
                   + 3.8 * 2 * pos[1] * pos[3] * pos[4]
                   + 4.1 * 2 * pos[1] * pos[3] * pos[5]
                   + 6.7 * 1 * pos[1] * pos[4] * pos[4]
                   + 2.8 * 2 * pos[1] * pos[4] * pos[5]
                   + 6.3 * 1 * pos[1] * pos[5] * pos[5]
                   + 9.5 * 1 * pos[2] * pos[3] * pos[3]
                   + 4.7 * 2 * pos[2] * pos[3] * pos[4]
                   + 5.1 * 2 * pos[2] * pos[3] * pos[5]
                   + 3.8 * 1 * pos[2] * pos[4] * pos[4]
                   + 4.7 * 2 * pos[2] * pos[4] * pos[5]
                   + 3.6 * 1 * pos[2] * pos[5] * pos[5];
        };

        _poly1_3 = [](const Coords1_3& pos) {
            return 3.4 * 1 * pos[0] * pos[3] * pos[3] * pos[3]
                   + 3.8 * 3 * pos[0] * pos[3] * pos[3] * pos[4]
                   + 9.2 * 3 * pos[0] * pos[3] * pos[3] * pos[5]
                   + 7.1 * 3 * pos[0] * pos[3] * pos[4] * pos[4]
                   + 6.8 * 6 * pos[0] * pos[3] * pos[4] * pos[5]
                   + 1.5 * 3 * pos[0] * pos[3] * pos[5] * pos[5]
                   + 6.7 * 1 * pos[0] * pos[4] * pos[4] * pos[4]
                   + 1.8 * 3 * pos[0] * pos[4] * pos[4] * pos[5]
                   + 6.2 * 3 * pos[0] * pos[4] * pos[5] * pos[5]
                   + 8.7 * 1 * pos[0] * pos[5] * pos[5] * pos[5]
                   + 9.9 * 1 * pos[1] * pos[3] * pos[3] * pos[3]
                   + 1.4 * 3 * pos[1] * pos[3] * pos[3] * pos[4]
                   + 5.7 * 3 * pos[1] * pos[3] * pos[3] * pos[5]
                   + 7.1 * 3 * pos[1] * pos[3] * pos[4] * pos[4]
                   + 6.1 * 6 * pos[1] * pos[3] * pos[4] * pos[5]
                   + 7.4 * 3 * pos[1] * pos[3] * pos[5] * pos[5]
                   + 2.6 * 1 * pos[1] * pos[4] * pos[4] * pos[4]
                   + 9.4 * 3 * pos[1] * pos[4] * pos[4] * pos[5]
                   + 4.8 * 3 * pos[1] * pos[4] * pos[5] * pos[5]
                   + 9.3 * 1 * pos[1] * pos[5] * pos[5] * pos[5]
                   + 2.7 * 1 * pos[2] * pos[3] * pos[3] * pos[3]
                   + 6.4 * 3 * pos[2] * pos[3] * pos[3] * pos[4]
                   + 5.1 * 3 * pos[2] * pos[3] * pos[3] * pos[5]
                   + 7.2 * 3 * pos[2] * pos[3] * pos[4] * pos[4]
                   + 6.5 * 6 * pos[2] * pos[3] * pos[4] * pos[5]
                   + 5.5 * 3 * pos[2] * pos[3] * pos[5] * pos[5]
                   + 1.2 * 1 * pos[2] * pos[4] * pos[4] * pos[4]
                   + 8.4 * 3 * pos[2] * pos[4] * pos[4] * pos[5]
                   + 8.7 * 3 * pos[2] * pos[4] * pos[5] * pos[5]
                   + 7.6 * 1 * pos[2] * pos[5] * pos[5] * pos[5];
        };

        _poly1_3_d00 = [](const Coords1_3& pos) {
            return 3.4 * 1 * pos[3] * pos[3] * pos[3]
                   + 3.8 * 3 * pos[3] * pos[3] * pos[4]
                   + 9.2 * 3 * pos[3] * pos[3] * pos[5]
                   + 7.1 * 3 * pos[3] * pos[4] * pos[4]
                   + 6.8 * 6 * pos[3] * pos[4] * pos[5]
                   + 1.5 * 3 * pos[3] * pos[5] * pos[5]
                   + 6.7 * 1 * pos[4] * pos[4] * pos[4]
                   + 1.8 * 3 * pos[4] * pos[4] * pos[5]
                   + 6.2 * 3 * pos[4] * pos[5] * pos[5]
                   + 8.7 * 1 * pos[5] * pos[5] * pos[5];
        };

        _poly1_3_d01 = [](const Coords1_3& pos) {
            return 9.9 * 1 * pos[3] * pos[3] * pos[3]
                   + 1.4 * 3 * pos[3] * pos[3] * pos[4]
                   + 5.7 * 3 * pos[3] * pos[3] * pos[5]
                   + 7.1 * 3 * pos[3] * pos[4] * pos[4]
                   + 6.1 * 6 * pos[3] * pos[4] * pos[5]
                   + 7.4 * 3 * pos[3] * pos[5] * pos[5]
                   + 2.6 * 1 * pos[4] * pos[4] * pos[4]
                   + 9.4 * 3 * pos[4] * pos[4] * pos[5]
                   + 4.8 * 3 * pos[4] * pos[5] * pos[5]
                   + 9.3 * 1 * pos[5] * pos[5] * pos[5];
        };

        _poly1_3_d02 = [](const Coords1_3& pos) {
            return 2.7 * 1 * pos[3] * pos[3] * pos[3]
                   + 6.4 * 3 * pos[3] * pos[3] * pos[4]
                   + 5.1 * 3 * pos[3] * pos[3] * pos[5]
                   + 7.2 * 3 * pos[3] * pos[4] * pos[4]
                   + 6.5 * 6 * pos[3] * pos[4] * pos[5]
                   + 5.5 * 3 * pos[3] * pos[5] * pos[5]
                   + 1.2 * 1 * pos[4] * pos[4] * pos[4]
                   + 8.4 * 3 * pos[4] * pos[4] * pos[5]
                   + 8.7 * 3 * pos[4] * pos[5] * pos[5]
                   + 7.6 * 1 * pos[5] * pos[5] * pos[5];
        };

        _poly1_3_d10 = [](const Coords1_3& pos) {
            return 3.4 * 1 * 3 * pos[0] * pos[3] * pos[3]
                   + 3.8 * 3 * 2 * pos[0] * pos[3] * pos[4]
                   + 9.2 * 3 * 2 * pos[0] * pos[3] * pos[5]
                   + 7.1 * 3 * 1 * pos[0] * pos[4] * pos[4]
                   + 6.8 * 6 * 1 * pos[0] * pos[4] * pos[5]
                   + 1.5 * 3 * 1 * pos[0] * pos[5] * pos[5]
                   + 9.9 * 1 * 3 * pos[1] * pos[3] * pos[3]
                   + 1.4 * 3 * 2 * pos[1] * pos[3] * pos[4]
                   + 5.7 * 3 * 2 * pos[1] * pos[3] * pos[5]
                   + 7.1 * 3 * 1 * pos[1] * pos[4] * pos[4]
                   + 6.1 * 6 * 1 * pos[1] * pos[4] * pos[5]
                   + 7.4 * 3 * 1 * pos[1] * pos[5] * pos[5]
                   + 2.7 * 1 * 3 * pos[2] * pos[3] * pos[3]
                   + 6.4 * 3 * 2 * pos[2] * pos[3] * pos[4]
                   + 5.1 * 3 * 2 * pos[2] * pos[3] * pos[5]
                   + 7.2 * 3 * 1 * pos[2] * pos[4] * pos[4]
                   + 6.5 * 6 * 1 * pos[2] * pos[4] * pos[5]
                   + 5.5 * 3 * 1 * pos[2] * pos[5] * pos[5];
        };

        _poly1_3_d11 = [](const Coords1_3& pos) {
            return 3.8 * 3 * 1 * pos[0] * pos[3] * pos[3]
                   + 7.1 * 3 * 2 * pos[0] * pos[3] * pos[4]
                   + 6.8 * 6 * 1 * pos[0] * pos[3] * pos[5]
                   + 6.7 * 1 * 3 * pos[0] * pos[4] * pos[4]
                   + 1.8 * 3 * 2 * pos[0] * pos[4] * pos[5]
                   + 6.2 * 3 * 1 * pos[0] * pos[5] * pos[5]
                   + 1.4 * 3 * 1 * pos[1] * pos[3] * pos[3]
                   + 7.1 * 3 * 2 * pos[1] * pos[3] * pos[4]
                   + 6.1 * 6 * 1 * pos[1] * pos[3] * pos[5]
                   + 2.6 * 1 * 3 * pos[1] * pos[4] * pos[4]
                   + 9.4 * 3 * 2 * pos[1] * pos[4] * pos[5]
                   + 4.8 * 3 * 1 * pos[1] * pos[5] * pos[5]
                   + 6.4 * 3 * 1 * pos[2] * pos[3] * pos[3]
                   + 7.2 * 3 * 2 * pos[2] * pos[3] * pos[4]
                   + 6.5 * 6 * 1 * pos[2] * pos[3] * pos[5]
                   + 1.2 * 1 * 3 * pos[2] * pos[4] * pos[4]
                   + 8.4 * 3 * 2 * pos[2] * pos[4] * pos[5]
                   + 8.7 * 3 * 1 * pos[2] * pos[5] * pos[5];
        };

        _poly1_3_d12 = [](const Coords1_3& pos) {
            return 9.2 * 3 * 1 * pos[0] * pos[3] * pos[3]
                   + 6.8 * 6 * 1 * pos[0] * pos[3] * pos[4]
                   + 1.5 * 3 * 2 * pos[0] * pos[3] * pos[5]
                   + 1.8 * 3 * 1 * pos[0] * pos[4] * pos[4]
                   + 6.2 * 3 * 2 * pos[0] * pos[4] * pos[5]
                   + 8.7 * 1 * 3 * pos[0] * pos[5] * pos[5]
                   + 5.7 * 3 * 1 * pos[1] * pos[3] * pos[3]
                   + 6.1 * 6 * 1 * pos[1] * pos[3] * pos[4]
                   + 7.4 * 3 * 2 * pos[1] * pos[3] * pos[5]
                   + 9.4 * 3 * 1 * pos[1] * pos[4] * pos[4]
                   + 4.8 * 3 * 2 * pos[1] * pos[4] * pos[5]
                   + 9.3 * 1 * 3 * pos[1] * pos[5] * pos[5]
                   + 5.1 * 3 * 1 * pos[2] * pos[3] * pos[3]
                   + 6.5 * 6 * 1 * pos[2] * pos[3] * pos[4]
                   + 5.5 * 3 * 2 * pos[2] * pos[3] * pos[5]
                   + 8.4 * 3 * 1 * pos[2] * pos[4] * pos[4]
                   + 8.7 * 3 * 2 * pos[2] * pos[4] * pos[5]
                   + 7.6 * 1 * 3 * pos[2] * pos[5] * pos[5];
        };

        _poly3 = [](const Coords3& pos) {
            return 8.1 * 1 * pos[0] * pos[0] * pos[0]
                   + 7.4 * 3 * pos[0] * pos[0] * pos[1]
                   + 3.6 * 3 * pos[0] * pos[0] * pos[2]
                   + 5.2 * 3 * pos[0] * pos[1] * pos[1]
                   + 4.1 * 6 * pos[0] * pos[1] * pos[2]
                   + 8.3 * 3 * pos[0] * pos[2] * pos[2]
                   + 9.5 * 1 * pos[1] * pos[1] * pos[1]
                   + 3.7 * 3 * pos[1] * pos[1] * pos[2]
                   + 0.5 * 3 * pos[1] * pos[2] * pos[2]
                   + 1.3 * 1 * pos[2] * pos[2] * pos[2];
        };
    }

    void testPoly1_2()
    {
        auto bezier1_2 = TPBT1_2(_poly1_2);
        for(auto i : pev::range(15))
        {
            auto randpos =
                    ((Coords1_2::Random() + Coords1_2::Ones()) / 2).eval();
            randpos.head<3>() /= randpos.head<3>().sum();
            randpos.tail<3>() /= randpos.tail<3>().sum();
            TS_ASSERT_DELTA(bezier1_2(randpos), _poly1_2(randpos), 1e-9);
        }
    }

    void testPoly1_2Splits()
    {
        auto split = TPBT1_2(_poly1_2).split<0>();

        auto tris = Triangle{{pev::Vec3d{1., 0., 0.},
                              pev::Vec3d{0., 1., 0.},
                              pev::Vec3d{0., 0., 1.}}}
                            .split();

        for(auto i : pev::range(4))
        {
            for(auto _ : pev::range(15))
            {
                auto randpos =
                        ((Coords1_2::Random() + Coords1_2::Ones()) / 2).eval();
                randpos.head<3>() /= randpos.head<3>().sum();
                randpos.tail<3>() /= randpos.tail<3>().sum();
                auto globalpos = (Coords1_2{} << tris[i](randpos.head<3>()),
                                  randpos.tail<3>())
                                         .finished();
                TS_ASSERT_DELTA(split[i](randpos), _poly1_2(globalpos), 1e-9);
            }
        }

        split = TPBT1_2(_poly1_2).split<1>();

        for(auto i : pev::range(4))
        {
            for(auto _ : pev::range(15))
            {
                auto randpos =
                        ((Coords1_2::Random() + Coords1_2::Ones()) / 2).eval();
                randpos.head<3>() /= randpos.head<3>().sum();
                randpos.tail<3>() /= randpos.tail<3>().sum();
                auto globalpos = (Coords1_2{} << randpos.head<3>(),
                                  tris[i](randpos.tail<3>()))
                                         .finished();
                TS_ASSERT_DELTA(split[i](randpos), _poly1_2(globalpos), 1e-9);
            }
        }
    }

    void testPoly1_3()
    {
        auto bezier1_3 = TPBT1_3(_poly1_3);
        for(auto i : pev::range(15))
        {
            auto randpos =
                    ((Coords1_3::Random() + Coords1_3::Ones()) / 2).eval();
            randpos.head<3>() /= randpos.head<3>().sum();
            randpos.tail<3>() /= randpos.tail<3>().sum();
            TS_ASSERT_DELTA(bezier1_3(randpos), _poly1_3(randpos), 1e-9);
        }
    }

    void testPoly1_3Splits()
    {
        auto split = TPBT1_3(_poly1_3).split<0>();

        auto tris = Triangle{{pev::Vec3d{1., 0., 0.},
                              pev::Vec3d{0., 1., 0.},
                              pev::Vec3d{0., 0., 1.}}}
                            .split();

        for(auto i : pev::range(4))
        {
            for(auto _ : pev::range(15))
            {
                auto randpos =
                        ((Coords1_3::Random() + Coords1_3::Ones()) / 2).eval();
                randpos.head<3>() /= randpos.head<3>().sum();
                randpos.tail<3>() /= randpos.tail<3>().sum();
                auto globalpos = (Coords1_3{} << tris[i](randpos.head<3>()),
                                  randpos.tail<3>())
                                         .finished();
                TS_ASSERT_DELTA(split[i](randpos), _poly1_3(globalpos), 1e-9);
            }
        }

        split = TPBT1_3(_poly1_3).split<1>();

        for(auto i : pev::range(4))
        {
            for(auto _ : pev::range(15))
            {
                auto randpos =
                        ((Coords1_3::Random() + Coords1_3::Ones()) / 2).eval();
                randpos.head<3>() /= randpos.head<3>().sum();
                randpos.tail<3>() /= randpos.tail<3>().sum();
                auto globalpos = (Coords1_3{} << randpos.head<3>(),
                                  tris[i](randpos.tail<3>()))
                                         .finished();
                TS_ASSERT_DELTA(split[i](randpos), _poly1_3(globalpos), 1e-9);
            }
        }
    }

    void testPoly1_3Deriv()
    {
        auto bezier1_3 = TPBT1_3(_poly1_3);
        auto bezier1_3_d00 = bezier1_3.derivative<0>(0);
        for(auto _ : pev::range(15))
        {
            auto randpos =
                    ((Coords1_3::Random() + Coords1_3::Ones()) / 2).eval();
            randpos.head<3>() /= randpos.head<3>().sum();
            randpos.tail<3>() /= randpos.tail<3>().sum();
            TS_ASSERT_DELTA(bezier1_3_d00(randpos.tail<3>()),
                            _poly1_3_d00(randpos),
                            1e-9);
        }

        auto bezier1_3_d01 = bezier1_3.derivative<0>(1);
        for(auto _ : pev::range(15))
        {
            auto randpos =
                    ((Coords1_3::Random() + Coords1_3::Ones()) / 2).eval();
            randpos.head<3>() /= randpos.head<3>().sum();
            randpos.tail<3>() /= randpos.tail<3>().sum();
            TS_ASSERT_DELTA(bezier1_3_d01(randpos.tail<3>()),
                            _poly1_3_d01(randpos),
                            1e-9);
        }

        auto bezier1_3_d02 = bezier1_3.derivative<0>(2);
        for(auto _ : pev::range(15))
        {
            auto randpos =
                    ((Coords1_3::Random() + Coords1_3::Ones()) / 2).eval();
            randpos.head<3>() /= randpos.head<3>().sum();
            randpos.tail<3>() /= randpos.tail<3>().sum();
            TS_ASSERT_DELTA(bezier1_3_d02(randpos.tail<3>()),
                            _poly1_3_d02(randpos),
                            1e-9);
        }

        auto bezier1_3_d10 = bezier1_3.derivative<1>(0);
        for(auto _ : pev::range(15))
        {
            auto randpos =
                    ((Coords1_3::Random() + Coords1_3::Ones()) / 2).eval();
            randpos.head<3>() /= randpos.head<3>().sum();
            randpos.tail<3>() /= randpos.tail<3>().sum();
            TS_ASSERT_DELTA(bezier1_3_d10(randpos),
                            _poly1_3_d10(randpos),
                            1e-9);
        }

        auto bezier1_3_d11 = bezier1_3.derivative<1>(1);
        for(auto _ : pev::range(15))
        {
            auto randpos =
                    ((Coords1_3::Random() + Coords1_3::Ones()) / 2).eval();
            randpos.head<3>() /= randpos.head<3>().sum();
            randpos.tail<3>() /= randpos.tail<3>().sum();
            TS_ASSERT_DELTA(bezier1_3_d11(randpos),
                            _poly1_3_d11(randpos),
                            1e-9);
        }

        auto bezier1_3_d12 = bezier1_3.derivative<1>(2);
        for(auto _ : pev::range(15))
        {
            auto randpos =
                    ((Coords1_3::Random() + Coords1_3::Ones()) / 2).eval();
            randpos.head<3>() /= randpos.head<3>().sum();
            randpos.tail<3>() /= randpos.tail<3>().sum();
            TS_ASSERT_DELTA(bezier1_3_d12(randpos),
                            _poly1_3_d12(randpos),
                            1e-9);
        }
    }

    void testPoly3()
    {
        auto bezier3 = TPBT3(_poly3);
        for(auto i : pev::range(15))
        {
            auto randpos = ((Coords3::Random() + Coords3::Ones()) / 2).eval();
            randpos.normalize();
            TS_ASSERT_DELTA(bezier3(randpos), _poly3(randpos), 1e-9);
        }
    }

    void testPoly3Splits()
    {
        auto split = TPBT3(_poly3).split<0>();

        auto tris = Triangle{{pev::Vec3d{1., 0., 0.},
                              pev::Vec3d{0., 1., 0.},
                              pev::Vec3d{0., 0., 1.}}}
                            .split();

        for(auto i : pev::range(4))
        {
            for(auto _ : pev::range(15))
            {
                auto randpos =
                        ((Coords3::Random() + Coords3::Ones()) / 2).eval();
                randpos.normalize();
                auto globalpos = tris[i](randpos);
                TS_ASSERT_DELTA(split[i](randpos), _poly3(globalpos), 1e-9);
            }
        }
    }
};

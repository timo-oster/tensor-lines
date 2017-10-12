#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include "../TensorProductBezierTriangle.hh"
#include "../utils.hh"

using doctest::Approx;

using TPBT1_2 = pev::TensorProductBezierTriangle<double, double, 1, 2>;
using TPBT1_3 = pev::TensorProductBezierTriangle<double, double, 1, 3>;
using TPBT3 = pev::TensorProductBezierTriangle<double, double, 3>;
using Triangle = pev::TensorProductBezierTriangle<pev::Vec3d, double, 1>;
// todo: test other variants of the class

using Coeffs1_2 = TPBT1_2::Coeffs;
using Coords1_2 = TPBT1_2::Coords;
using Coeffs1_3 = TPBT1_3::Coeffs;
using Coords1_3 = TPBT1_3::Coords;
using Coeffs3 = TPBT3::Coeffs;
using Coords3 = TPBT3::Coords;

TEST_CASE("Testing polynomial reproduction for Degree 3")
{
    GIVEN("A trivariate polynomial of degree 3")
    {
        auto _poly3 = [](const Coords3& pos) {
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

        WHEN("We try reproducing it with TensorProductBezierTriangle<double, "
             "double, 3>")
        {
            auto bezier3 = TPBT3(_poly3);

            THEN("It must correctly reproduce the value at 15 random locations")
            {
                for(auto i : pev::range(15))
                {
                    auto randpos =
                            ((Coords3::Random() + Coords3::Ones()) / 2).eval();
                    randpos.normalize();
                    REQUIRE(bezier3(randpos) == Approx(_poly3(randpos)));
                }
            }

            WHEN("We try to split it into 4 new triangles")
            {
                auto split = TPBT3(_poly3).split<0>();

                auto tris = Triangle{{pev::Vec3d{1., 0., 0.},
                                      pev::Vec3d{0., 1., 0.},
                                      pev::Vec3d{0., 0., 1.}}}
                                    .split();

                THEN("Each triangle must reproduce the original function in "
                     "its split-off part")
                {
                    for(auto i : pev::range(4))
                    {
                        for(auto _ : pev::range(15))
                        {
                            auto randpos =
                                    ((Coords3::Random() + Coords3::Ones())
                                     / 2).eval();
                            randpos.normalize();
                            auto globalpos = tris[i](randpos);
                            REQUIRE(split[i](randpos)
                                    == Approx(_poly3(globalpos)));
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("Testing polynomial reproduction for degrees 1, 2")
{
    GIVEN("A tensor product of trivariate polynomials of degree 1 and 2")
    {
        auto _poly1_2 = [](const Coords1_2& pos) {
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

        WHEN("We try reproducing it with TensorProductBezierTriangle<double, "
             "double, 1, 2>")
        {
            auto bezier1_2 = TPBT1_2(_poly1_2);

            THEN("It must correctly reproduce the value at 15 random locations")
            {
                for(auto i : pev::range(15))
                {
                    auto randpos = ((Coords1_2::Random() + Coords1_2::Ones())
                                    / 2).eval();
                    randpos.head<3>() /= randpos.head<3>().sum();
                    randpos.tail<3>() /= randpos.tail<3>().sum();
                    REQUIRE(bezier1_2(randpos) == Approx(_poly1_2(randpos)));
                }
            }

            auto tris = Triangle{{pev::Vec3d{1., 0., 0.},
                                  pev::Vec3d{0., 1., 0.},
                                  pev::Vec3d{0., 0., 1.}}}
                                .split();

            WHEN("We try to split it into 4 new triangles in the first space")
            {
                auto split = bezier1_2.split<0>();

                THEN("Each triangle must reproduce the original function in "
                     "its split-off part")
                {
                    for(auto i : pev::range(4))
                    {
                        for(auto _ : pev::range(15))
                        {
                            auto randpos =
                                    ((Coords1_2::Random() + Coords1_2::Ones())
                                     / 2).eval();
                            randpos.head<3>() /= randpos.head<3>().sum();
                            randpos.tail<3>() /= randpos.tail<3>().sum();
                            auto globalpos =
                                    (Coords1_2{} << tris[i](randpos.head<3>()),
                                     randpos.tail<3>())
                                            .finished();
                            REQUIRE(split[i](randpos)
                                    == Approx(_poly1_2(globalpos)));
                        }
                    }
                }
            }

            WHEN("We try to split it into 4 new triangles in the second space")
            {
                auto split = bezier1_2.split<1>();

                THEN("Each triangle must reproduce the original function in "
                     "its split-off part")
                {
                    for(auto i : pev::range(4))
                    {
                        for(auto _ : pev::range(15))
                        {
                            auto randpos =
                                    ((Coords1_2::Random() + Coords1_2::Ones())
                                     / 2).eval();
                            randpos.head<3>() /= randpos.head<3>().sum();
                            randpos.tail<3>() /= randpos.tail<3>().sum();
                            auto globalpos = (Coords1_2{} << randpos.head<3>(),
                                              tris[i](randpos.tail<3>()))
                                                     .finished();
                            REQUIRE(split[i](randpos)
                                    == Approx(_poly1_2(globalpos)));
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("Testing polynomial reproduction for degrees 1, 3")
{
    GIVEN("A tensor product of trivariate polynomials of degree 1 and 3")
    {
        auto _poly1_3 = [](const Coords1_3& pos) {
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

        WHEN("We try reproducing it with TensorProductBezierTriangle<double, "
             "double, 1, 3>")
        {
            auto bezier1_3 = TPBT1_3(_poly1_3);

            THEN("It must correctly reproduce the value at 15 random locations")
            {
                for(auto i : pev::range(15))
                {
                    auto randpos = ((Coords1_3::Random() + Coords1_3::Ones())
                                    / 2).eval();
                    randpos.head<3>() /= randpos.head<3>().sum();
                    randpos.tail<3>() /= randpos.tail<3>().sum();
                    REQUIRE(bezier1_3(randpos) == Approx(_poly1_3(randpos)));
                }
            }

            auto tris = Triangle{{pev::Vec3d{1., 0., 0.},
                                  pev::Vec3d{0., 1., 0.},
                                  pev::Vec3d{0., 0., 1.}}}
                                .split();

            WHEN("We try to split it into 4 new triangles in the first space")
            {
                auto split = bezier1_3.split<0>();

                THEN("Each triangle must reproduce the original function in "
                     "its split-off part")
                {
                    for(auto i : pev::range(4))
                    {
                        for(auto _ : pev::range(15))
                        {
                            auto randpos =
                                    ((Coords1_3::Random() + Coords1_3::Ones())
                                     / 2).eval();
                            randpos.head<3>() /= randpos.head<3>().sum();
                            randpos.tail<3>() /= randpos.tail<3>().sum();
                            auto globalpos =
                                    (Coords1_3{} << tris[i](randpos.head<3>()),
                                     randpos.tail<3>())
                                            .finished();
                            REQUIRE(split[i](randpos)
                                    == Approx(_poly1_3(globalpos)));
                        }
                    }
                }
            }

            WHEN("We try to split it into 4 new triangles in the second space")
            {
                auto split = bezier1_3.split<1>();

                THEN("Each triangle must reproduce the original function in "
                     "its split-off part")
                {
                    for(auto i : pev::range(4))
                    {
                        for(auto _ : pev::range(15))
                        {
                            auto randpos =
                                    ((Coords1_3::Random() + Coords1_3::Ones())
                                     / 2).eval();
                            randpos.head<3>() /= randpos.head<3>().sum();
                            randpos.tail<3>() /= randpos.tail<3>().sum();
                            auto globalpos = (Coords1_3{} << randpos.head<3>(),
                                              tris[i](randpos.tail<3>()))
                                                     .finished();
                            REQUIRE(split[i](randpos)
                                    == Approx(_poly1_3(globalpos)));
                        }
                    }
                }
            }

            GIVEN("Its derivative in the first variable of the first space")
            {
                auto _poly1_3_d00 = [](const Coords1_3& pos) {
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

                WHEN("We try to compute the derivative")
                {
                    auto bezier1_3_d00 = bezier1_3.derivative<0>(0);

                    THEN("It must match the correct solution at 15 random "
                         "positions")
                    {
                        for(auto _ : pev::range(15))
                        {
                            auto randpos =
                                    ((Coords1_3::Random() + Coords1_3::Ones())
                                     / 2).eval();
                            randpos.head<3>() /= randpos.head<3>().sum();
                            randpos.tail<3>() /= randpos.tail<3>().sum();
                            REQUIRE(bezier1_3_d00(randpos.tail<3>())
                                    == Approx(_poly1_3_d00(randpos)));
                        }
                    }
                }
            }

            GIVEN("Its derivative in the second variable of the first space")
            {
                auto _poly1_3_d01 = [](const Coords1_3& pos) {
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

                WHEN("We try to compute the derivative")
                {
                    auto bezier1_3_d01 = bezier1_3.derivative<0>(1);

                    THEN("It must match the correct solution at 15 random "
                         "positions")
                    {
                        for(auto _ : pev::range(15))
                        {
                            auto randpos =
                                    ((Coords1_3::Random() + Coords1_3::Ones())
                                     / 2).eval();
                            randpos.head<3>() /= randpos.head<3>().sum();
                            randpos.tail<3>() /= randpos.tail<3>().sum();
                            REQUIRE(bezier1_3_d01(randpos.tail<3>())
                                    == Approx(_poly1_3_d01(randpos)));
                        }
                    }
                }
            }

            GIVEN("Its derivative in the third variable of the first space")
            {
                auto _poly1_3_d02 = [](const Coords1_3& pos) {
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

                WHEN("We try to compute the derivative")
                {
                    auto bezier1_3_d02 = bezier1_3.derivative<0>(2);

                    THEN("It must match the correct solution at 15 random "
                         "positions")
                    {
                        for(auto _ : pev::range(15))
                        {
                            auto randpos =
                                    ((Coords1_3::Random() + Coords1_3::Ones())
                                     / 2).eval();
                            randpos.head<3>() /= randpos.head<3>().sum();
                            randpos.tail<3>() /= randpos.tail<3>().sum();
                            REQUIRE(bezier1_3_d02(randpos.tail<3>())
                                    == Approx(_poly1_3_d02(randpos)));
                        }
                    }
                }
            }

            GIVEN("Its derivative in the first variable of the second space")
            {
                auto _poly1_3_d10 = [](const Coords1_3& pos) {
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

                WHEN("We try to compute the derivative")
                {
                    auto bezier1_3_d10 = bezier1_3.derivative<1>(0);

                    THEN("It must match the correct solution at 15 random "
                         "positions")
                    {
                        for(auto _ : pev::range(15))
                        {
                            auto randpos =
                                    ((Coords1_3::Random() + Coords1_3::Ones())
                                     / 2).eval();
                            randpos.head<3>() /= randpos.head<3>().sum();
                            randpos.tail<3>() /= randpos.tail<3>().sum();
                            REQUIRE(bezier1_3_d10(randpos)
                                    == Approx(_poly1_3_d10(randpos)));
                        }
                    }
                }
            }

            GIVEN("Its derivative in the second variable of the second space")
            {
                auto _poly1_3_d11 = [](const Coords1_3& pos) {
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

                WHEN("We try to compute the derivative")
                {
                    auto bezier1_3_d11 = bezier1_3.derivative<1>(1);

                    THEN("It must match the correct solution at 15 random "
                         "positions")
                    {
                        for(auto _ : pev::range(15))
                        {
                            auto randpos =
                                    ((Coords1_3::Random() + Coords1_3::Ones())
                                     / 2).eval();
                            randpos.head<3>() /= randpos.head<3>().sum();
                            randpos.tail<3>() /= randpos.tail<3>().sum();
                            REQUIRE(bezier1_3_d11(randpos)
                                    == Approx(_poly1_3_d11(randpos)));
                        }
                    }
                }
            }

            GIVEN("Its derivative in the third variable of the second space")
            {
                auto _poly1_3_d12 = [](const Coords1_3& pos) {
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

                WHEN("We try to compute the derivative")
                {
                    auto bezier1_3_d12 = bezier1_3.derivative<1>(2);

                    THEN("It must match the correct solution at 15 random "
                         "positions")
                    {
                        for(auto _ : pev::range(15))
                        {
                            auto randpos =
                                    ((Coords1_3::Random() + Coords1_3::Ones())
                                     / 2).eval();
                            randpos.head<3>() /= randpos.head<3>().sum();
                            randpos.tail<3>() /= randpos.tail<3>().sum();
                            REQUIRE(bezier1_3_d12(randpos)
                                    == Approx(_poly1_3_d12(randpos)));
                        }
                    }
                }
            }
        }
    }
}

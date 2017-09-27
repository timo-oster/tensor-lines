#ifndef CPP_TENSOR_FIED_HH
#define CPP_TENSOR_FIED_HH

#include "utils.hh"


namespace pev
{
class TensorField
{
public:
    virtual ~TensorField() {}

    virtual Mat3d t(const Vec3d& x) const = 0;

    virtual Mat3d tx(const Vec3d& x) const = 0;

    virtual Mat3d ty(const Vec3d& x) const = 0;

    virtual Mat3d tz(const Vec3d& x) const = 0;
};

class TestField : public TensorField
{
public:
    Mat3d t(const Vec3d& x) const override
    {
        auto result = Mat3d{};

        result(0, 0) = (1.0/2.0)*pow(x[0], 2) + (5.0/4.0)*pow(x[1], 2);
        result(0, 1) = -3.0/4.0*x[0]*x[1];
        result(0, 2) = (1.0/4.0)*x[1]*(-pow(x[0], 2) - pow(x[1], 2) + 4);
        result(1, 0) = -3.0/4.0*x[0]*x[1];
        result(1, 1) = (5.0/4.0)*pow(x[0], 2) + (1.0/2.0)*pow(x[1], 2);
        result(1, 2) = (1.0/4.0)*x[0]*(pow(x[0], 2) + pow(x[1], 2) - 4);
        result(2, 0) = (1.0/4.0)*x[1]*(-pow(x[0], 2) - pow(x[1], 2) + 4);
        result(2, 1) = (1.0/4.0)*x[0]*(pow(x[0], 2) + pow(x[1], 2) - 4);
        result(2, 2) = (1.0/4.0)*pow(pow(x[0], 2) + pow(x[1], 2), 2) + 1;

        return result;
    }

    Mat3d tx(const Vec3d& x) const override
    {
        auto result = Mat3d{};

        result(0, 0) = x[0];
        result(0, 1) = -3.0/4.0*x[1];
        result(0, 2) = -1.0/2.0*x[0]*x[1];
        result(1, 0) = -3.0/4.0*x[1];
        result(1, 1) = (5.0/2.0)*x[0];
        result(1, 2) = (3.0/4.0)*pow(x[0], 2) + (1.0/4.0)*pow(x[1], 2) - 1;
        result(2, 0) = -1.0/2.0*x[0]*x[1];
        result(2, 1) = (3.0/4.0)*pow(x[0], 2) + (1.0/4.0)*pow(x[1], 2) - 1;
        result(2, 2) = x[0]*(pow(x[0], 2) + pow(x[1], 2));

        return result;
    }

    Mat3d ty(const Vec3d& x) const override
    {
        auto result = Mat3d{};

        result(0, 0) = (5.0/2.0)*x[1];
        result(0, 1) = -3.0/4.0*x[0];
        result(0, 2) = -1.0/4.0*pow(x[0], 2) - 3.0/4.0*pow(x[1], 2) + 1;
        result(1, 0) = -3.0/4.0*x[0];
        result(1, 1) = x[1];
        result(1, 2) = (1.0/2.0)*x[0]*x[1];
        result(2, 0) = -1.0/4.0*pow(x[0], 2) - 3.0/4.0*pow(x[1], 2) + 1;
        result(2, 1) = (1.0/2.0)*x[0]*x[1];
        result(2, 2) = x[1]*(pow(x[0], 2) + pow(x[1], 2));

        return result;
    }

    Mat3d tz(const Vec3d& /*x*/) const override
    {
        return Mat3d::Zero();
    }
};
} // namespace pev
#endif

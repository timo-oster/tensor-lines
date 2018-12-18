#ifndef CPP_TENSOR_FIED_HH
#define CPP_TENSOR_FIED_HH

#include "TensorLineDefinitions.hh"
#include "utils.hh"

namespace tl
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

        result(0, 0) = (1.0/4.0)*(-2*pow(x[0], 2)*(pow(x[0], 2) + pow(x[1], 2) + 1) - 3*pow(x[1], 2) + 4*(pow(x[0], 2) + pow(x[1], 2))*(pow(x[0], 2) + pow(x[1], 2) + 1))/((pow(x[0], 2) + pow(x[1], 2))*(pow(x[0], 2) + pow(x[1], 2) + 1));
        result(0, 1) = (1.0/4.0)*x[0]*x[1]*(-2*pow(x[0], 2) - 2*pow(x[1], 2) + 1)/((pow(x[0], 2) + pow(x[1], 2))*(pow(x[0], 2) + pow(x[1], 2) + 1));
        result(0, 2) = (3.0/4.0)*x[1]/(pow(x[0], 2) + pow(x[1], 2) + 1);
        result(1, 0) = -x[0]*x[1]*(2*pow(x[0], 2) + 2*pow(x[1], 2) - 1)/(4*pow(x[0], 4) + 8*pow(x[0], 2)*pow(x[1], 2) + 4*pow(x[0], 2) + 4*pow(x[1], 4) + 4*pow(x[1], 2));
        result(1, 1) = (1.0/4.0)*(4*pow(x[0], 2)*(pow(x[0], 2) + pow(x[1], 2)) + pow(x[0], 2) + 2*pow(x[1], 2)*(pow(x[0], 2) + pow(x[1], 2) + 1))/((pow(x[0], 2) + pow(x[1], 2))*(pow(x[0], 2) + pow(x[1], 2) + 1));
        result(1, 2) = -3*x[0]/(4*pow(x[0], 2) + 4*pow(x[1], 2) + 4);
        result(2, 0) = (3.0/4.0)*x[1]/(pow(x[0], 2) + pow(x[1], 2) + 1);
        result(2, 1) = -3*x[0]/(4*pow(x[0], 2) + 4*pow(x[1], 2) + 4);
        result(2, 2) = (1.0/4.0)*(pow(x[0], 2) + pow(x[1], 2) + 4)/(pow(x[0], 2) + pow(x[1], 2) + 1);

        return result;
    }

    Mat3d tx(const Vec3d& x) const override
    {
        auto result = Mat3d{};

        result(0, 0) = -1.0/2.0*x[0]*(-2*pow(x[0], 2)*(pow(x[0], 2) + pow(x[1], 2) + 1) - 3*pow(x[1], 2) + 4*(pow(x[0], 2) + pow(x[1], 2))*(pow(x[0], 2) + pow(x[1], 2) + 1))/((pow(x[0], 2) + pow(x[1], 2))*pow(pow(x[0], 2) + pow(x[1], 2) + 1, 2)) - 1.0/2.0*x[0]*(-2*pow(x[0], 2)*(pow(x[0], 2) + pow(x[1], 2) + 1) - 3*pow(x[1], 2) + 4*(pow(x[0], 2) + pow(x[1], 2))*(pow(x[0], 2) + pow(x[1], 2) + 1))/(pow(pow(x[0], 2) + pow(x[1], 2), 2)*(pow(x[0], 2) + pow(x[1], 2) + 1)) + (1.0/4.0)*(-4*pow(x[0], 3) + 8*x[0]*(pow(x[0], 2) + pow(x[1], 2)) + 4*x[0]*(pow(x[0], 2) + pow(x[1], 2) + 1))/((pow(x[0], 2) + pow(x[1], 2))*(pow(x[0], 2) + pow(x[1], 2) + 1));
        result(0, 1) = -1.0/2.0*pow(x[0], 2)*x[1]*(-2*pow(x[0], 2) - 2*pow(x[1], 2) + 1)/((pow(x[0], 2) + pow(x[1], 2))*pow(pow(x[0], 2) + pow(x[1], 2) + 1, 2)) - pow(x[0], 2)*x[1]/((pow(x[0], 2) + pow(x[1], 2))*(pow(x[0], 2) + pow(x[1], 2) + 1)) - 1.0/2.0*pow(x[0], 2)*x[1]*(-2*pow(x[0], 2) - 2*pow(x[1], 2) + 1)/(pow(pow(x[0], 2) + pow(x[1], 2), 2)*(pow(x[0], 2) + pow(x[1], 2) + 1)) + (1.0/4.0)*x[1]*(-2*pow(x[0], 2) - 2*pow(x[1], 2) + 1)/((pow(x[0], 2) + pow(x[1], 2))*(pow(x[0], 2) + pow(x[1], 2) + 1));
        result(0, 2) = -3.0/2.0*x[0]*x[1]/pow(pow(x[0], 2) + pow(x[1], 2) + 1, 2);
        result(1, 0) = -4*pow(x[0], 2)*x[1]/(4*pow(x[0], 4) + 8*pow(x[0], 2)*pow(x[1], 2) + 4*pow(x[0], 2) + 4*pow(x[1], 4) + 4*pow(x[1], 2)) - x[0]*x[1]*(2*pow(x[0], 2) + 2*pow(x[1], 2) - 1)*(-16*pow(x[0], 3) - 16*x[0]*pow(x[1], 2) - 8*x[0])/pow(4*pow(x[0], 4) + 8*pow(x[0], 2)*pow(x[1], 2) + 4*pow(x[0], 2) + 4*pow(x[1], 4) + 4*pow(x[1], 2), 2) - x[1]*(2*pow(x[0], 2) + 2*pow(x[1], 2) - 1)/(4*pow(x[0], 4) + 8*pow(x[0], 2)*pow(x[1], 2) + 4*pow(x[0], 2) + 4*pow(x[1], 4) + 4*pow(x[1], 2));
        result(1, 1) = -1.0/2.0*x[0]*(4*pow(x[0], 2)*(pow(x[0], 2) + pow(x[1], 2)) + pow(x[0], 2) + 2*pow(x[1], 2)*(pow(x[0], 2) + pow(x[1], 2) + 1))/((pow(x[0], 2) + pow(x[1], 2))*pow(pow(x[0], 2) + pow(x[1], 2) + 1, 2)) - 1.0/2.0*x[0]*(4*pow(x[0], 2)*(pow(x[0], 2) + pow(x[1], 2)) + pow(x[0], 2) + 2*pow(x[1], 2)*(pow(x[0], 2) + pow(x[1], 2) + 1))/(pow(pow(x[0], 2) + pow(x[1], 2), 2)*(pow(x[0], 2) + pow(x[1], 2) + 1)) + (1.0/4.0)*(8*pow(x[0], 3) + 4*x[0]*pow(x[1], 2) + 8*x[0]*(pow(x[0], 2) + pow(x[1], 2)) + 2*x[0])/((pow(x[0], 2) + pow(x[1], 2))*(pow(x[0], 2) + pow(x[1], 2) + 1));
        result(1, 2) = 24*pow(x[0], 2)/pow(4*pow(x[0], 2) + 4*pow(x[1], 2) + 4, 2) - 3/(4*pow(x[0], 2) + 4*pow(x[1], 2) + 4);
        result(2, 0) = -3.0/2.0*x[0]*x[1]/pow(pow(x[0], 2) + pow(x[1], 2) + 1, 2);
        result(2, 1) = 24*pow(x[0], 2)/pow(4*pow(x[0], 2) + 4*pow(x[1], 2) + 4, 2) - 3/(4*pow(x[0], 2) + 4*pow(x[1], 2) + 4);
        result(2, 2) = (1.0/2.0)*x[0]/(pow(x[0], 2) + pow(x[1], 2) + 1) - 1.0/2.0*x[0]*(pow(x[0], 2) + pow(x[1], 2) + 4)/pow(pow(x[0], 2) + pow(x[1], 2) + 1, 2);

        return result;
    }

    Mat3d ty(const Vec3d& x) const override
    {
        auto result = Mat3d{};

        result(0, 0) = -1.0/2.0*x[1]*(-2*pow(x[0], 2)*(pow(x[0], 2) + pow(x[1], 2) + 1) - 3*pow(x[1], 2) + 4*(pow(x[0], 2) + pow(x[1], 2))*(pow(x[0], 2) + pow(x[1], 2) + 1))/((pow(x[0], 2) + pow(x[1], 2))*pow(pow(x[0], 2) + pow(x[1], 2) + 1, 2)) - 1.0/2.0*x[1]*(-2*pow(x[0], 2)*(pow(x[0], 2) + pow(x[1], 2) + 1) - 3*pow(x[1], 2) + 4*(pow(x[0], 2) + pow(x[1], 2))*(pow(x[0], 2) + pow(x[1], 2) + 1))/(pow(pow(x[0], 2) + pow(x[1], 2), 2)*(pow(x[0], 2) + pow(x[1], 2) + 1)) + (1.0/4.0)*(-4*pow(x[0], 2)*x[1] + 8*x[1]*(pow(x[0], 2) + pow(x[1], 2)) + 8*x[1]*(pow(x[0], 2) + pow(x[1], 2) + 1) - 6*x[1])/((pow(x[0], 2) + pow(x[1], 2))*(pow(x[0], 2) + pow(x[1], 2) + 1));
        result(0, 1) = -1.0/2.0*x[0]*pow(x[1], 2)*(-2*pow(x[0], 2) - 2*pow(x[1], 2) + 1)/((pow(x[0], 2) + pow(x[1], 2))*pow(pow(x[0], 2) + pow(x[1], 2) + 1, 2)) - x[0]*pow(x[1], 2)/((pow(x[0], 2) + pow(x[1], 2))*(pow(x[0], 2) + pow(x[1], 2) + 1)) - 1.0/2.0*x[0]*pow(x[1], 2)*(-2*pow(x[0], 2) - 2*pow(x[1], 2) + 1)/(pow(pow(x[0], 2) + pow(x[1], 2), 2)*(pow(x[0], 2) + pow(x[1], 2) + 1)) + (1.0/4.0)*x[0]*(-2*pow(x[0], 2) - 2*pow(x[1], 2) + 1)/((pow(x[0], 2) + pow(x[1], 2))*(pow(x[0], 2) + pow(x[1], 2) + 1));
        result(0, 2) = -3.0/2.0*pow(x[1], 2)/pow(pow(x[0], 2) + pow(x[1], 2) + 1, 2) + (3.0/4.0)/(pow(x[0], 2) + pow(x[1], 2) + 1);
        result(1, 0) = -4*x[0]*pow(x[1], 2)/(4*pow(x[0], 4) + 8*pow(x[0], 2)*pow(x[1], 2) + 4*pow(x[0], 2) + 4*pow(x[1], 4) + 4*pow(x[1], 2)) - x[0]*x[1]*(2*pow(x[0], 2) + 2*pow(x[1], 2) - 1)*(-16*pow(x[0], 2)*x[1] - 16*pow(x[1], 3) - 8*x[1])/pow(4*pow(x[0], 4) + 8*pow(x[0], 2)*pow(x[1], 2) + 4*pow(x[0], 2) + 4*pow(x[1], 4) + 4*pow(x[1], 2), 2) - x[0]*(2*pow(x[0], 2) + 2*pow(x[1], 2) - 1)/(4*pow(x[0], 4) + 8*pow(x[0], 2)*pow(x[1], 2) + 4*pow(x[0], 2) + 4*pow(x[1], 4) + 4*pow(x[1], 2));
        result(1, 1) = -1.0/2.0*x[1]*(4*pow(x[0], 2)*(pow(x[0], 2) + pow(x[1], 2)) + pow(x[0], 2) + 2*pow(x[1], 2)*(pow(x[0], 2) + pow(x[1], 2) + 1))/((pow(x[0], 2) + pow(x[1], 2))*pow(pow(x[0], 2) + pow(x[1], 2) + 1, 2)) - 1.0/2.0*x[1]*(4*pow(x[0], 2)*(pow(x[0], 2) + pow(x[1], 2)) + pow(x[0], 2) + 2*pow(x[1], 2)*(pow(x[0], 2) + pow(x[1], 2) + 1))/(pow(pow(x[0], 2) + pow(x[1], 2), 2)*(pow(x[0], 2) + pow(x[1], 2) + 1)) + (1.0/4.0)*(8*pow(x[0], 2)*x[1] + 4*pow(x[1], 3) + 4*x[1]*(pow(x[0], 2) + pow(x[1], 2) + 1))/((pow(x[0], 2) + pow(x[1], 2))*(pow(x[0], 2) + pow(x[1], 2) + 1));
        result(1, 2) = 24*x[0]*x[1]/pow(4*pow(x[0], 2) + 4*pow(x[1], 2) + 4, 2);
        result(2, 0) = -3.0/2.0*pow(x[1], 2)/pow(pow(x[0], 2) + pow(x[1], 2) + 1, 2) + (3.0/4.0)/(pow(x[0], 2) + pow(x[1], 2) + 1);
        result(2, 1) = 24*x[0]*x[1]/pow(4*pow(x[0], 2) + 4*pow(x[1], 2) + 4, 2);
        result(2, 2) = (1.0/2.0)*x[1]/(pow(x[0], 2) + pow(x[1], 2) + 1) - 1.0/2.0*x[1]*(pow(x[0], 2) + pow(x[1], 2) + 4)/pow(pow(x[0], 2) + pow(x[1], 2) + 1, 2);

        return result;
    }

    Mat3d tz(const Vec3d& /*x*/) const override
    {
        return Mat3d::Zero();
    }
};


class TestField2 : public TensorField
{
public:
    Mat3d t(const Vec3d& x) const override
    {
        auto result = Mat3d{};


        result(0, 0) = (3.0/8.0)*pow(x[0], 2) - 1.0/4.0*x[0]*x[1] + (3.0/8.0)*pow(x[1], 2) + 1.0/10.0;
        result(0, 1) = (1.0/8.0)*pow(x[0], 2) - 1.0/8.0*pow(x[1], 2);
        result(0, 2) = (1.0/8.0)*pow(x[0], 3) - 3.0/8.0*pow(x[0], 2)*x[1] + (1.0/8.0)*x[0]*pow(x[1], 2) - 3.0/8.0*pow(x[1], 3) + (9.0/10.0)*x[1];
        result(1, 0) = (1.0/8.0)*pow(x[0], 2) - 1.0/8.0*pow(x[1], 2);
        result(1, 1) = (3.0/8.0)*pow(x[0], 2) + (1.0/4.0)*x[0]*x[1] + (3.0/8.0)*pow(x[1], 2) + 1.0/10.0;
        result(1, 2) = (3.0/8.0)*pow(x[0], 3) + (1.0/8.0)*pow(x[0], 2)*x[1] + (3.0/8.0)*x[0]*pow(x[1], 2) - 9.0/10.0*x[0] + (1.0/8.0)*pow(x[1], 3);
        result(2, 0) = 0.;
        result(2, 1) = 0.;
        result(2, 2) = 1.;

        return result;
    }

    Mat3d tx(const Vec3d& x) const override
    {
        auto result = Mat3d{};

        result(0, 0) = (3.0/4.0)*x[0] - 1.0/4.0*x[1];
        result(0, 1) = (1.0/4.0)*x[0];
        result(0, 2) = (3.0/8.0)*pow(x[0], 2) - 3.0/4.0*x[0]*x[1] + (1.0/8.0)*pow(x[1], 2);
        result(1, 0) = (1.0/4.0)*x[0];
        result(1, 1) = (3.0/4.0)*x[0] + (1.0/4.0)*x[1];
        result(1, 2) = (9.0/8.0)*pow(x[0], 2) + (1.0/4.0)*x[0]*x[1] + (3.0/8.0)*pow(x[1], 2) - 9.0/10.0;
        result(2, 0) = 0.;
        result(2, 1) = 0.;
        result(2, 2) = 0.;

        return result;
    }

    Mat3d ty(const Vec3d& x) const override
    {
        auto result = Mat3d{};

        result(0, 0) = -1.0/4.0*x[0] + (3.0/4.0)*x[1];
        result(0, 1) = -1.0/4.0*x[1];
        result(0, 2) = -3.0/8.0*pow(x[0], 2) + (1.0/4.0)*x[0]*x[1] - 9.0/8.0*pow(x[1], 2) + 9.0/10.0;
        result(1, 0) = -1.0/4.0*x[1];
        result(1, 1) = (1.0/4.0)*x[0] + (3.0/4.0)*x[1];
        result(1, 2) = (1.0/8.0)*pow(x[0], 2) + (3.0/4.0)*x[0]*x[1] + (3.0/8.0)*pow(x[1], 2);
        result(2, 0) = 0.;
        result(2, 1) = 0.;
        result(2, 2) = 0.;

        return result;
    }

    Mat3d tz(const Vec3d& /*x*/) const override
    {
        return Mat3d::Zero();
    }
};


class TestFieldImag : public TensorField
{
public:
    Mat3d t(const Vec3d& pos) const override
    {
        auto x = pos[0];
        auto y = pos[1];
        auto result = Mat3d{};
        result(0, 0) = 1.,
        result(0, 1) = 1./sqrt(pow(x, 2) + pow(y, 2) + 1);
        result(0, 2) = x/sqrt(pow(x, 2) + pow(y, 2) + 1);
        result(1, 0) = -1/sqrt(pow(x, 2) + pow(y, 2) + 1);
        result(1, 1) = 1.;
        result(1, 2) = y/sqrt(pow(x, 2) + pow(y, 2) + 1);
        result(2, 0) = -x/sqrt(pow(x, 2) + pow(y, 2) + 1);
        result(2, 1) = -y/sqrt(pow(x, 2) + pow(y, 2) + 1);
        result(2, 2) = 1.;
        return result;
    }

    Mat3d tx(const Vec3d& pos) const override
    {
        auto x = pos[0];
        auto y = pos[1];
        auto result = Mat3d{};
        result(0, 0) = 0.;
        result(0, 1) = -x/pow(pow(x, 2) + pow(y, 2) + 1, 3.0/2.0);
        result(0, 2) = -pow(x, 2)/pow(pow(x, 2) + pow(y, 2) + 1, 3.0/2.0) + pow(pow(x, 2) + pow(y, 2) + 1, -1.0/2.0);
        result(1, 0) = x/pow(pow(x, 2) + pow(y, 2) + 1, 3.0/2.0);
        result(1, 1) = 0.;
        result(1, 2) = -x*y/pow(pow(x, 2) + pow(y, 2) + 1, 3.0/2.0);
        result(2, 0) = pow(x, 2)/pow(pow(x, 2) + pow(y, 2) + 1, 3.0/2.0) - 1/sqrt(pow(x, 2) + pow(y, 2) + 1);
        result(2, 1) = x*y/pow(pow(x, 2) + pow(y, 2) + 1, 3.0/2.0);
        result(2, 2) = 0.;
        return result;
    }

    Mat3d ty(const Vec3d& pos) const override
    {
        auto x = pos[0];
        auto y = pos[1];
        auto result = Mat3d{};
        result(0, 0) = 0.;
        result(0, 1) = -y/pow(pow(x, 2) + pow(y, 2) + 1, 3.0/2.0);
        result(0, 2) = -x*y/pow(pow(x, 2) + pow(y, 2) + 1, 3.0/2.0);
        result(1, 0) = y/pow(pow(x, 2) + pow(y, 2) + 1, 3.0/2.0);
        result(1, 1) = 0.;
        result(1, 2) = -pow(y, 2)/pow(pow(x, 2) + pow(y, 2) + 1, 3.0/2.0) + pow(pow(x, 2) + pow(y, 2) + 1, -1.0/2.0);
        result(2, 0) = x*y/pow(pow(x, 2) + pow(y, 2) + 1, 3.0/2.0);
        result(2, 1) = pow(y, 2)/pow(pow(x, 2) + pow(y, 2) + 1, 3.0/2.0) - 1/sqrt(pow(x, 2) + pow(y, 2) + 1);
        result(2, 2) = 0.;
        return result;
    }

    Mat3d tz(const Vec3d& /*pos*/) const override
    {
        return Mat3d::Zero();
    }
};


class TensorVortexSimple : public TensorField
{
public:
    Mat3d t(const Vec3d& x) const override
    {
        auto vec = Vec3d{-x[1], x[0], 1.};
        return vec * vec.transpose();
    }

    Mat3d tx(const Vec3d& x) const override
    {
        auto result = Mat3d{};

        result(0, 0) = 0.;
        result(0, 1) = -x[1];
        result(0, 2) = 0.;
        result(1, 0) = -x[1];
        result(1, 1) = 2.*x[0];
        result(1, 2) = 1.;
        result(2, 0) = 0.;
        result(2, 1) = 1.;
        result(2, 2) = 0.;

        return result;
    }

    Mat3d ty(const Vec3d& x) const override
    {
        auto result = Mat3d{};

        result(0, 0) = 2.*x[1];
        result(0, 1) = -x[0];
        result(0, 2) = -1.;
        result(1, 0) = -x[0];
        result(1, 1) = 0.;
        result(1, 2) = 0.;
        result(2, 0) = -1.;
        result(2, 1) = 0.;
        result(2, 2) = 0.;

        return result;
    }

    Mat3d tz(const Vec3d& /*x*/) const override
    {
        return Mat3d::Zero();
    }
};


class Identity : public TensorField
{
public:
    Mat3d t(const Vec3d& /*x*/) const override
    {
        return Mat3d::Identity();
    }

    Mat3d tx(const Vec3d& /*x*/) const override
    {
        return Mat3d::Zero();
    }

    Mat3d ty(const Vec3d& /*x*/) const override
    {
        return Mat3d::Zero();
    }

    Mat3d tz(const Vec3d& /*x*/) const override
    {
        return Mat3d::Zero();
    }
};

class ConstantNonSymmetric : public TensorField
{
public:
    Mat3d t(const Vec3d& /*x*/) const override
    {
        return (Mat3d{} << 1., -0.5,  0.5,
                           0.,  0.5, -0.25,
                           0.,  0.,   0.25)
                .finished();
    }

    Mat3d tx(const Vec3d& /*x*/) const override
    {
        return Mat3d::Zero();
    }

    Mat3d ty(const Vec3d& /*x*/) const override
    {
        return Mat3d::Zero();
    }

    Mat3d tz(const Vec3d& /*x*/) const override
    {
        return Mat3d::Zero();
    }
};

class SingleTopoLine : public TensorField
{
public:
    Mat3d t(const Vec3d& pos) const override
    {
        auto z = pos[2];
        auto y = pos[1];
        auto x = pos[0];

        auto af = 8*x*(1./5)-4*y*(1./5)+(1./5)*z+1;
        auto bf = 2*x-3*y*(1./5)-2*z*(1./5)+1;
        auto cf = -(1./5)*x-(1./2)*y+7*z*(1./10);

        return (Mat3d{} << af, cf, 0,
                           cf, bf, 0,
                           0,  0,  4).finished();
    }

    Mat3d tx(const Vec3d& /*x*/) const override
    {
        return Mat3d::Zero();
    }

    Mat3d ty(const Vec3d& /*x*/) const override
    {
        return Mat3d::Zero();
    }

    Mat3d tz(const Vec3d& /*x*/) const override
    {
        return Mat3d::Zero();
    }
};


} // namespace tl
#endif

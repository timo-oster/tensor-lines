#!/usr/bin/env python3

import sys
import argparse

from fractions import *
from operator import mul, add
from math import factorial
from functools import reduce
import numpy as np
from itertools import product


def indices(rank, ndims=3):
    '''Generate indices for control points in n-dimensional barycentric coordinates'''
    return [i for i in product(reversed(range(rank + 1)), repeat=ndims) if sum(i) == rank]


def domain_points(rank, ndims=3):
    '''Generate coordinates for control points in n-dimensional barycentric coordinates'''
    if rank == 0:
        return [[1./3., 1./3., 1./3.]]
    return [[i / rank for i in n] for n in indices(rank, ndims)]


def binomial(rank, nums):
    '''Compute binomial coefficients for n-dimensional Bezier-Bernstein polynomials'''
    return factorial(rank) // (reduce(mul, (factorial(i) for i in nums)))


def binom_coeffs(rank, ndims):
    return [binomial(rank, i) for i in indices(rank, ndims)]


def transform_base(rank, p, ndims=3):
    '''Transform a barycentric coordinate into the polynomial base'''
    return np.array([binomial(rank, index) * reduce(mul, (p**i for p, i in zip(p, index))) for index in indices(rank)])


def system_matrix(rank, ndims=3):
    '''Build the system matrix consisting of all domain points transformed into the polynomial base.
       If C = system_matrix(rank, ndims) and f the values of the polynomial function at the domain points,
       then C*b = f, where b are the coefficients of the Bernstein-Bézier triangle.'''
    return np.array([transform_base(rank, p, ndims) for p in domain_points(rank, ndims)])


def split_trans(num):
    '''Compute the operators for transforming barycentric coordinates into four smaller triangles'''
    if num == 0:
        return np.array([[1, 0, 0],
                         [1 / 2, 1 / 2, 0],
                         [1 / 2, 0, 1 / 2]]).T
    elif num == 1:
        return np.array([[1 / 2, 1 / 2, 0],
                         [0, 1, 0],
                         [0, 1 / 2, 1 / 2]]).T
    elif num == 2:
        return np.array([[1 / 2, 0, 1 / 2],
                         [0, 1 / 2, 1 / 2],
                         [0, 0, 1]]).T

    elif num == 3:
        return np.array([[1 / 2, 1 / 2, 0],
                         [0, 1 / 2, 1 / 2],
                         [1 / 2, 0, 1 / 2]]).T
    else:
        return None


def split_operator(rank, num):
    '''Compute the operators for transforming the coefficients of a Bernstein-Bézier triangle into the
       coefficients for a smaller triangle.
               o
              / \
             / 2 \
            o-----o
           / \ 3 / \
          / 0 \ / 1 \
         o-----o-----o
    '''
    system_inv = np.linalg.inv(system_matrix(rank))
    split_coords = np.array([transform_base(rank, p) for p in (
        split_trans(num) @ np.array(domain_points(rank)).T).T])
    return system_inv @ split_coords


def deriv_operator(rank, dim, ndims=3):
    '''Generate the operator needed to transform coefficients of a polynomial of degree rank into coefficients of
        its derivative with degree rank-1'''
    select = [rank if i[dim] > 0 else 0 for i in indices(rank, ndims)]
    op = np.diag(select)
    return np.array([o for o in op if sum(o) > 0])


class tensor_product_bezier_triangle:
    def __init__(self, degrees, dims):
        assert len(degrees) == len(
            dims), "Length of degrees and dims must match"
        self.degrees = degrees
        self.dims = dims

    def multi_indices(self):
        if len(self.degrees) > 1:
            return np.array(list(
                reduce(add, l)
                for l in reduce(
                    product,
                    (indices(r, d)
                     for r, d in zip(self.degrees, self.dims)))))
        else:
            return np.array(list(
                reduce(
                    product,
                    (indices(r, d)
                     for r, d in zip(self.degrees, self.dims)))))

    def domain_points(self):
        if len(self.degrees) > 1:
            return np.array(list(
                reduce(add, l)
                for l in reduce(
                    product,
                    (domain_points(r, d)
                     for r, d in zip(self.degrees, self.dims)))))
        else:
            return np.array(list(
                reduce(
                    product,
                    (domain_points(r, d)
                     for r, d in zip(self.degrees, self.dims)))))

    def binom_coeffs(self):
        if len(self.degrees) > 1:
            return np.array(list(
                reduce(mul, b)
                for b in reduce(
                    product,
                    (binom_coeffs(r, d)
                     for r, d in zip(self.degrees, self.dims)))))
        else:
            return np.array(list(
                reduce(
                    product,
                    (binom_coeffs(r, d)
                     for r, d in zip(self.degrees, self.dims)))))

    def system_matrix(self):
        return np.array(list(reduce(np.kron, (system_matrix(r, d) for r, d in zip(self.degrees, self.dims)))))

    def system_inv(self):
        return np.linalg.inv(self.system_matrix())

    def split_operator(self, indices):
        assert len(self.degrees) == len(indices)
        return np.array(list(
            reduce(
                np.kron,
                (split_operator(r, i - 1)
                 if i > 0
                 else np.eye(((r + 1) * (r + 2)) // 2)
                 for r, i in zip(self.degrees, indices)))))

    def deriv_operator(self, space, dim):
        assert space < len(self.dims)
        assert dim < self.dims[space]
        return np.array(list(
            reduce(
                np.kron,
                (deriv_operator(r, dim)
                 if i == space
                 else np.eye(((r + 1) * (r + 2)) // 2)
                 for i, r in enumerate(self.degrees)))))


def format_coeff(c):
    f = Fraction(c).limit_denominator(100)
    if f.numerator == 0:
        return "0."
    elif f.denominator == 1:
        return str(f.numerator) + '.'
    else:
        return "{0.numerator}./{0.denominator}".format(Fraction(c).limit_denominator(100))


def make_list(list, sep=','):
    for i, l in enumerate(list[:-1]):
        list[i] = l + ","
    return list


def format_matrix(m, fieldw=5):
    lines = [", ".join(("{:>" + str(fieldw) + "}").format(s)
                       for s in (format_coeff(c) for c in line)) for line in m]
    return make_list(lines)


def generate_basis(tp):
    lines = [" * ".join(
            [str(b)] + [" * ".join(
                ["pos[" + str(i) + "]"] * index[i])
                for i in range(len(index))
                if index[i] > 0])
             for b, index
             in zip(tp.binom_coeffs(), tp.multi_indices())]
    return make_list(lines)


def generate_trans_func(mat):
    matrix = np.array(mat)

    for i in range(matrix.shape[0]):
        line = "out[" + str(i) + "] = "
        line += " + ".join(((format_coeff(matrix[i, j]) + " * ") if matrix[i, j] != 1. else "") + "in[" + str(
            j) + "]" for j in range(matrix.shape[1]) if abs(matrix[i, j]) > 1e-9) + ";"
        yield line


def indent(string, level, tab_width=4):
    return "\n".join(" " * (tab_width * level) + l for l in string.splitlines())


header_template = """#ifndef CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_{deg_str}_HH
#define CPP_TENSOR_PRODUCT_BEZIER_TRIANGLE_{deg_str}_HH
#include "TensorProductBezierTriangle.hh"

namespace pev
{{
{Traits}

{Class}

{Derivatives}
}} // namespace pev
#endif"""

traits_template = """template <typename T, typename C>
struct TensorProductTraits<TensorProductBezierTriangle<T, C, {deg}>>
{{
    static constexpr std::size_t NCoords = {NCoords};
    static constexpr std::size_t NCoeffs = {NCoeffs};
}};"""

class_template = """template <typename T, typename C>
class TensorProductBezierTriangle<T, C, {deg}>
        : public TensorProductBezierTriangleBase<
                    TensorProductBezierTriangle<T, C, {deg}>, T, C, {deg}>
{{
public:
    using Self = TensorProductBezierTriangle<T, C, {deg}>;
    using Base = TensorProductBezierTriangleBase<
                    TensorProductBezierTriangle<T, C, {deg}>, T, C, {deg}>;
    using Traits = TensorProductTraits<Self>;
    using Coords = typename Base::Coords;
    using Coeffs = typename Base::Coeffs;

    enum Indices : std::size_t
    {{
{Indices}
    }};

    friend Base;
    using Base::Base;

private:
    using Basis = Eigen::Matrix<C, Traits::NCoeffs, 1>;
    using DomainPoints = Eigen::Matrix<C, Traits::NCoeffs, Traits::NCoords>;
    template <std::size_t D>
    using DerivCoeffs =
            typename TensorProductDerivativeType<D, T, C, {deg}>::Coeffs;

    static const DomainPoints& domainPoints()
    {{
        static const auto result = (DomainPoints{{}} <<
{DomainPoints}
            ).finished();
        return result;
    }}

    static Basis makeBasis(const Coords& pos)
    {{
        return (Basis{{}} <<
{Basis}
            ).finished();
    }}

    template<std::size_t I, std::size_t D>
    static Coeffs splitCoeffs(const Coeffs& in)
    {{
        static_assert(D < {NDegrees}, "Split dimension D must be smaller than {NDegrees}");
        static_assert(I < 4, "Subdivision index must be between 0 and 3");
        auto out = Coeffs{{}};
{SplitCoeffs}
        return out;
    }}

    static Coeffs computeCoeffs(const Coeffs& in)
    {{
        auto out = Coeffs{{}};
{ComputeCoeffs}
        return out;
    }}
}};"""

deriv_template = """template <typename T, typename C>
struct TensorProductDerivativeType<{D}, T, C, {deg}>
{{
    using type = TensorProductBezierTriangle<T, C, {deriv_deg}>;
}};

template <typename T, typename C>
struct TensorProductDerivative<{D}, T, C, {deg}>
{{
    using Coeffs = typename TensorProductBezierTriangle<T, C, {deg}>::Coeffs;
    using DerivCoeffs = typename TensorProductDerivativeType_t<{D}, T, C, {deg}>::Coeffs;
    static DerivCoeffs deriv_op(const Coeffs& in, std::size_t dir)
    {{
        assert(dir >= 0 && dir < 3);
        auto out = DerivCoeffs{{}};
{DerivOps}
        return out;
    }}
}};"""


def codegen_indices(tp):
    return ",\n".join("i" + "".join(str(n) for n in i) for i in tp.multi_indices())


def codegen_splitcoeffs(tp):
    code = ("switch(D)\n" +
            "{\n")
    for d in range(len(tp.degrees)):
        code += ("    case {}: switch(I)\n" +
                 "    {{\n").format(d)
        for i in range(4):
            code += ("        case {}:\n" +
                     "        {{\n").format(i)
            split_indices = [0 for i in range(len(tp.degrees))]
            split_indices[d] = i + 1
            for s in generate_trans_func(tp.split_operator(split_indices)):
                code += "            " + s + "\n"
            code += ("        }\n" +
                     "        break;\n")
        code += ("    }\n" +
                 "    break;\n")
    code += "}\n"

    return code


def codegen_derivatives(tp):
    deg_args = ", ".join(str(i) for i in tp.degrees)
    derivatives = ""
    for d in range(len(tp.degrees)):
        if tp.degrees[d] > 0:
            deriv_deg = [i for i in tp.degrees]
            deriv_deg[d] -= 1
            deriv_deg = ", ".join(str(i) for i in deriv_deg)
            deriv_ops = ("switch(dir)\n" +
                         "{\n")
            for i in range(3):
                deriv_ops += ("    case {}:\n" +
                              "    {{\n").format(i)
                for s in generate_trans_func(tp.deriv_operator(d, i)):
                    deriv_ops += "        " + s + "\n"
                deriv_ops += ("    }\n"
                              "    break;\n")
            deriv_ops += ("    default:\n"
                          "        assert(false);\n")
            deriv_ops += "}\n"
            derivatives += deriv_template.format(
                D=d, deg=deg_args, deriv_deg=deriv_deg,
                DerivOps=indent(deriv_ops, 2))
    return derivatives


def generate_tpbt_file(degrees):
    deg_s = [str(i) for i in degrees]
    deg_str = '_'.join(deg_s)
    deg_args = ", ".join(deg_s)

    tp = tensor_product_bezier_triangle(degrees, [3 for i in degrees])

    ncoords = 3 * len(tp.degrees)
    ncoeffs = len(tp.multi_indices())

    cindices = indent(codegen_indices(tp), 2)
    domainpoints = indent("\n".join(format_matrix(tp.domain_points())), 3)
    basis = indent("\n".join(generate_basis(tp)), 3)
    splitcoeffs = indent(codegen_splitcoeffs(tp), 3)
    computecoeffs = indent("\n".join(generate_trans_func(tp.system_inv())), 3)

    traits_code = traits_template.format(deg=deg_args, NCoeffs=ncoeffs,
                                         NCoords=ncoords)

    class_code = class_template.format(deg=deg_args, NDegrees=len(tp.degrees),
                                       Indices=cindices,
                                       DomainPoints=domainpoints,
                                       Basis=basis, SplitCoeffs=splitcoeffs,
                                       ComputeCoeffs=computecoeffs)

    derivatives_code = codegen_derivatives(tp)

    return header_template.format(deg_str=deg_str, Traits=traits_code,
                                  Class=class_code,
                                  Derivatives=derivatives_code)


def main():

    parser = argparse.ArgumentParser(description="Generate source files for " +
                                     "TensorProductBezierTriangle "
                                     "specializations of multiple degrees")
    parser.add_argument("-p", "--pretend",
                        help="Print the names of the files that would be " +
                        "generated, but do not generate them",
                        action="store_true")
    parser.add_argument("degrees", metavar="D",
                        help="The maximum degrees of the polynomials",
                        type=int, nargs="+")
    args = parser.parse_args()

    combinations = product(*(range(i + 1) for i in args.degrees))
    filenames = []

    for deg in combinations:
        filename = "TensorProductBezierTriangle{}.hh".format(
            "_".join(str(i) for i in deg))
        filenames += [filename]
        if not args.pretend:
            code = generate_tpbt_file(deg)
            f = open(filename, mode='w')
            f.write(code)
            f.close()

    for f in filenames:
        print(f)


if __name__ == "__main__":
    main()

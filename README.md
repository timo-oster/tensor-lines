# Find Tensor Feature Lines in Piecewise Linear Tensor Fields

Implements algorithm for finding different feature lines in piecewise linear tensor fields. Includes degenerate lines, tensor core lines, and parallel eigenvector lines.
Based on the work from two of my papers:

* T. Oster, C. Rössl and H. Theisel,
  _Core Lines in 3D Second-Order Tensor Fields_,
  Computer Graphics Forum (Proc. EuroVis), 2018, https://doi.org/10.1111/cgf.13423

* T. Oster, C. Rössl and H. Theisel,
  _The Parallel Eigenvectors Operator_,
  Proc. of Vision, Modeling, and Visualization (VMV), 2018, https://doi.org/10.2312/vmv.20181251

## Dependencies

* [Eigen 3](http://eigen.tuxfamily.org)
* [Boost](http://www.boost.org/)
* [VTK](http://www.vtk.org/)
* [Python 3](https://www.python.org/)
* [cpp_utils](https://github.com/timo-oster/cpp-utils)
* OpenMP (optional)

## Build process
~~~
cd tensor-lines/cpp
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
make install
~~~

## Components

### tensor_lines
Main program. Computes feature lines on a VTK Unstructured grid file.
Execute `tensor_lines -h` for valid command line options. Input file
needs to be in VTK legacy format with tensors as point data (arrays
with 9 components containing 3x3 tensor in row-major order).

The main algorithm is implemented in `src/TensorLines.cc` and does
not depend on VTK. A VTK filter using the algorithm to find intersections of feature lines with tetrahedral cell faces and connecting them to lines is implemented in
`src/vtkTensorLines.cc`.

### generate_tet_dataset
Small tool to generate example datasets of a linear tensor field. Execute
`generate_tet_dataset -h` for usage information. Generates a mesh in
tetrahedral form with variable number of subdivision levels. Tensors at the
four corners can be specified manually or randomly. Subdivision interpolates
linearly between the corners.

### generate_grid_dataset
Small tool to generate example datasets of several analytic tensor fields with
variable sampling density. Execute `generate_grid_dataset -h` for usage
information. Samples the analytic tensor field on a regular grid and
subdivides the cells into tetrahedra.
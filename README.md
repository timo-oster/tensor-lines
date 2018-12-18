# Find Parallel Eigenvector Lines in Tetrahedral Datasets

Implements an algorithm for finding lines of parallel eigenvectors in two
different piecewise linear tensor fields.

## Dependencies

* [Eigen 3](http://eigen.tuxfamily.org)
* [Boost](http://www.boost.org/)
* [VTK](http://www.vtk.org/)
* [Python 3](https://www.python.org/)
* [cpp_utils](https://github.com/timo-oster/cpp-utils)
* OpenMP(optional)

## Build process
~~~
cd parallel_eigenvectors/src
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
make install
~~~

## Components

### parallel_eigenvectors
Main program. Computes parallel eigenvectors on a VTK Unstructured grid file.
Execute `parallel_eigenvectors -h` for valid command line options. Input file
needs to be in VTK legacy format with two different PointData tensors (arrays
with 9 components containing 3x3 tensor in row-major order).

The main algorithm is implemented in `src/ParallelEigenvectors.cc` and does
not depend on VTK. A VTK filter using the algorithm to find PEV points on
tetrahedral cell faces and connecting them to lines is implemented in
`src/vtkParallelEigenvectors.cc`.

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
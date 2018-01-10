#include "utils.hh"

#include <boost/program_options.hpp>

#include <vtkCell.h>
#include <vtkCellIterator.h>
#include <vtkDoubleArray.h>
#include <vtkIdList.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkSmartPointer.h>
#include <vtkSubdivideTetra.h>
#include <vtkUnstructuredGrid.h>
#include <vtkUnstructuredGridWriter.h>

#include <iostream>
#include <random>
#include <string>
#include <stdexcept>

using namespace cpp_utils;

namespace po = boost::program_options;

using Mat3d = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;

Mat3d inputMatrix(const std::string& name)
{
    std::cout << "Enter values for 3x3 matrix " << name << std::endl;
    auto result = Mat3d{};
    std::cin >> result(0, 0) >> result(0, 1) >> result(0, 2) >> result(1, 0)
             >> result(1, 1) >> result(1, 2) >> result(2, 0) >> result(2, 1)
             >> result(2, 2);
    return result;
}


template <typename R, typename G>
Mat3d randMatrix(R& rnd, G& gen, bool symmetric = false)
{
    auto result = Mat3d{};
    result << rnd(gen), rnd(gen), rnd(gen),
              rnd(gen), rnd(gen), rnd(gen),
              rnd(gen), rnd(gen), rnd(gen);
    if(symmetric)
    {
        result = 0.5 * (result + result.transpose().eval());
    }
    return result;
}


int main(int argc, char const* argv[])
{
    using cpp_utils::range;
    using cpp_utils::make_string;

    auto random_seed = uint32_t{42};
    auto num_subdivisions = int32_t{8};
    auto out_name = std::string{"Grid.vtk"};
    auto symmetric = false;
    auto interactive = false;
    auto gen_derivatives = false;

    try
    {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "produce help message")
            ("random,r",
                po::value<uint32_t>(&random_seed),
                "Random seed for tensor generation")
            ("symmetric,s",
                po::bool_switch(&symmetric),
                "Generate symmetric tensor. Only works in conjunction with "
                "--random")
            ("interactive,i",
                po::bool_switch(&interactive),
                "Interactive mode (read matrices from stdin)")
            ("derivatives,d",
                po::bool_switch(&gen_derivatives),
                "Also generate derivatives Sx, Sy, and Sz")
            ("subdivision-level,l",
                po::value<int32_t>(&num_subdivisions)
                        ->required()
                        ->default_value(num_subdivisions),
                "Number of times the tetrahedron is subdivided")
            ("output,o",
                po::value<std::string>(&out_name)
                        ->required()->default_value(out_name),
                "Name of the output file");

        auto vm = po::variables_map{};
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if(vm.empty() || vm.count("help"))
        {
            std::cout << desc << "\n";
            return 0;
        }

        po::notify(vm);

        if(vm.count("random") > 0 && interactive)
        {
            throw po::error(
                    "Conflicting options --random and --interactive");
        }
        if(symmetric && interactive)
        {
            throw po::error(
                    "Conflicting options --symmetric and --interactive");
        }
    }
    catch(std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    catch(...)
    {
        std::cerr << "Exception of unknown type!\n";
    }

    auto gen = std::mt19937{random_seed};
    auto rnd = std::uniform_real_distribution<>{-1.0, 1.0};

    // Build vtkUnstructuredGrid with tetrahedral cells
    auto grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
    auto points = vtkSmartPointer<vtkPoints>::New();
    points->InsertNextPoint( 1.01979867,  1.00974785,  0.96975252);
    points->InsertNextPoint(-1.01979867,  0.99015215, -0.98975218);
    points->InsertNextPoint( 0.97980134, -1.01014782, -1.00974785);
    points->InsertNextPoint(-0.97980134, -0.98975218,  1.02974752);
    grid->SetPoints(points);

    auto cell = vtkSmartPointer<vtkIdList>::New();
    cell->InsertNextId(0);
    cell->InsertNextId(1);
    cell->InsertNextId(2);
    cell->InsertNextId(3);
    grid->InsertNextCell(VTK_TETRA, cell);

    if(!gen_derivatives)
    {
        auto s_field = vtkSmartPointer<vtkDoubleArray>::New();
        s_field->SetName("S");
        s_field->SetNumberOfComponents(9);
        s_field->SetNumberOfTuples(points->GetNumberOfPoints());
        auto t_field = vtkSmartPointer<vtkDoubleArray>::New();
        t_field->SetName("T");
        t_field->SetNumberOfComponents(9);
        t_field->SetNumberOfTuples(points->GetNumberOfPoints());

        for(auto i : range(points->GetNumberOfPoints()))
        {
            if(interactive)
            {
                s_field->SetTuple(
                        i, inputMatrix(make_string() << "S" << (i + 1)).data());
                t_field->SetTuple(
                        i, inputMatrix(make_string() << "T" << (i + 1)).data());
            }
            else
            {
                s_field->SetTuple(i, randMatrix(rnd, gen, symmetric).data());
                t_field->SetTuple(i, randMatrix(rnd, gen, symmetric).data());
            }
        }
        grid->GetPointData()->SetTensors(s_field);
        grid->GetPointData()->AddArray(t_field);
    }
    else
    {
        auto sx_field = vtkSmartPointer<vtkDoubleArray>::New();
        sx_field->SetName("Sx");
        sx_field->SetNumberOfComponents(9);
        sx_field->SetNumberOfTuples(points->GetNumberOfPoints());
        auto sy_field = vtkSmartPointer<vtkDoubleArray>::New();
        sy_field->SetName("Sy");
        sy_field->SetNumberOfComponents(9);
        sy_field->SetNumberOfTuples(points->GetNumberOfPoints());
        auto sz_field = vtkSmartPointer<vtkDoubleArray>::New();
        sz_field->SetName("Sz");
        sz_field->SetNumberOfComponents(9);
        sz_field->SetNumberOfTuples(points->GetNumberOfPoints());
        auto s_field = vtkSmartPointer<vtkDoubleArray>::New();
        s_field->SetName("S");
        s_field->SetNumberOfComponents(9);
        s_field->SetNumberOfTuples(points->GetNumberOfPoints());

        for(auto i : range(points->GetNumberOfPoints()))
        {
            if(interactive)
            {
                s_field->SetTuple(
                        i, inputMatrix(make_string() << "S" << (i + 1)).data());
                sx_field->SetTuple(
                        i, inputMatrix(make_string() << "Sx" << (i + 1)).data());
                sy_field->SetTuple(
                        i, inputMatrix(make_string() << "Sy" << (i + 1)).data());
                sz_field->SetTuple(
                        i, inputMatrix(make_string() << "Sz" << (i + 1)).data());
            }
            else
            {
                s_field->SetTuple(i, randMatrix(rnd, gen, symmetric).data());
                sx_field->SetTuple(i, randMatrix(rnd, gen, false).data());
                sy_field->SetTuple(i, randMatrix(rnd, gen, false).data());
                sz_field->SetTuple(i, randMatrix(rnd, gen, false).data());
            }
        }
        grid->GetPointData()->AddArray(sx_field);
        grid->GetPointData()->AddArray(sy_field);
        grid->GetPointData()->AddArray(sz_field);
        grid->GetPointData()->AddArray(s_field);
    }

    auto sub_filter = vtkSmartPointer<vtkSubdivideTetra>::New();

    for(auto _ : range(num_subdivisions))
    {
        sub_filter->SetInputData(grid);
        auto new_grid = vtkSmartPointer<vtkUnstructuredGrid>(
                vtkUnstructuredGrid::SafeDownCast(
                        sub_filter->GetOutputDataObject(0)));
        sub_filter->Update();
        grid->DeepCopy(new_grid);
    }

    auto inwriter = vtkSmartPointer<vtkUnstructuredGridWriter>::New();
    inwriter->SetInputData(grid);
    inwriter->SetFileName(out_name.c_str());
    inwriter->SetFileTypeToBinary();
    inwriter->Write();

    return 0;
}

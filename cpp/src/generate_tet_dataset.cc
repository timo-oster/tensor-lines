#include "utils.hh"

#include <boost/program_options.hpp>

#include <iostream>
#include <random>
#include <string>

#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkIdList.h>
#include <vtkCell.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkDoubleArray.h>
#include <vtkUnstructuredGridWriter.h>
#include <vtkSubdivideTetra.h>

#include <vtkCellIterator.h>

namespace po = boost::program_options;

template<typename R, typename G>
peigv::mat3d rand_matrix(R& rnd, G& gen)
{
    auto result = peigv::mat3d{};
    result << rnd(gen), rnd(gen), rnd(gen),
              rnd(gen), rnd(gen), rnd(gen),
              rnd(gen), rnd(gen), rnd(gen);
    return result;
}

int main(int argc, char const *argv[])
{
    using namespace peigv;

    auto random_seed = uint32_t{42};
    auto num_subdivisions = uint32_t{8};
    auto out_name = std::string{"Grid.vtk"};

    try
    {
        po::options_description desc("Allowed options");
        desc.add_options()
        ("help,h", "produce help message")
        ("seed,s",
            po::value<uint32_t>()->required()->default_value(random_seed),
            "Random seed for tensor generation")
        ("subdivision-level,l",
            po::value<uint32_t>()->required()->default_value(num_subdivisions),
            "Number of times the tetrahedron is subdivided")
        ("output,o",
            po::value<std::string>()->required()->default_value(out_name),
            "Name of the output file")
        ;

        auto vm = po::variables_map{};
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if(vm.empty() || vm.count("help"))
        {
            std::cout << desc << "\n";
            return 0;
        }

        po::notify(vm);

        random_seed = vm["seed"].as<uint32_t>();
        num_subdivisions = vm["subdivision-level"].as<uint32_t>();
        out_name = vm["output"].as<std::string>();

    }
    catch (std::exception& e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch (...)
    {
        std::cerr << "Exception of unknown type!\n";
    }

    auto gen = std::mt19937{random_seed};
    auto rnd = std::uniform_real_distribution<>{-1.0, 1.0};

    // Build vtkUnstructuredGrid with tetrahedral cells
    auto grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
    auto points = vtkSmartPointer<vtkPoints>::New();
    points->InsertNextPoint(1., 1., 1.);
    points->InsertNextPoint(-1., 1., -1.);
    points->InsertNextPoint(1., -1., -1.);
    points->InsertNextPoint(-1., -1., 1.);
    grid->SetPoints(points);

    auto cell = vtkSmartPointer<vtkIdList>::New();
    cell->InsertNextId(0);
    cell->InsertNextId(1);
    cell->InsertNextId(2);
    cell->InsertNextId(3);
    grid->InsertNextCell(VTK_TETRA, cell);

    auto s_field = vtkSmartPointer<vtkDoubleArray>::New();
    s_field->SetName("S");
    s_field->SetNumberOfComponents(9);
    s_field->SetNumberOfTuples(points->GetNumberOfPoints());
    auto t_field = vtkSmartPointer<vtkDoubleArray>::New();
    t_field->SetName("T");
    t_field->SetNumberOfComponents(9);
    t_field->SetNumberOfTuples(points->GetNumberOfPoints());
    for(auto i: range(points->GetNumberOfPoints()))
    {
        s_field->SetTuple(i, rand_matrix(rnd, gen).data());
        t_field->SetTuple(i, rand_matrix(rnd, gen).data());
    }
    grid->GetPointData()->SetTensors(s_field);
    grid->GetPointData()->AddArray(t_field);

    auto sub_filter = vtkSmartPointer<vtkSubdivideTetra>::New();
    sub_filter->SetInputData(grid);

    auto new_grid = vtkSmartPointer<vtkUnstructuredGrid>(
            vtkUnstructuredGrid::SafeDownCast(
                sub_filter->GetOutputDataObject(0)));

    sub_filter->Update();

    for(auto i: range(num_subdivisions-1))
    {
        grid->DeepCopy(new_grid);
        sub_filter->SetInputData(grid);
        new_grid = vtkSmartPointer<vtkUnstructuredGrid>(
            vtkUnstructuredGrid::SafeDownCast(
                sub_filter->GetOutputDataObject(0)));
        sub_filter->Update();
    }

    auto inwriter = vtkSmartPointer<vtkUnstructuredGridWriter>::New();
    inwriter->SetInputData(new_grid);
    inwriter->SetFileName(out_name.c_str());
    inwriter->Write();

    return 0;
}

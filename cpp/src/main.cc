#include "utils.hh"
#include "ParallelEigenvectors.hh"
#include "vtkParallelEigenvectors.hh"

#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPolyData.h>
#include <vtkIdList.h>
#include <vtkCell.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkDoubleArray.h>
#include <vtkUnstructuredGridReader.h>
#include <vtkPolyDataWriter.h>

#include <boost/program_options.hpp>

#include <iostream>
#include <string>

namespace po = boost::program_options;

int main(int argc, char const *argv[])
{
    using namespace peigv;

    auto input_file = std::string{};
    auto spatial_epsilon = 1e-3;
    auto direction_epsilon = 1e-3;

    try
    {
        auto desc = po::options_description("Allowed options");
        desc.add_options()
        ("help,h", "produce help message")
        ("spatial-epsilon,e", po::value<double>()->required()->default_value(1e-3),
            "epsilon for spatial subdivision")
        ("direction-epsilon,d", po::value<double>()->required()->default_value(1e-3),
            "epsilon for directional subdivision")
        ("input-file,i", po::value<std::string>()->required(),
            "name of the input file (VTK format)")
        ;

        auto podesc = po::positional_options_description{};
        podesc.add("input-file", 1);

        auto vm = po::variables_map{};
        po::store(po::command_line_parser(argc, argv)
                  .options(desc).positional(podesc).run(),
                  vm);

        if(vm.empty() || vm.count("help"))
        {
            std::cout << desc << "\n";
            return 0;
        }

        po::notify(vm);

        input_file = vm["input-file"].as<std::string>();
        spatial_epsilon = vm["spatial-epsilon"].as<double>();
        direction_epsilon = vm["direction-epsilon"].as<double>();
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

    auto reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(input_file.c_str());

    auto vtkpeigv = vtkSmartPointer<vtkParallelEigenvectors>::New();
    vtkpeigv->SetSpatialEpsilon(spatial_epsilon);
    vtkpeigv->SetDirectionEpsilon(direction_epsilon);

    vtkpeigv->SetInputConnection(0, reader->GetOutputPort(0));
    vtkpeigv->SetInputArrayToProcess(
            0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS, "S");
    vtkpeigv->SetInputArrayToProcess(
            1, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS, "T");

    auto outwriter = vtkSmartPointer<vtkPolyDataWriter>::New();
    outwriter->SetInputConnection(0, vtkpeigv->GetOutputPort(0));
    outwriter->SetFileName("Parallel_Eigenvectors.vtk");
    outwriter->Update();
    outwriter->Write();

    return 0;
}

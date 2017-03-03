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
#include <vtkUnstructuredGridWriter.h>
#include <vtkPolyDataWriter.h>
#include <vtkCallbackCommand.h>
#include <vtkCommand.h>

#include <boost/program_options.hpp>

#include <iostream>
#include <string>

namespace po = boost::program_options;

void ProgressFunction(vtkObject* caller,
                      long unsigned int vtkNotUsed(eventId),
                      void* vtkNotUsed(clientData),
                      void* vtkNotUsed(callData) )
{
  auto* filter = static_cast<vtkParallelEigenvectors*>(caller);
  std::cout << "Progress: " << std::setprecision(2) << (filter->GetProgress()*100) << "%    \r";
}

int main(int argc, char const *argv[])
{
    using namespace peigv;

    auto input_file = std::string{};
    auto spatial_epsilon = 1e-3;
    auto direction_epsilon = 1e-3;
    auto cluster_epsilon = 1e-3;
    auto parallelity_epsilon = 1e-3;
    auto out_name = std::string{"Parallel_Eigenvectors.vtk"};

    try
    {
        auto desc = po::options_description("Allowed options");
        desc.add_options()
        ("help,h", "produce help message")
        ("spatial-epsilon,e", po::value<double>(&spatial_epsilon)->required()->default_value(spatial_epsilon),
            "epsilon for spatial subdivision")
        ("direction-epsilon,d", po::value<double>(&direction_epsilon)->required()->default_value(direction_epsilon),
            "epsilon for directional subdivision")
        ("cluster-epsilon,c", po::value<double>(&cluster_epsilon)->required()->default_value(cluster_epsilon),
            "epsilon for clustering")
        ("parallelity-epsilon,p", po::value<double>(&parallelity_epsilon)->required()->default_value(parallelity_epsilon),
            "epsilon for eigenvector parallelity")
        ("input-file,i", po::value<std::string>(&input_file)->required(),
            "name of the input file (VTK format)")
        ("output,o",
            po::value<std::string>(&out_name)->required()->default_value(out_name),
            "Name of the output file")
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

    auto progressCallback = vtkSmartPointer<vtkCallbackCommand>::New();
    progressCallback->SetCallback(ProgressFunction);

    auto vtkpeigv = vtkSmartPointer<vtkParallelEigenvectors>::New();
    vtkpeigv->SetSpatialEpsilon(spatial_epsilon);
    vtkpeigv->SetDirectionEpsilon(direction_epsilon);
    vtkpeigv->AddObserver(vtkCommand::ProgressEvent, progressCallback);

    vtkpeigv->SetInputConnection(0, reader->GetOutputPort(0));
    vtkpeigv->SetInputArrayToProcess(
            0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS, "S");
    vtkpeigv->SetInputArrayToProcess(
            1, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS, "T");

    auto outwriter = vtkSmartPointer<vtkPolyDataWriter>::New();
    outwriter->SetInputConnection(0, vtkpeigv->GetOutputPort(0));
    outwriter->SetFileName(out_name.c_str());
    outwriter->SetFileTypeToBinary();
    outwriter->Update();
    outwriter->Write();

    auto out2writer = vtkSmartPointer<vtkUnstructuredGridWriter>::New();
    out2writer->SetInputConnection(0, vtkpeigv->GetOutputPort(1));
    out2writer->SetFileName("Point_Cells.vtk");
    out2writer->SetFileTypeToBinary();
    out2writer->Update();
    out2writer->Write();

    return 0;
}

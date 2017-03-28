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
#include <vtkCleanPolyData.h>
#include <vtkStripper.h>
#include <vtkUnstructuredGridReader.h>
#include <vtkUnstructuredGridWriter.h>
#include <vtkPolyDataWriter.h>
#include <vtkCallbackCommand.h>
#include <vtkCommand.h>

#include <boost/program_options.hpp>

#include <iostream>
#include <string>

#ifdef __linux__
#include <thread>
#include <signal.h>

bool term = false;

void terminate(int signum)
{
    term = true;
}
#endif // __linux__

namespace po = boost::program_options;

void ProgressFunction(vtkObject* caller,
                      long unsigned int vtkNotUsed(eventId),
                      void* vtkNotUsed(clientData),
                      void* vtkNotUsed(callData) )
{
  auto* filter = static_cast<vtkParallelEigenvectors*>(caller);
  std::cout << "Progress: " << std::fixed <<  std::setprecision(4)
            << (filter->GetProgress()*100) << "%    \r";
}

int main(int argc, char const *argv[])
{
    using namespace pev;

    auto input_file = std::string{};
    auto spatial_epsilon = 1e-3;
    auto direction_epsilon = 1e-3;
    auto cluster_epsilon = 1e-3;
    auto parallelity_epsilon = 1e-3;
    auto min_tensor_norm = 1e-3;
    auto out_name = std::string{"Parallel_Eigenvectors.vtk"};
    auto s_field_name = std::string{"S"};
    auto t_field_name = std::string{"T"};

    try
    {
        auto desc = po::options_description("Allowed options");
        desc.add_options()
            ("help,h", "produce help message")
            ("spatial-epsilon,e",
                po::value<double>(&spatial_epsilon)
                    ->required()->default_value(spatial_epsilon),
                "epsilon for spatial subdivision")
            ("direction-epsilon,d",
                po::value<double>(&direction_epsilon)
                    ->required()->default_value(direction_epsilon),
                "epsilon for directional subdivision")
            ("cluster-epsilon,c",
                po::value<double>(&cluster_epsilon)
                    ->required()->default_value(cluster_epsilon),
                "epsilon for clustering")
            ("parallelity-epsilon,p",
                po::value<double>(&parallelity_epsilon)
                    ->required()->default_value(parallelity_epsilon),
                "epsilon for eigenvector parallelity")
            ("min-tensor-norm,m",
             po::value<double>(&min_tensor_norm)
                ->required()->default_value(min_tensor_norm),
                "minimum norm of tensors necessary for a cell to be considered")
            ("input-file,i",
                po::value<std::string>(&input_file)->required(),
                "name of the input file (VTK format)")
            ("s-field-name,s",
                po::value<std::string>(&s_field_name)
                    ->required()->default_value(s_field_name),
                "name of the first input tensor field")
            ("t_field_name,t",
                po::value<std::string>(&t_field_name)
                    ->required()->default_value(t_field_name),
                "name of the second input tensor field")
            ("output,o",
                po::value<std::string>(&out_name)
                    ->required()->default_value(out_name),
                "Name of the output file");

        auto podesc = po::positional_options_description{};
        podesc.add("input-file", 1);

        auto vm = po::variables_map{};
        po::store(
                po::command_line_parser(argc, argv)
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

#ifdef __linux__
    // Set up signal handling for early termination
    struct sigaction new_action;
    struct sigaction old_action;
    new_action.sa_handler = terminate;
    sigemptyset (&new_action.sa_mask);
    new_action.sa_flags = 0;

    sigaction(SIGINT, nullptr, &old_action);
    if(old_action.sa_handler != SIG_IGN)
    {
        sigaction(SIGINT, &new_action, nullptr);
    }
#endif // __linux__

    auto reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(input_file.c_str());

    auto progressCallback = vtkSmartPointer<vtkCallbackCommand>::New();
    progressCallback->SetCallback(ProgressFunction);

    auto vtkpev = vtkSmartPointer<vtkParallelEigenvectors>::New();
    vtkpev->SetSpatialEpsilon(spatial_epsilon);
    vtkpev->SetDirectionEpsilon(direction_epsilon);
    vtkpev->SetClusterEpsilon(cluster_epsilon);
    vtkpev->SetParallelityEpsilon(parallelity_epsilon);
    vtkpev->SetMinTensorNorm(min_tensor_norm);
    vtkpev->AddObserver(vtkCommand::ProgressEvent, progressCallback);

    vtkpev->SetInputConnection(0, reader->GetOutputPort(0));
    vtkpev->SetInputArrayToProcess(
            0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS,
            s_field_name.c_str());
    vtkpev->SetInputArrayToProcess(
            1, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS,
            t_field_name.c_str());

#ifdef __linux__
    // Set up thread to check for program termination and set AbortExecute
    auto check_terminate = std::thread([&vtkpev]()
    {
        while(!term && vtkpev->GetProgress() < 1.)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if(term) vtkpev->AbortExecuteOn();
    });
#endif // __linux__

    // Merge duplicate/close points
    auto cleaner = vtkSmartPointer<vtkCleanPolyData>::New();
    cleaner->SetTolerance(cluster_epsilon);
    cleaner->ConvertLinesToPointsOn();
    cleaner->PointMergingOn();
    cleaner->SetInputConnection(0, vtkpev->GetOutputPort(0));

    // Merge line segments to longer line strips
    auto stripper = vtkSmartPointer<vtkStripper>::New();
    stripper->JoinContiguousSegmentsOn();
    stripper->SetMaximumLength(10000);
    stripper->SetInputConnection(0, cleaner->GetOutputPort(0));

    auto outwriter = vtkSmartPointer<vtkPolyDataWriter>::New();
    outwriter->SetInputConnection(0, vtkpev->GetOutputPort(0));
    outwriter->SetFileName(out_name.c_str());
    outwriter->SetFileTypeToBinary();
    outwriter->Update();
    outwriter->Write();

#ifdef __linux__
    check_terminate.join();
#endif // __linux__

    return 0;
}

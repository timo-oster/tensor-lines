#include "ParallelEigenvectors.hh"
#include "utils.hh"
#include "vtkParallelEigenvectors.h"

#include <vtkCallbackCommand.h>
#include <vtkCell.h>
#include <vtkCleanPolyData.h>
#include <vtkCommand.h>
#include <vtkCountVertices.h>
#include <vtkDoubleArray.h>
#include <vtkIdList.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataWriter.h>
#include <vtkSmartPointer.h>
#include <vtkStripper.h>
#include <vtkUnstructuredGrid.h>
#include <vtkUnstructuredGridReader.h>
#include <vtkUnstructuredGridWriter.h>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

#include <Eigen/Geometry>

#include <iostream>
#include <string>

#ifdef __linux__
#include <signal.h>
#include <thread>


bool term = false;

void terminate(int /*signum*/)
{
    if(!term)
    {
        term = true;
    }
    else
    {
        std::exit(1);
    }
}
#endif // __linux__


namespace po = boost::program_options;

std::istream& operator>>(std::istream& in, vtkParallelEigenvectors::LineType& ltype)
{
    auto token = std::string{};
    in >> token;

    boost::to_upper(token);

    if(token == "PEV")
    {
        ltype = vtkParallelEigenvectors::LineType::ParallelEigenvectors;
    }
    else if(token == "TSH")
    {
        ltype = vtkParallelEigenvectors::LineType::TensorSujudiHaimes;
    }
    else if(token == "TOPO")
    {
        ltype = vtkParallelEigenvectors::LineType::TensorTopology;
    }
    else
    {
        throw po::validation_error(po::validation_error::invalid_option_value, "ltype", token);
    }

    return in;
}


std::ostream& operator<<(std::ostream& out, const vtkParallelEigenvectors::LineType& ftype)
{
    switch(ftype)
    {
        case vtkParallelEigenvectors::LineType::ParallelEigenvectors:
            out << "pev";
            break;
        case vtkParallelEigenvectors::LineType::TensorSujudiHaimes:
            out << "tsh";
            break;
        case vtkParallelEigenvectors::LineType::TensorTopology:
            out << "topo";
            break;
        default:
            assert(false);
    }
    return out;
}


void ProgressFunction(vtkObject* caller,
                      long unsigned int vtkNotUsed(eventId),
                      void* vtkNotUsed(clientData),
                      void* vtkNotUsed(callData))
{
    auto* filter = static_cast<vtkParallelEigenvectors*>(caller);
    std::cout << "Progress: " << std::fixed << std::setprecision(4)
              << (filter->GetProgress() * 100) << "%    \r";
}


int main(int argc, char const* argv[])
{
    using namespace pev;

    auto input_file = std::string{};
    auto tolerance = 1e-6;
    auto cluster_epsilon = 1e-3;
    auto max_candidates = std::size_t{1000};
    auto out_name = std::string{"Parallel_Eigenvectors_Lines.vtk"};
    auto out2_name = std::string{"Parallel_Eigenvectors_Lines_NLTris.vtk"};
    auto s_field_name = std::string{"S"};
    auto t_field_name = std::string{"T"};
    auto line_type = vtkParallelEigenvectors::ParallelEigenvectors;

    try
    {
        auto desc = po::options_description("Allowed options");
        desc.add_options()
            ("help,h", "produce help message")
            ("tolerance,e",
                po::value<double>(&tolerance)
                        ->required()->default_value(tolerance),
                "maximum deviation of the target functions from zero")
            ("cluster-epsilon,c",
                po::value<double>(&cluster_epsilon)
                        ->required()->default_value(cluster_epsilon),
                "epsilon for clustering")
            ("max-candidates,m",
                po::value<std::size_t>(&max_candidates)
                        ->required()->default_value(max_candidates),
                "Maximum number of candidate triangles on a face before "
                "breaking off and assuming a non-line structure")
            ("input-file,i",
                po::value<std::string>(&input_file)->required(),
                "name of the input file (VTK format)")
            ("s-field-name,s",
                po::value<std::string>(&s_field_name)
                    ->required()->default_value(s_field_name),
                "name of the first input tensor field")
            ("t-field-name,t",
                po::value<std::string>(&t_field_name)
                        ->required()->default_value(t_field_name),
                "name of the second input tensor field")
            ("output,o",
                po::value<std::string>(&out_name),
                "Name of the output file")
            ("output2",
                po::value<std::string>(&out2_name),
                "Name of the second output file containing faces that might "
                "contain non-line structures")
            ("line-type,l",
                po::value<vtkParallelEigenvectors::LineType>(&line_type),
                "Select the type of line to compute (Parallel Eigenvectors: pev, "
                "Tensor Sujudi Haimes: tsh, Tensor Topology: topo)");

        auto podesc = po::positional_options_description{};
        podesc.add("input-file", 1);

        auto vm = po::variables_map{};
        po::store(po::command_line_parser(argc, argv)
                          .options(desc)
                          .positional(podesc)
                          .run(),
                  vm);

        if(vm.empty() || vm.count("help"))
        {
            std::cout << desc << "\n";
            return 0;
        }
        po::notify(vm);

        if(line_type != vtkParallelEigenvectors::ParallelEigenvectors && !vm["t-field-name"].defaulted())
        {
            std::cout << "Warning: You are not using --line-type=pev. "
                         "--t-field-name will be ignored."
                      << std::endl;
        }
        if(vm.count("output") == 0)
        {
            auto lastindex = input_file.find_last_of(".");
            auto rawname = input_file.substr(0, lastindex);
            switch(line_type)
            {
                case vtkParallelEigenvectors::ParallelEigenvectors:
                    out_name = rawname + "_PEV";
                    break;
                case vtkParallelEigenvectors::TensorSujudiHaimes:
                    out_name = rawname + "_TSH";
                    break;
                case vtkParallelEigenvectors::TensorTopology:
                    out_name = rawname + "_Topo";
            }
            out_name = out_name + "Lines.vtk";
        }
        if(vm.count("output2") == 0)
        {
            auto lastindex = out_name.find_last_of(".");
            auto rawname = out_name.substr(0, lastindex);
            out2_name = rawname + "_NLTri.vtk";
        }
    }
    catch(std::exception& e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...)
    {
        std::cerr << "Exception of unknown type!\n";
    }

#ifdef __linux__
    // Set up signal handling for early termination
    struct sigaction new_action;
    struct sigaction old_action;
    new_action.sa_handler = terminate;
    sigemptyset(&new_action.sa_mask);
    new_action.sa_flags = 0;

    sigaction(SIGINT, nullptr, &old_action);
    if(old_action.sa_handler != SIG_IGN)
    {
        sigaction(SIGINT, &new_action, nullptr);
    }
#endif // __linux__

#ifndef NDEBUG
    std::cout << "Running in DEBUG mode" << std::endl;
#endif

    auto reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(input_file.c_str());

    auto progressCallback = vtkSmartPointer<vtkCallbackCommand>::New();
    progressCallback->SetCallback(ProgressFunction);

    auto vtkpev = vtkSmartPointer<vtkParallelEigenvectors>::New();
    vtkpev->SetTolerance(tolerance);
    vtkpev->SetClusterEpsilon(cluster_epsilon);
    vtkpev->SetMaxCandidates(max_candidates);
    vtkpev->SetLineType(line_type);
    vtkpev->AddObserver(vtkCommand::ProgressEvent, progressCallback);

    vtkpev->SetInputConnection(0, reader->GetOutputPort(0));
    if(line_type == vtkParallelEigenvectors::ParallelEigenvectors)
    {
        vtkpev->SetInputArrayToProcess(0,
                                       0,
                                       0,
                                       vtkDataObject::FIELD_ASSOCIATION_POINTS,
                                       s_field_name.c_str());
        vtkpev->SetInputArrayToProcess(1,
                                       0,
                                       0,
                                       vtkDataObject::FIELD_ASSOCIATION_POINTS,
                                       t_field_name.c_str());
    }
    else
    {
        vtkpev->SetInputArrayToProcess(0,
                                       0,
                                       0,
                                       vtkDataObject::FIELD_ASSOCIATION_POINTS,
                                       s_field_name.c_str());
    }

#ifdef __linux__
    // Set up thread to check for program termination and set AbortExecute
    auto check_terminate = std::thread([&vtkpev]() {
        while(!term && vtkpev->GetProgress() < 1.)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if(term)
        {
            std::cout << "Received interrupt. "
                      << "Requesting termination..." << std::endl;
            vtkpev->AbortExecuteOn();
        }
    });
#endif // __linux__

    // Merge line segments to longer line strips
    // auto stripper = vtkSmartPointer<vtkStripper>::New();
    // stripper->JoinContiguousSegmentsOn();
    // stripper->SetMaximumLength(10000);
    // stripper->SetInputConnection(0, vtkpev->GetOutputPort(0));
    // stripper->Update();

    // auto counter = vtkSmartPointer<vtkCountVertices>::New();
    // counter->SetOutputArrayName("Vertex Count");
    // counter->SetInputData(data);

    auto outwriter = vtkSmartPointer<vtkPolyDataWriter>::New();
    outwriter->SetInputConnection(vtkpev->GetOutputPort(0));
    outwriter->SetFileName(out_name.c_str());
    outwriter->SetFileTypeToBinary();
    outwriter->Update();
    outwriter->Write();

    auto out2writer = vtkSmartPointer<vtkPolyDataWriter>::New();
    outwriter->SetInputConnection(vtkpev->GetOutputPort(1));
    outwriter->SetFileName(out2_name.c_str());
    outwriter->SetFileTypeToBinary();
    outwriter->Update();
    outwriter->Write();

#ifdef __linux__
    check_terminate.join();
#endif // __linux__

    return 0;
}

#include "utils.hh"

#include <vtkSmartPointer.h>
#include <vtkCellArray.h>
#include <vtkIdList.h>
#include <vtkUnstructuredGrid.h>
#include <vtkUnstructuredGridWriter.h>
#include <vtkPointData.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkFloatArray.h>
#include <vtkDelimitedTextReader.h>
#include <vtkTable.h>

#include <boost/program_options.hpp>

#include <string>
#include <iostream>

using namespace cpp_utils;

namespace po = boost::program_options;

int main(int argc, char const *argv[])
{
    using namespace pev;
    auto node_file = std::string{};
    auto topo_file = std::string{};
    auto out_name = std::string{"Dataset.vtk"};

    try
    {
        auto desc = po::options_description("Allowed options");
        desc.add_options()
            ("help,h", "produce help message")
            ("node-file,n",
                po::value<std::string>(&node_file)->required(),
                "File containing the nodal data")
            ("topo-file,t",
                po::value<std::string>(&topo_file)->required(),
                "File containing the topology information")
            ("output,o",
                po::value<std::string>(&out_name)
                        ->required()->default_value(out_name),
                "Name of the output file (vtk format");

        auto podesc = po::positional_options_description{};
        podesc.add("node-file", 1);
        podesc.add("topo-file", 2);

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

    auto node_reader = vtkSmartPointer<vtkDelimitedTextReader>::New();
    node_reader->SetFileName(node_file.c_str());
    node_reader->DetectNumericColumnsOn();
    node_reader->SetHaveHeaders(true);
    node_reader->SetFieldDelimiterCharacters(",");
    node_reader->Update();

    auto node_data = vtkSmartPointer<vtkTable>(node_reader->GetOutput());

    if(node_data->GetNumberOfColumns() < 10)
    {
        std::cout << "Wrong number of columns read in nodal data! ("
                  << node_data->GetNumberOfColumns() << ")\n";
    }

    auto topo_reader = vtkSmartPointer<vtkDelimitedTextReader>::New();
    topo_reader->SetFileName(topo_file.c_str());
    topo_reader->DetectNumericColumnsOn();
    topo_reader->SetHaveHeaders(false);
    topo_reader->SetFieldDelimiterCharacters(" ");
    topo_reader->MergeConsecutiveDelimitersOn();
    topo_reader->Update();

    auto topo_data = vtkSmartPointer<vtkTable>(topo_reader->GetOutput());

    if(topo_data->GetNumberOfColumns() != 5)
    {
        std::cout << "Wrong number of columns read in topology! ("
                  << topo_data->GetNumberOfColumns() << ")\n";
    }

    auto dataset = vtkSmartPointer<vtkUnstructuredGrid>::New();
    dataset->Allocate(topo_data->GetNumberOfRows());
    dataset->SetPoints(vtkPoints::New());
    dataset->GetPoints()->SetNumberOfPoints(node_data->GetNumberOfRows());

    auto px = vtkSmartPointer<vtkDoubleArray>(vtkDoubleArray::SafeDownCast(
            node_data->GetColumnByName(" x location")));
    if(!px) std::cout << "px wrong type" << std::endl;
    auto py = vtkSmartPointer<vtkDoubleArray>(vtkDoubleArray::SafeDownCast(
            node_data->GetColumnByName(" y location")));
    if(!py) std::cout << "py wrong type" << std::endl;
    auto pz = vtkSmartPointer<vtkDoubleArray>(vtkDoubleArray::SafeDownCast(
            node_data->GetColumnByName(" z location")));
    if(!pz) std::cout << "pz wrong type" << std::endl;

    for(auto i: range(node_data->GetNumberOfRows()))
    {
        double p[] = {0., 0., 0.};
        p[0] = px->GetValue(i);
        p[1] = py->GetValue(i);
        p[2] = pz->GetValue(i);
        dataset->GetPoints()->SetPoint(i, p);
    }

    auto sigma = vtkSmartPointer<vtkDoubleArray>::New();
    sigma->SetName("sigma_full");
    sigma->SetNumberOfComponents(9);
    sigma->SetNumberOfTuples(node_data->GetNumberOfRows());
    sigma->CopyComponent(
            0,
            vtkDataArray::SafeDownCast(node_data->GetColumnByName(" stress_x")),
            0);
    sigma->CopyComponent(
            1,
            vtkDataArray::SafeDownCast(node_data->GetColumnByName(" stress_xy")),
            0);
    sigma->CopyComponent(
            2,
            vtkDataArray::SafeDownCast(node_data->GetColumnByName(" stress_xz")),
            0);
    sigma->CopyComponent(
            3,
            vtkDataArray::SafeDownCast(node_data->GetColumnByName(" stress_xy")),
            0);
    sigma->CopyComponent(
            4,
            vtkDataArray::SafeDownCast(node_data->GetColumnByName(" stress_y")),
            0);
    sigma->CopyComponent(
            5,
            vtkDataArray::SafeDownCast(node_data->GetColumnByName(" stress_yz")),
            0);
    sigma->CopyComponent(
            6,
            vtkDataArray::SafeDownCast(node_data->GetColumnByName(" stress_xz")),
            0);
    sigma->CopyComponent(
            7,
            vtkDataArray::SafeDownCast(node_data->GetColumnByName(" stress_yz")),
            0);
    sigma->CopyComponent(
            8,
            vtkDataArray::SafeDownCast(node_data->GetColumnByName(" stress_z")),
            0);
    dataset->GetPointData()->AddArray(sigma);

    auto displacement = vtkSmartPointer<vtkDoubleArray>::New();
    displacement->SetName("U");
    displacement->SetNumberOfComponents(3);
    displacement->SetNumberOfTuples(node_data->GetNumberOfRows());
    displacement->CopyComponent(
            0,
            vtkDataArray::SafeDownCast(node_data->GetColumnByName(" displacement_x")),
            0);
    displacement->CopyComponent(
            1,
            vtkDataArray::SafeDownCast(node_data->GetColumnByName(" displacement_y")),
            0);
    displacement->CopyComponent(
            2,
            vtkDataArray::SafeDownCast(node_data->GetColumnByName(" displacement_z")),
            0);
    dataset->GetPointData()->AddArray(displacement);

    auto i1 = vtkSmartPointer<vtkIntArray>(
            vtkIntArray::SafeDownCast(topo_data->GetColumn(1)));
    auto i2 = vtkSmartPointer<vtkIntArray>(
            vtkIntArray::SafeDownCast(topo_data->GetColumn(2)));
    auto i3 = vtkSmartPointer<vtkIntArray>(
            vtkIntArray::SafeDownCast(topo_data->GetColumn(3)));
    auto i4 = vtkSmartPointer<vtkIntArray>(
            vtkIntArray::SafeDownCast(topo_data->GetColumn(4)));

    for(auto i: range(topo_data->GetNumberOfRows()))
    {
        auto pids = vtkSmartPointer<vtkIdList>::New();
        pids->SetNumberOfIds(4);
        pids->SetId(0, i1->GetValue(i)-1);
        pids->SetId(1, i2->GetValue(i)-1);
        pids->SetId(2, i3->GetValue(i)-1);
        pids->SetId(3, i4->GetValue(i)-1);
        dataset->InsertNextCell(VTK_TETRA, pids);
    }

    auto writer = vtkSmartPointer<vtkUnstructuredGridWriter>::New();
    writer->SetFileName(out_name.c_str());
    writer->SetFileTypeToBinary();
    writer->SetInputDataObject(dataset);
    writer->Write();
}

#include "utils.hh"

#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyDataWriter.h>
#include <vtkCellArray.h>
#include <vtkIdList.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <vtkCleanPolyData.h>
#include <vtkStripper.h>

#include <boost/program_options.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <iostream>
#include <string>

using namespace cpp_utils;

namespace po = boost::program_options;

int main(int argc, char const *argv[])
{
    using namespace pev;
    auto input_file = std::string{};
    auto tolerance = 0.;
    auto out_name = std::string{"Parallel_Eigenvectors.vtk"};

    try
    {
        auto desc = po::options_description("Allowed options");
        desc.add_options()
            ("help,h", "produce help message")
            ("tolerance,e",
                po::value<double>(&tolerance)
                        ->required()->default_value(tolerance),
                "maximum average eigenvector<->line angle in degrees")
            ("input-file,i",
                po::value<std::string>(&input_file)->required(),
                "name of the input file (PolyData in VTK format)")
            ("output,o",
                po::value<std::string>(&out_name)
                        ->required()->default_value(out_name),
                "Name of the output file");

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

    auto reader = vtkSmartPointer<vtkPolyDataReader>::New();
    reader->SetFileName(input_file.c_str());
    reader->Update();

    auto data = vtkSmartPointer<vtkPolyData>(reader->GetOutput());
    auto lines = vtkSmartPointer<vtkCellArray>(data->GetLines());

    auto output = vtkSmartPointer<vtkPolyData>::New();
    output->Allocate();
    output->SetPoints(vtkPoints::New());
    output->GetPointData()->DeepCopy(data->GetPointData());
    output->GetPoints()->DeepCopy(data->GetPoints());

    lines->InitTraversal();
    auto points = vtkSmartPointer<vtkIdList>::New();

    auto eivec = vtkSmartPointer<vtkDataArray>(
            data->GetPointData()->GetArray("Eigenvector"));
    while(lines->GetNextCell(points))
    {
        using Vec3d = Eigen::Vector3d;
        if(points->GetNumberOfIds() < 3) continue;
        auto p0 = Vec3d{};
        auto p1 = Vec3d{};
        auto ev = Vec3d{};
        data->GetPoint(points->GetId(0), p0.data());
        data->GetPoint(points->GetId(1), p1.data());
        eivec->GetTuple(points->GetId(0), ev.data());

        if((p1-p0).normalized().cross(ev.normalized()).norm() < sin(tolerance/180*3.14))
        {
            auto newpts = vtkSmartPointer<vtkIdList>::New();
            newpts->SetNumberOfIds(2);
            newpts->SetId(0, points->GetId(0));
            newpts->SetId(1, points->GetId(1));
            output->InsertNextCell(VTK_LINE, newpts);
        }

        for(auto i: range(1, points->GetNumberOfIds()-1))
        {
            data->GetPoint(points->GetId(i-1), p0.data());
            data->GetPoint(points->GetId(i+1), p1.data());
            eivec->GetTuple(points->GetId(i), ev.data());
            if((p1-p0).normalized().cross(ev.normalized()).norm() < sin(tolerance/180*3.14))
            {
                auto newpts = vtkSmartPointer<vtkIdList>::New();
                newpts->SetNumberOfIds(2);
                newpts->SetId(0, points->GetId(i));
                newpts->SetId(1, points->GetId(i+1));
                output->InsertNextCell(VTK_LINE, newpts);
            }
        }
    }

    auto cleaner = vtkSmartPointer<vtkCleanPolyData>::New();
    cleaner->PointMergingOff();
    cleaner->SetInputDataObject(output);

    auto stripper = vtkSmartPointer<vtkStripper>::New();
    stripper->JoinContiguousSegmentsOn();
    stripper->SetMaximumLength(10000);
    stripper->SetInputConnection(cleaner->GetOutputPort());
    stripper->Update();

    auto stripped = vtkSmartPointer<vtkPolyData>(stripper->GetOutput());
    auto stripped_lines = vtkSmartPointer<vtkCellArray>(stripped->GetLines());

    stripped->BuildLinks();
    stripped_lines->InitTraversal();
    auto id = vtkIdType{0};
    while(stripped_lines->GetNextCell(points))
    {
        if(points->GetNumberOfIds() < 4)
        {
            stripped->DeleteCell(id);
        }
        ++id;
    }
    stripped->RemoveDeletedCells();

    auto writer = vtkSmartPointer<vtkPolyDataWriter>::New();
    writer->SetInputDataObject(stripped);
    writer->SetFileName(out_name.c_str());
    writer->SetFileTypeToBinary();
    writer->Update();
    writer->Write();
    return 0;
}

#include "utils.hh"

#include "TensorField.hh"

#include <vtkSmartPointer.h>
#include <vtkDoubleArray.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkDataSetTriangleFilter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkUnstructuredGridWriter.h>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

#include <iostream>
#include <string>
#include <memory>

namespace po = boost::program_options;

using namespace pev;

enum class FieldType
{
    TEST
};

std::unique_ptr<TensorField> makeTensorField(FieldType ftype)
{
    switch(ftype)
    {
        case FieldType::TEST:
            return std::unique_ptr<TestField>{new TestField{}};
        default:
            return nullptr;
    }
}

std::istream& operator>>(std::istream& in, FieldType& ftype)
{
    auto token = std::string{};
    in >> token;

    boost::to_upper(token);

    if(token == "TEST")
    {
        ftype = FieldType::TEST;
    }
    else
    {
        throw po::validation_error(po::validation_error::invalid_option_value, "ftype", token);
    }

    return in;
}

std::ostream& operator<<(std::ostream& out, const FieldType& ftype)
{
    switch(ftype)
    {
        case FieldType::TEST:
            out << "test";
            break;
        default:
            assert(false);
    }
    return out;
}

int main(int argc, char const* argv[])
{
    using namespace pev;

    auto ftype = FieldType::TEST;
    auto minx = 0.;
    auto maxx = 0.;
    auto miny = 0.;
    auto maxy = 0.;
    auto minz = 0.;
    auto maxz = 0.;
    auto np = 0u;
    auto out_name = std::string{"Grid.vtk"};

    try
    {
        po::options_description desc("Allowed options");
        desc.add_options()("help,h", "produce help message")(
                "num-points,n",
                po::value(&np)->required(),
                "Number of grid points in the largest extent of the box")(
                "min-x",
                po::value(&minx)->required(),
                "Lower bound in x direction")("max-x",
                                              po::value(&maxx)->required(),
                                              "Upper bound in x direction")(
                "min-y",
                po::value(&miny)->required(),
                "Lower bound in y direction")("max-y",
                                              po::value(&maxy)->required(),
                                              "Upper bound in y direction")(
                "min-z",
                po::value(&minz)->required(),
                "Lower bound in z direction")("max-z",
                                              po::value(&maxz)->required(),
                                              "Upper bound in z direction")(
                "ftype,f",
                po::value(&ftype)->required()->default_value(ftype),
                "Type of tensor field to generate")(
                "output,o",
                po::value<std::string>(&out_name)->required()->default_value(
                        out_name),
                "Name of the output file");

        auto vm = po::variables_map{};
        po::store(po::parse_command_line(argc, argv, desc), vm);

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

    auto tfield = makeTensorField(ftype);

    auto maxdim = std::max({maxx-minx, maxy-miny, maxz-minz});
    auto h = maxdim/(np-1);

    auto nx = int(std::ceil((maxx-minx)/h)) + 1;
    auto ny = int(std::ceil((maxy-miny)/h)) + 1;
    auto nz = int(std::ceil((maxz-minz)/h)) + 1;

    auto grid = vtkSmartPointer<vtkImageData>::New();
    grid->SetDimensions(nx, ny, nz);
    grid->SetOrigin(minx, miny, minz);
    grid->SetSpacing((maxx - minx) / (nx - 1),
                     (maxy - miny) / (ny - 1),
                     (maxz - minz) / (nz - 1));

    auto s_data = vtkSmartPointer<vtkDoubleArray>::New();
    s_data->SetName("S");
    s_data->SetNumberOfComponents(9);
    s_data->SetNumberOfTuples(grid->GetNumberOfPoints());
    grid->GetPointData()->AddArray(s_data);

    s_data->Print(std::cout);

    auto sx_data = vtkSmartPointer<vtkDoubleArray>::New();
    sx_data->SetName("Sx");
    sx_data->SetNumberOfComponents(9);
    sx_data->SetNumberOfTuples(grid->GetNumberOfPoints());
    grid->GetPointData()->AddArray(sx_data);

    sx_data->Print(std::cout);

    auto sy_data = vtkSmartPointer<vtkDoubleArray>::New();
    sy_data->SetName("Sy");
    sy_data->SetNumberOfComponents(9);
    sy_data->SetNumberOfTuples(grid->GetNumberOfPoints());
    grid->GetPointData()->AddArray(sy_data);

    sy_data->Print(std::cout);

    auto sz_data = vtkSmartPointer<vtkDoubleArray>::New();
    sz_data->SetName("Sz");
    sz_data->SetNumberOfComponents(9);
    sz_data->SetNumberOfTuples(grid->GetNumberOfPoints());
    grid->GetPointData()->AddArray(sz_data);

    sz_data->Print(std::cout);

    for(auto i: range(grid->GetNumberOfPoints()))
    {
        using MapV3d = Eigen::Map<Vec3d>;
        auto pos = Vec3d{MapV3d{grid->GetPoint(i)}};
        auto s = tfield->t(pos);
        auto sx = tfield->tx(pos);
        auto sy = tfield->ty(pos);
        auto sz = tfield->tz(pos);
        s_data->SetTuple(i, s.data());
        sx_data->SetTuple(i, sx.data());
        sy_data->SetTuple(i, sy.data());
        sz_data->SetTuple(i, sz.data());
    }

    grid->Print(std::cout);

    auto tri_filt = vtkSmartPointer<vtkDataSetTriangleFilter>::New();
    tri_filt->Print(std::cout);
    tri_filt->SetInputData(grid);


    auto writer = vtkSmartPointer<vtkUnstructuredGridWriter>::New();
    writer->SetInputConnection(0, tri_filt->GetOutputPort());
    writer->SetFileName(out_name.c_str());
    writer->SetFileTypeToBinary();
    writer->Update();
    writer->Write();
}

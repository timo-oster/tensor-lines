#include "vtkParallelEigenvectors.hh"

#include "ParallelEigenvectors.hh"

#include <vtkPolyData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkDataSetAttributes.h>
#include <vtkSmartPointer.h>
#include <vtkCellIterator.h>
#include <vtkTetra.h>
#include <vtkGenericCell.h>
#include <vtkCellArray.h>
#include <vtkMergePoints.h>

#include <vtkCommand.h>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkObjectFactory.h>
#include <vtkStreamingDemandDrivenPipeline.h>

#include <iostream>

vtkStandardNewMacro(vtkParallelEigenvectors);

vtkParallelEigenvectors::vtkParallelEigenvectors()
{
  this->SetNumberOfInputPorts(1);
  this->SetNumberOfOutputPorts(1);
  this->SetInputArrayToProcess(0, 0, 0,
                               vtkDataObject::FIELD_ASSOCIATION_POINTS,
                               vtkDataSetAttributes::TENSORS);
  this->SetInputArrayToProcess(1, 0, 0,
                               vtkDataObject::FIELD_ASSOCIATION_POINTS,
                               vtkDataSetAttributes::TENSORS);
}


vtkParallelEigenvectors::~vtkParallelEigenvectors()
{
}


// void vtkParallelEigenvectors::PrintSelf(ostream& os, vtkIndent indent)
// {
//     this->Superclass::PrintSelf(os, indent);
// }


vtkPolyData* vtkParallelEigenvectors::GetOutput()
{
    return this->GetOutput(0);
}


vtkPolyData* vtkParallelEigenvectors::GetOutput(int port)
{
    return vtkPolyData::SafeDownCast(this->GetOutputDataObject(port));
}

int vtkParallelEigenvectors::ProcessRequest(
        vtkInformation* request,
        vtkInformationVector** inputVector,
        vtkInformationVector* outputVector)
{
  // Create an output object of the correct type.
    if(request->Has(vtkDemandDrivenPipeline::REQUEST_DATA_OBJECT()))
    {
        return this->RequestDataObject(request, inputVector, outputVector);
    }
    // generate the data
    if(request->Has(vtkDemandDrivenPipeline::REQUEST_DATA()))
    {
        return this->RequestData(request, inputVector, outputVector);
    }

    if(request->Has(vtkStreamingDemandDrivenPipeline::REQUEST_UPDATE_EXTENT()))
    {
        return this->RequestUpdateExtent(request, inputVector, outputVector);
    }

    // execute information
    if(request->Has(vtkDemandDrivenPipeline::REQUEST_INFORMATION()))
    {
        return this->RequestInformation(request, inputVector, outputVector);
    }

    return this->Superclass::ProcessRequest(request, inputVector, outputVector);
}

int vtkParallelEigenvectors::FillOutputPortInformation(
    int vtkNotUsed(port), vtkInformation* info)
{
    // now add our info
    info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkPolyData");
    return 1;
}


int vtkParallelEigenvectors::FillInputPortInformation(
        int vtkNotUsed(port), vtkInformation* info)
{
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkUnstructuredGrid");
    return 1;
}


int vtkParallelEigenvectors::RequestDataObject(
        vtkInformation* vtkNotUsed(request),
        vtkInformationVector** vtkNotUsed(inputVector),
        vtkInformationVector* outputVector )
{
    // RequestDataObject (RDO) is an earlier pipeline pass. During RDO, each
    // filter is supposed to produce an empty data object of the proper type

    auto* outInfo = outputVector->GetInformationObject(0);
    auto* output = vtkPolyData::SafeDownCast(
            outInfo->Get(vtkDataObject::DATA_OBJECT()));

    if(!output)
    {
        output = vtkPolyData::New();
        outInfo->Set(vtkDataObject::DATA_OBJECT(), output);
        output->FastDelete();

        this->GetOutputPortInformation(0)->Set(
                vtkDataObject::DATA_EXTENT_TYPE(), output->GetExtentType());
    }

    return 1;
}


int vtkParallelEigenvectors::RequestInformation(
        vtkInformation* vtkNotUsed(request),
        vtkInformationVector** vtkNotUsed(inputVector),
        vtkInformationVector* vtkNotUsed(outputVector))
{
    return 1;
}


int vtkParallelEigenvectors::RequestUpdateExtent(
    vtkInformation* vtkNotUsed(request),
    vtkInformationVector** inputVector,
    vtkInformationVector* vtkNotUsed(outputVector))
{
    auto numInputPorts = this->GetNumberOfInputPorts();
    for(auto i = 0; i < numInputPorts; i++)
    {
        auto numInputConnections = this->GetNumberOfInputConnections(i);
        for(auto j = 0; j < numInputConnections; j++)
        {
            auto* inputInfo = inputVector[i]->GetInformationObject(j);
            inputInfo->Set(vtkStreamingDemandDrivenPipeline::EXACT_EXTENT(), 1);
        }
    }
    return 1;
}


int vtkParallelEigenvectors::RequestData(
    vtkInformation* vtkNotUsed(request),
    vtkInformationVector **inputVector,
    vtkInformationVector* outputVector )
{
    using namespace peigv;

    // Later on RequestData (RD) happens. During RD each filter examines any
    // inputs it has, then fills in that empty data object with real data.

    auto* outInfo = outputVector->GetInformationObject(0);
    auto* output = vtkPolyData::SafeDownCast(
            outInfo->Get( vtkDataObject::DATA_OBJECT() ) );

    auto* inInfo = inputVector[0]->GetInformationObject(0);
    auto* input = vtkUnstructuredGrid::SafeDownCast(
            inInfo->Get(vtkDataObject::DATA_OBJECT()));

    // Get Cells/Faces of input data set
    // Get PointData of input data set
    // Get the two DataArrays corresponding to the tensor data
    auto* array1 = this->GetInputArrayToProcess(0, inputVector);
    auto* array2 = this->GetInputArrayToProcess(1, inputVector);

    // Check if the data arrays have the correct number of components
    if(array1->GetNumberOfComponents() != 9
       || array2->GetNumberOfComponents() != 9)
    {
        vtkErrorMacro(<<"both input arrays must be tensors with 9 components.");
        return 0;
    }

    if(this->GetInputArrayInformation(0)->Get(vtkDataObject::FIELD_ASSOCIATION())
       != vtkDataObject::FIELD_ASSOCIATION_POINTS
       || this->GetInputArrayInformation(1)->Get(vtkDataObject::FIELD_ASSOCIATION())
            != vtkDataObject::FIELD_ASSOCIATION_POINTS)
    {
        vtkErrorMacro(<<"both input arrays must be point data.");
        return 0;
    }

    output->SetPoints(vtkPoints::New());
    output->SetVerts(vtkCellArray::New());

    // Filter for automatically merging identical points
    auto locator = vtkSmartPointer<vtkMergePoints>::New();
    locator->InitPointInsertion(output->GetPoints(), input->GetBounds());

    auto it = vtkSmartPointer<vtkCellIterator>(input->NewCellIterator());
    auto current_cell = 0;
    for(it->InitTraversal(); !it->IsDoneWithTraversal(); it->GoToNextCell())
    {
        if(it->GetCellType() != VTK_TETRA)
        {
            continue;
        }

        auto* cell_points = it->GetPoints();
        auto* point_ids = it->GetPointIds();
        assert(it->GetNumberOfPoints() == 4);

        // todo: compute face normal and only continue if it faces in a
        // certain direction

        using mat3dm = Eigen::Map<peigv::mat3d>;
        using vec3dm = Eigen::Map<peigv::vec3d>;

        auto add_peigv_points = [&](int i1, int i2, int i3)
        {
            auto s1 = mat3d{mat3dm{array1->GetTuple(point_ids->GetId(i1))}};
            auto s2 = mat3d{mat3dm{array1->GetTuple(point_ids->GetId(i2))}};
            auto s3 = mat3d{mat3dm{array1->GetTuple(point_ids->GetId(i3))}};
            auto t1 = mat3d{mat3dm{array2->GetTuple(point_ids->GetId(i1))}};
            auto t2 = mat3d{mat3dm{array2->GetTuple(point_ids->GetId(i2))}};
            auto t3 = mat3d{mat3dm{array2->GetTuple(point_ids->GetId(i3))}};
            auto p1 = vec3d{vec3dm{cell_points->GetPoint(i1)}};
            auto p2 = vec3d{vec3dm{cell_points->GetPoint(i2)}};
            auto p3 = vec3d{vec3dm{cell_points->GetPoint(i3)}};

            auto points = peigv::findParallelEigenvectors(
                    s1, s2, s3, t1, t2, t3, p1, p2, p3,
                    this->_spatial_epsilon, this->_direction_epsilon);
            for(const auto& p: points)
            {
                auto pid = locator->InsertNextPoint(p.data());
                output->InsertNextCell(VTK_VERTEX, 1, &pid);
            }
        };

        add_peigv_points(0, 1, 2);
        add_peigv_points(1, 3, 2);
        add_peigv_points(0, 3, 1);
        add_peigv_points(0, 2, 3);

        // todo: link points to faces somehow so that later line connection per
        // tet is possible

        ++current_cell;
        this->UpdateProgress(double(current_cell)
                             / double(input->GetNumberOfCells()));
    }

    return 1;
}

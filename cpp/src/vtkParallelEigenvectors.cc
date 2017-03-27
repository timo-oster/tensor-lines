#include "vtkParallelEigenvectors.hh"

#include "ParallelEigenvectors.hh"

#include <Eigen/Geometry>

#include <vtkPolyData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkDataSetAttributes.h>
#include <vtkSmartPointer.h>
#include <vtkCellIterator.h>
#include <vtkTetra.h>
#include <vtkGenericCell.h>
#include <vtkCellArray.h>
#include <vtkMergePoints.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkIdList.h>

#include <vtkCommand.h>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkObjectFactory.h>
#include <vtkStreamingDemandDrivenPipeline.h>

#include <map>
#include <vector>
#include <list>
#include <unordered_set>
#include <iostream>
#include <future>
#include <chrono>

#include <ctpl_stl.h>

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
    int port, vtkInformation* info)
{
    // now add our info
    if(port == 0)
    {
        info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkPolyData");
    }
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

    using namespace std::chrono;
    using hours = duration<double, std::chrono::hours::period>;
    using minutes = duration<double, std::chrono::minutes::period>;
    using seconds = duration<double, std::chrono::seconds::period>;
    using milliseconds = duration<double, std::chrono::milliseconds::period>;

    // vtk Tensors are stored in row major order by convention
    using Mat3d = Eigen::Matrix<double, 3, 3, Eigen::RowMajor, 3, 3>;
    using Mat3dm = Eigen::Map<Mat3d>;
    using Vec3dm = Eigen::Map<peigv::Vec3d>;

    auto* outInfo = outputVector->GetInformationObject(0);
    auto* output = vtkPolyData::SafeDownCast(
            outInfo->Get(vtkDataObject::DATA_OBJECT()));

    auto* inInfo = inputVector[0]->GetInformationObject(0);
    auto* input = vtkUnstructuredGrid::SafeDownCast(
            inInfo->Get(vtkDataObject::DATA_OBJECT()));

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

    // Point and CellArrays for output dataset
    output->SetPoints(vtkPoints::New());
    output->SetVerts(vtkCellArray::New());

    // Output arrays for point information
    auto eig_rank1 = vtkSmartPointer<vtkDoubleArray>::New();
    eig_rank1->SetName("Rank1");
    output->GetPointData()->AddArray(eig_rank1);
    auto eig_rank2 = vtkSmartPointer<vtkDoubleArray>::New();
    eig_rank2->SetName("Rank2");
    output->GetPointData()->AddArray(eig_rank2);
    auto eival1 = vtkSmartPointer<vtkDoubleArray>::New();
    eival1->SetName("Eigenvalue 1");
    output->GetPointData()->AddArray(eival1);
    auto eival2 = vtkSmartPointer<vtkDoubleArray>::New();
    eival2->SetName("Eigenvalue 2");
    output->GetPointData()->AddArray(eival2);
    auto eivec = vtkSmartPointer<vtkDoubleArray>::New();
    eivec->SetName("Eigenvector");
    eivec->SetNumberOfComponents(3);
    output->GetPointData()->SetVectors(eivec);
    auto imag1 = vtkSmartPointer<vtkDoubleArray>::New();
    imag1->SetName("Imaginary 1");
    output->GetPointData()->AddArray(imag1);
    auto imag2 = vtkSmartPointer<vtkDoubleArray>::New();
    imag2->SetName("Imaginary 2");
    output->GetPointData()->AddArray(imag2);

    // Structure for mapping cell IDs to parallel eigenvector points
    auto cell_map = std::map<vtkIdType, vtkSmartPointer<vtkIdList>>{};

    ctpl::thread_pool tpool(4);
    auto futures = std::list<
                        std::pair<vtkIdType,
                                  std::future<peigv::PointList>
                        >
                    >{};

    auto find_pev_points = [](int /*tid*/,
                              const Mat3d& s1, const Mat3d& s2, const Mat3d& s3,
                              const Mat3d& t1, const Mat3d& t2, const Mat3d& t3,
                              const Vec3d& p1, const Vec3d& p2, const Vec3d& p3,
                              double spatial_epsilon, double direction_epsilon,
                              double cluster_epsilon, double parallelity_epsilon)
    {
        return peigv::findParallelEigenvectors(s1, s2, s3, t1, t2, t3, p1, p2, p3,
                                               spatial_epsilon, direction_epsilon,
                                               cluster_epsilon, parallelity_epsilon);
    };

    auto start = high_resolution_clock::now();

    auto progress = 0.;
    auto step = 1. / double(input->GetNumberOfCells()*4);

    auto it = vtkSmartPointer<vtkCellIterator>(input->NewCellIterator());
    for(it->InitTraversal(); !it->IsDoneWithTraversal(); it->GoToNextCell())
    {
        if(it->GetCellType() != VTK_TETRA)
        {
            continue;
        }

        auto* cell_points = it->GetPoints();
        auto* point_ids = it->GetPointIds();
        assert(it->GetNumberOfPoints() == 4);

        // Add parallel eigenvector points on the face with the given indices
        // to the output dataset
        auto add_peigv_points = [&](int i1, int i2, int i3)
        {
            auto p1 = Vec3d{Vec3dm{cell_points->GetPoint(i1)}};
            auto p2 = Vec3d{Vec3dm{cell_points->GetPoint(i2)}};
            auto p3 = Vec3d{Vec3dm{cell_points->GetPoint(i3)}};

            auto s1 = Mat3d{Mat3dm{array1->GetTuple(point_ids->GetId(i1))}};
            auto s2 = Mat3d{Mat3dm{array1->GetTuple(point_ids->GetId(i2))}};
            auto s3 = Mat3d{Mat3dm{array1->GetTuple(point_ids->GetId(i3))}};

            auto t1 = Mat3d{Mat3dm{array2->GetTuple(point_ids->GetId(i1))}};
            auto t2 = Mat3d{Mat3dm{array2->GetTuple(point_ids->GetId(i2))}};
            auto t3 = Mat3d{Mat3dm{array2->GetTuple(point_ids->GetId(i3))}};

            // Only process faces where all tensors are above the threshold
            if(s1.norm() < this->GetMinTensorNorm()
               || s2.norm() < this->GetMinTensorNorm()
               || s3.norm() < this->GetMinTensorNorm()
               || t1.norm() < this->GetMinTensorNorm()
               || t2.norm() < this->GetMinTensorNorm()
               || t3.norm() < this->GetMinTensorNorm())
            {
                progress += step;
                this->UpdateProgress(progress);
                return;
            }

            auto handle = tpool.push(
                    find_pev_points,
                    s1, s2, s3, t1, t2, t3, p1, p2, p3,
                    this->GetSpatialEpsilon(), this->GetDirectionEpsilon(),
                    this->GetClusterEpsilon(),  this->GetParallelityEpsilon());

            futures.push_back(std::make_pair(it->GetCellId(), std::move(handle)));
        };
        add_peigv_points(0, 1, 2);
        add_peigv_points(1, 3, 2);
        add_peigv_points(0, 3, 1);
        add_peigv_points(0, 2, 3);

        // Wait for some completed tasks here so the waiting list doesn't get
        // too long and clogs up memory
        while(futures.size() > 256)
        {
            for(auto it = std::begin(futures); it != std::end(futures); ++it)
            {
                auto& v = *it;
                if(!v.second.valid())
                {
                    std::cout << "Got an invalid future for cell " << v.first << std::endl;
                    continue;
                }
                if(v.second.wait_for(std::chrono::milliseconds{1}) != std::future_status::ready)
                {
                    continue;
                }
                auto cid = v.first;
                auto points = v.second.get();
                for(const auto& p: points)
                {
                    auto pid = output->GetPoints()->InsertNextPoint(p.pos.data());
                    eig_rank1->InsertValue(pid, double(p.s_rank));
                    eig_rank2->InsertValue(pid, double(p.t_rank));
                    eival1->InsertValue(pid, p.s_eival);
                    eival2->InsertValue(pid, p.t_eival);
                    eivec->InsertTuple(pid, p.eivec.data());
                    imag1->InsertValue(pid, p.s_has_imaginary ? 1. : 0.);
                    imag2->InsertValue(pid, p.t_has_imaginary ? 1. : 0.);
                    if(!cell_map[cid].Get())
                    {
                        cell_map[cid] = vtkSmartPointer<vtkIdList>::New();
                    }
                    cell_map[cid]->InsertNextId(pid);
                }
                progress += step;
                this->UpdateProgress(progress);
                it = futures.erase(it);
                --it;
            }
        }
    }

    // Wait for the remaining tasks to complete
    for(auto& v: futures)
    {
        if(!v.second.valid())
        {
            std::cout << "Got an invalid future for cell " << v.first << std::endl;
            continue;
        }
        auto cid = v.first;
        auto points = v.second.get();
        for(const auto& p: points)
        {
            auto pid = output->GetPoints()->InsertNextPoint(p.pos.data());
            eig_rank1->InsertValue(pid, double(p.s_rank));
            eig_rank2->InsertValue(pid, double(p.t_rank));
            eival1->InsertValue(pid, p.s_eival);
            eival2->InsertValue(pid, p.t_eival);
            eivec->InsertTuple(pid, p.eivec.data());
            imag1->InsertValue(pid, p.s_has_imaginary ? 1. : 0.);
            imag2->InsertValue(pid, p.t_has_imaginary ? 1. : 0.);
            if(!cell_map[cid].Get())
            {
                cell_map[cid] = vtkSmartPointer<vtkIdList>::New();
            }
            cell_map[cid]->InsertNextId(pid);
        }
        progress += step;
        this->UpdateProgress(progress);
    }

    auto end_pointsearch = high_resolution_clock::now();

    output->SetLines(vtkCellArray::New());

    auto cell_points = vtkSmartPointer<vtkIdList>::New();

    for(const auto& c: cell_map)
    {
        auto point_list = c.second;
        auto npoints = point_list->GetNumberOfIds();
        // For cells with exactly two parallel eigenvector points, connect them
        // with a line
        if(npoints == 2)
        {
            output->InsertNextCell(VTK_LINE, point_list);
        }
        else
        {
            // Match points by eigenvector direction

            using Matrix3X = Eigen::Matrix3Xd;
            using MatrixX = Eigen::MatrixXd;

            // Get eigenvectors of cell points
            auto eigdirs = Matrix3X(3, npoints).eval();
            for(auto i: range(npoints))
            {
                eigdirs.col(i) = Vec3dm{eivec->GetTuple(point_list->GetId(i))};
            }

            // Compute pairwise vector deviations
            auto dist = MatrixX::Ones(npoints, npoints).eval();
            for(auto i: range(npoints))
            {
                for(auto j: range(i + 1, npoints))
                {
                    dist(i, j) = eigdirs.col(i)
                                        .cross(eigdirs.col(j))
                                        .squaredNorm();
                }
            }

            // Greedily find closest two vectors and connect
            auto unlinked = std::unordered_set<vtkIdType>{};
            for(auto i: range(npoints))
            {
                unlinked.insert(i);
            }
            while(dist.sum() < npoints * npoints)
            {
                std::cout << dist << "\n" << std::endl;
                auto row = Matrix3X::Index{};
                auto col = Matrix3X::Index{};
                dist.minCoeff(&row, &col);
                std::cout << "Min row: " << row << ", min col: " << col << std::endl;
                auto line = vtkSmartPointer<vtkIdList>::New();
                line->SetNumberOfIds(2);
                line->InsertId(0, point_list->GetId(row));
                line->InsertId(1, point_list->GetId(col));
                output->InsertNextCell(VTK_LINE, line);
                dist.col(col).setOnes();
                dist.col(row).setOnes();
                dist.row(col).setOnes();
                dist.row(row).setOnes();
                unlinked.erase(row);
                unlinked.erase(col);
            }

            // Add vertex for last unlinked point if any
            for(auto i: unlinked)
            {
                output->InsertNextCell(VTK_VERTEX, 1, point_list->GetPointer(i));
            }
            std::cout << "Cell " << c.first
            << " has " << npoints << " points" << std::endl;
            input->GetCellPoints(c.first, cell_points);
            auto s1 = Mat3d{Mat3dm{array1->GetTuple(cell_points->GetId(0))}};
            auto s2 = Mat3d{Mat3dm{array1->GetTuple(cell_points->GetId(1))}};
            auto s3 = Mat3d{Mat3dm{array1->GetTuple(cell_points->GetId(2))}};
            auto s4 = Mat3d{Mat3dm{array1->GetTuple(cell_points->GetId(3))}};

            auto t1 = Mat3d{Mat3dm{array2->GetTuple(cell_points->GetId(0))}};
            auto t2 = Mat3d{Mat3dm{array2->GetTuple(cell_points->GetId(1))}};
            auto t3 = Mat3d{Mat3dm{array2->GetTuple(cell_points->GetId(2))}};
            auto t4 = Mat3d{Mat3dm{array2->GetTuple(cell_points->GetId(3))}};

            auto fmt = Eigen::IOFormat(Eigen::FullPrecision, 0,
                                       " ", "\n", "", "", "", "");

            std::cout << "S1: \n" << s1.format(fmt) << std::endl;
            std::cout << "T1: \n" << t1.format(fmt) << std::endl;
            std::cout << "S2: \n" << s2.format(fmt) << std::endl;
            std::cout << "T2: \n" << t2.format(fmt) << std::endl;
            std::cout << "S3: \n" << s3.format(fmt) << std::endl;
            std::cout << "T3: \n" << t3.format(fmt) << std::endl;
            std::cout << "S4: \n" << s4.format(fmt) << std::endl;
            std::cout << "T4: \n" << t4.format(fmt) << std::endl;
        }
    }

    auto end_all = high_resolution_clock::now();
    auto duration_all = seconds(end_all - start);
    auto duration_pointsearch = seconds(end_pointsearch - start);

    std::cout << "Processed dataset in "
              << hours(duration_all).count() << " hours" << std::endl;
    std::cout << "Point search time: "
              << hours(duration_pointsearch).count() << " hours" << std::endl;
    std::cout << "Postprocessing time: "
              << seconds(duration_all - duration_pointsearch).count()
              << " seconds" << std::endl;
    std::cout << "Number of faces processed: ~" << input->GetNumberOfCells()*4/2
              << " * 2" << std::endl;
    std::cout << "Average time per face: "
              << (milliseconds(duration_pointsearch) / (input->GetNumberOfCells()*4)).count()
              << " milliseconds" << std::endl;

    return 1;
}

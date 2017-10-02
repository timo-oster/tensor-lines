#include "vtkParallelEigenvectors.hh"

#include "ParallelEigenvectors.hh"

#include <Eigen/Geometry>

#include <vtkCellIterator.h>
#include <vtkDataArray.h>
#include <vtkDataSet.h>
#include <vtkDataSetAttributes.h>
#include <vtkDoubleArray.h>
#include <vtkIdList.h>
#include <vtkIntArray.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>

#include <vtkCommand.h>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkObjectFactory.h>
#include <vtkStreamingDemandDrivenPipeline.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <list>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace
{
// vtk Tensors are stored in row major order by convention
using Mat3d = Eigen::Matrix<double, 3, 3, Eigen::RowMajor, 3, 3>;
using Mat3dm = Eigen::Map<Mat3d>;
using Vec3d = pev::Vec3d;
using Vec3dm = Eigen::Map<Vec3d>;

struct TriFace
{
    std::array<vtkIdType, 3> points;

    friend bool operator==(const TriFace& t1, const TriFace& t2)
    {
        return t1.points == t2.points;
    }
};

struct FaceHash
{
    std::size_t offset;

    explicit FaceHash(std::size_t offset) : offset(offset)
    {
    }

    std::size_t operator()(const TriFace& face) const
    {
        auto pts = face.points;
        std::sort(std::begin(pts), std::end(pts));
        return std::hash<std::size_t>{}(
                std::size_t(pts[0]) + offset * std::size_t(pts[1])
                + offset * offset * std::size_t(pts[2]));
    }
};


using FaceMap = std::unordered_map<TriFace, std::vector<vtkIdType>, FaceHash>;

FaceMap buildFaceMap(vtkDataSet* dataset)
{
    auto face_map =
            FaceMap(std::size_t(dataset->GetNumberOfCells() * 4),
                    FaceHash(std::size_t(dataset->GetNumberOfPoints())));

    auto add_face = [&](vtkIdList* point_ids,
                        vtkIdType cell_id,
                        vtkIdType i1,
                        vtkIdType i2,
                        vtkIdType i3) {
        auto tri = TriFace{{point_ids->GetId(i1),
                            point_ids->GetId(i2),
                            point_ids->GetId(i3)}};

        auto f = face_map.find(tri);
        if(f != std::end(face_map))
        {
            f->second.push_back(cell_id);
        }
        else
        {
            face_map[tri] = std::vector<vtkIdType>{cell_id};
        }
    };

    // Collect unique faces and remember participating cells
    auto it = vtkSmartPointer<vtkCellIterator>(dataset->NewCellIterator());
    for(it->InitTraversal(); !it->IsDoneWithTraversal(); it->GoToNextCell())
    {
        if(it->GetCellType() != VTK_TETRA) continue;

        auto* point_ids = it->GetPointIds();
        auto cid = it->GetCellId();

        add_face(point_ids, cid, 0, 1, 2);
        add_face(point_ids, cid, 1, 3, 2);
        add_face(point_ids, cid, 0, 3, 1);
        add_face(point_ids, cid, 0, 2, 3);
    }
    return face_map;
}


std::vector<pev::PointList> computePEVPoints(const std::vector<TriFace>& faces,
                                             vtkPoints* points,
                                             vtkDataArray* array1,
                                             vtkDataArray* array2,
                                             vtkAlgorithm* progress_alg,
                                             const pev::PEVOptions& opts)
{
    const auto step = 1. / double(faces.size());
    progress_alg->UpdateProgress(0);
    auto results = std::vector<pev::PointList>(faces.size());
    auto terminate = false;
#pragma omp parallel for
    for(auto i = std::size_t{0}; i < faces.size(); ++i)
    {
#pragma omp flush(terminate)
        if(terminate) continue;

        auto face = faces[i];

        auto p1 = Vec3d{};
        points->GetPoint(face.points[0], p1.data());
        auto p2 = Vec3d{};
        points->GetPoint(face.points[1], p2.data());
        auto p3 = Vec3d{};
        points->GetPoint(face.points[2], p3.data());

        auto s1 = Mat3d{};
        array1->GetTuple(face.points[0], s1.data());
        auto s2 = Mat3d{};
        array1->GetTuple(face.points[1], s2.data());
        auto s3 = Mat3d{};
        array1->GetTuple(face.points[2], s3.data());

        auto t1 = Mat3d{};
        array2->GetTuple(face.points[0], t1.data());
        auto t2 = Mat3d{};
        array2->GetTuple(face.points[1], t2.data());
        auto t3 = Mat3d{};
        array2->GetTuple(face.points[2], t3.data());

        auto points = pev::findParallelEigenvectors(
                pev::TensorInterp{{s1, s2, s3}},
                pev::TensorInterp{{t1, t2, t3}},
                pev::Triangle{{p1, p2, p3}},
                opts);
        results[i] = points;
#pragma omp critical(progress)
        {
            progress_alg->UpdateProgress(progress_alg->GetProgress() + step);
            if(progress_alg->GetAbortExecute())
            {
                std::cout << "Terminate request accepted" << std::endl;
                terminate = true;
            }
        }
    }
    return results;
}

std::vector<pev::PointList> computeSHPoints(const std::vector<TriFace>& faces,
                                             vtkPoints* points,
                                             vtkDataArray* array1,
                                             vtkDataArray* array2,
                                             vtkDataArray* array3,
                                             vtkDataArray* array4,
                                             vtkAlgorithm* progress_alg,
                                             const pev::PEVOptions& opts)
{
    const auto step = 1. / double(faces.size());
    progress_alg->UpdateProgress(0);
    auto results = std::vector<pev::PointList>(faces.size());
    auto terminate = false;
#pragma omp parallel for
    for(auto i = std::size_t{0}; i < faces.size(); ++i)
    {
#pragma omp flush(terminate)
        if(terminate) continue;

        auto face = faces[i];

        auto p1 = Vec3d{};
        points->GetPoint(face.points[0], p1.data());
        auto p2 = Vec3d{};
        points->GetPoint(face.points[1], p2.data());
        auto p3 = Vec3d{};
        points->GetPoint(face.points[2], p3.data());

        auto s1 = Mat3d{};
        array1->GetTuple(face.points[0], s1.data());
        auto s2 = Mat3d{};
        array1->GetTuple(face.points[1], s2.data());
        auto s3 = Mat3d{};
        array1->GetTuple(face.points[2], s3.data());

        auto sx1 = Mat3d{};
        array2->GetTuple(face.points[0], sx1.data());
        auto sx2 = Mat3d{};
        array2->GetTuple(face.points[1], sx2.data());
        auto sx3 = Mat3d{};
        array2->GetTuple(face.points[2], sx3.data());

        auto sy1 = Mat3d{};
        array3->GetTuple(face.points[0], sy1.data());
        auto sy2 = Mat3d{};
        array3->GetTuple(face.points[1], sy2.data());
        auto sy3 = Mat3d{};
        array3->GetTuple(face.points[2], sy3.data());

        auto sz1 = Mat3d{};
        array4->GetTuple(face.points[0], sz1.data());
        auto sz2 = Mat3d{};
        array4->GetTuple(face.points[1], sz2.data());
        auto sz3 = Mat3d{};
        array4->GetTuple(face.points[2], sz3.data());

        auto points = pev::findTensorSujudiHaimes(
                pev::TensorInterp{{s1, s2, s3}},
                pev::TensorInterp{{sx1, sx2, sx3}},
                pev::TensorInterp{{sy1, sy2, sy3}},
                pev::TensorInterp{{sz1, sz2, sz3}},
                pev::Triangle{{p1, p2, p3}},
                opts);
        results[i] = points;
#pragma omp critical(progress)
        {
            progress_alg->UpdateProgress(progress_alg->GetProgress() + step);
            if(progress_alg->GetAbortExecute())
            {
                std::cout << "Terminate request accepted" << std::endl;
                terminate = true;
            }
        }
    }
    return results;
}
}


vtkStandardNewMacro(vtkParallelEigenvectors)


vtkParallelEigenvectors::vtkParallelEigenvectors()
{
    this->SetNumberOfInputPorts(1);
    this->SetNumberOfOutputPorts(1);
    this->SetInputArrayToProcess(0,
                                 0,
                                 0,
                                 vtkDataObject::FIELD_ASSOCIATION_POINTS,
                                 vtkDataSetAttributes::TENSORS);
    this->SetInputArrayToProcess(1,
                                 0,
                                 0,
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

int vtkParallelEigenvectors::ProcessRequest(vtkInformation* request,
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


int vtkParallelEigenvectors::FillOutputPortInformation(int port,
                                                       vtkInformation* info)
{
    // now add our info
    if(port == 0)
    {
        info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkPolyData");
    }
    return 1;
}


int vtkParallelEigenvectors::FillInputPortInformation(int vtkNotUsed(port),
                                                      vtkInformation* info)
{
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkUnstructuredGrid");
    return 1;
}


int vtkParallelEigenvectors::RequestDataObject(
        vtkInformation* vtkNotUsed(request),
        vtkInformationVector** vtkNotUsed(inputVector),
        vtkInformationVector* outputVector)
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


int vtkParallelEigenvectors::RequestData(vtkInformation* vtkNotUsed(request),
                                         vtkInformationVector** inputVector,
                                         vtkInformationVector* outputVector)
{
    using pev::range;
    using namespace std::chrono;
    using minutes = duration<double, std::chrono::minutes::period>;
    using seconds = duration<double, std::chrono::seconds::period>;
    using milliseconds = duration<double, std::chrono::milliseconds::period>;

    auto* outInfo = outputVector->GetInformationObject(0);
    auto* output = vtkPolyData::SafeDownCast(
            outInfo->Get(vtkDataObject::DATA_OBJECT()));

    auto* inInfo = inputVector[0]->GetInformationObject(0);
    auto* input = vtkUnstructuredGrid::SafeDownCast(
            inInfo->Get(vtkDataObject::DATA_OBJECT()));

    // Get the two DataArrays corresponding to the tensor data
    auto* array1 = this->GetInputArrayToProcess(0, inputVector);
    auto* array2 = this->GetInputArrayToProcess(1, inputVector);

    auto* array3 = this->GetInputArrayToProcess(2, inputVector);
    auto* array4 = this->GetInputArrayToProcess(3, inputVector);

    // Check if the data arrays have the correct number of components
    if(array1->GetNumberOfComponents() != 9
       || array2->GetNumberOfComponents() != 9
       || (_use_sujudi_haimes && (array3->GetNumberOfComponents() != 9
                                  || array4->GetNumberOfComponents() != 9)))
    {
        vtkErrorMacro(<< "all input arrays must be tensors with 9 components.");
        return 0;
    }

    if(this->GetInputArrayInformation(0)->Get(
               vtkDataObject::FIELD_ASSOCIATION())
               != vtkDataObject::FIELD_ASSOCIATION_POINTS
       || this->GetInputArrayInformation(1)->Get(
                  vtkDataObject::FIELD_ASSOCIATION())
                  != vtkDataObject::FIELD_ASSOCIATION_POINTS
       || (_use_sujudi_haimes
           && (this->GetInputArrayInformation(2)->Get(
                       vtkDataObject::FIELD_ASSOCIATION())
                       != vtkDataObject::FIELD_ASSOCIATION_POINTS
               || this->GetInputArrayInformation(3)->Get(
                          vtkDataObject::FIELD_ASSOCIATION())
                          != vtkDataObject::FIELD_ASSOCIATION_POINTS)))
    {
        vtkErrorMacro(<< "all input arrays must be point data.");
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
    auto csize = vtkSmartPointer<vtkIntArray>::New();
    csize->SetName("Cluster Size");
    output->GetPointData()->AddArray(csize);
    auto pos_unc = vtkSmartPointer<vtkDoubleArray>::New();
    pos_unc->SetName("Position Uncertainty");
    output->GetPointData()->AddArray(pos_unc);
    auto dir_unc = vtkSmartPointer<vtkDoubleArray>::New();
    dir_unc->SetName("Direction Uncertainty");
    output->GetPointData()->AddArray(dir_unc);

    // map faces to cell ids
    auto face_map = buildFaceMap(input);

    // Copy faces to array for parallel looping
    auto faces = std::vector<TriFace>{};
    faces.reserve(face_map.size());
    for(const auto& f : face_map)
    {
        faces.push_back(f.first);
    }

    auto start = high_resolution_clock::now();

    auto opts = pev::PEVOptions{this->GetSpatialEpsilon(),
                                this->GetDirectionEpsilon(),
                                this->GetClusterEpsilon()};

    auto fresults = std::vector<pev::PointList>{};

    if(_use_sujudi_haimes)
    {
        fresults = computeSHPoints(faces,
                                   input->GetPoints(),
                                   array1,
                                   array2,
                                   array3,
                                   array4,
                                   this,
                                   opts);
    }
    else
    {
        fresults = computePEVPoints(
                faces, input->GetPoints(), array1, array2, this, opts);
    }

    auto end_pointsearch = high_resolution_clock::now();

    // map cell IDs to parallel eigenvector points found on their faces
    auto cell_map = std::map<vtkIdType, vtkSmartPointer<vtkIdList>>{};

    for(auto i : range(faces.size()))
    {
        auto pev_points = fresults[i];
        auto cids = face_map[faces[i]];
        for(const auto& p : pev_points)
        {
            auto pid = output->GetPoints()->InsertNextPoint(p.pos.data());
            eig_rank1->InsertValue(pid, double(p.s_rank));
            eig_rank2->InsertValue(pid, double(p.t_rank));
            eival1->InsertValue(pid, p.s_eival);
            eival2->InsertValue(pid, p.t_eival);
            eivec->InsertTuple(pid, p.eivec.data());
            imag1->InsertValue(pid, p.s_has_imaginary ? 1. : 0.);
            imag2->InsertValue(pid, p.t_has_imaginary ? 1. : 0.);
            csize->InsertValue(pid, p.cluster_size);
            pos_unc->InsertValue(pid, p.pos_uncertainty);
            dir_unc->InsertValue(pid, p.dir_uncertainty);

            // Add pev point to all cells participating in this face
            for(auto c : cids)
            {
                if(!cell_map[c].Get())
                {
                    cell_map[c] = vtkSmartPointer<vtkIdList>::New();
                }
                cell_map[c]->InsertNextId(pid);
            }
        }
    }

    output->SetLines(vtkCellArray::New());

    auto cell_points = vtkSmartPointer<vtkIdList>::New();

    for(const auto& c : cell_map)
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
            for(auto i : range(npoints))
            {
                eigdirs.col(i) = Vec3dm{eivec->GetTuple(point_list->GetId(i))};
            }

            // Compute pairwise vector deviations
            auto dist = MatrixX::Ones(npoints, npoints).eval();
            for(auto i : range(npoints))
            {
                for(auto j : range(i + 1, npoints))
                {
                    dist(i, j) =
                            eigdirs.col(i).cross(eigdirs.col(j)).squaredNorm();
                }
            }

            // Greedily find closest two vectors and connect
            auto unlinked = std::unordered_set<vtkIdType>{};
            for(auto i : range(npoints))
            {
                unlinked.insert(i);
            }
            while(dist.sum() < npoints * npoints)
            {
                auto row = Matrix3X::Index{};
                auto col = Matrix3X::Index{};
                dist.minCoeff(&row, &col);
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
            for(auto i : unlinked)
            {
                output->InsertNextCell(
                        VTK_VERTEX, 1, point_list->GetPointer(i));
            }
        }
    }

    auto end_all = high_resolution_clock::now();
    auto duration_all = seconds(end_all - start);
    auto duration_pointsearch = seconds(end_pointsearch - start);

    std::cout << "Processed dataset in " << minutes(duration_all).count()
              << " minutes" << std::endl;
    std::cout << "Point search time: " << minutes(duration_pointsearch).count()
              << " minutes" << std::endl;
    std::cout << "Postprocessing time: "
              << seconds(duration_all - duration_pointsearch).count()
              << " seconds" << std::endl;
    std::cout << "Number of faces processed: " << face_map.size() << std::endl;
    std::cout << "Average time per face: "
              << (milliseconds(duration_pointsearch) / face_map.size()).count()
              << " milliseconds" << std::endl;

    this->UpdateProgress(1.);

    return 1;
}

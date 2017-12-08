#ifndef CPP_VTK_PARALLEL_EIGENVECTORS_HH
#define CPP_VTK_PARALLEL_EIGENVECTORS_HH

#include "vtkAlgorithm.h"

class vtkPolyData;

class VTK_EXPORT vtkParallelEigenvectors : public vtkAlgorithm
{
public:
    enum LineType
    {
        ParallelEigenvectors,
        TensorSujudiHaimes,
        TensorTopology
    };

    static vtkParallelEigenvectors* New();
    vtkTypeMacro(vtkParallelEigenvectors, vtkAlgorithm);

    vtkParallelEigenvectors(const vtkParallelEigenvectors&) = delete;
    void operator=(const vtkParallelEigenvectors&) = delete;

    double GetTolerance() const
    {
        return _tolerance;
    }
    void SetTolerance(double value)
    {
        _tolerance = value;
    }

    double GetClusterEpsilon() const
    {
        return _cluster_epsilon;
    }
    void SetClusterEpsilon(double value)
    {
        _cluster_epsilon = value;
    }

    std::size_t GetMaxCandidates() const
    {
        return _max_candidates;
    }
    void SetMaxCandidates(std::size_t& value)
    {
        _max_candidates = value;
    }

    LineType GetLineType() const
    {
        return _line_type;
    }
    void SetLineType(LineType lt)
    {
        _line_type = lt;
    }

    // Get the output data object for a port on this algorithm.
    vtkPolyData* GetOutput();
    vtkPolyData* GetOutput(int);

    // see vtkAlgorithm for details
    virtual int ProcessRequest(vtkInformation*,
                               vtkInformationVector**,
                               vtkInformationVector*) override;

protected:
    vtkParallelEigenvectors();
    ~vtkParallelEigenvectors();

    // This is called by the superclass.
    // This is the method you should override.
    virtual int RequestDataObject(vtkInformation* request,
                                  vtkInformationVector** inputVector,
                                  vtkInformationVector* outputVector);

    // convenience method
    virtual int RequestInformation(vtkInformation* request,
                                   vtkInformationVector** inputVector,
                                   vtkInformationVector* outputVector);

    // Description:
    // This is called by the superclass.
    // This is the method you should override.
    virtual int RequestData(vtkInformation* request,
                            vtkInformationVector** inputVector,
                            vtkInformationVector* outputVector);

    // Description:
    // This is called by the superclass.
    // This is the method you should override.
    virtual int RequestUpdateExtent(vtkInformation*,
                                    vtkInformationVector**,
                                    vtkInformationVector*);

    virtual int FillOutputPortInformation(int port,
                                          vtkInformation* info) override;
    virtual int FillInputPortInformation(int port,
                                         vtkInformation* info) override;

private:
    double _tolerance = 1e-6;
    double _cluster_epsilon = 1e-4;
    std::size_t _max_candidates = 100;
    LineType _line_type = LineType::ParallelEigenvectors;
};

#endif

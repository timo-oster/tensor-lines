#ifndef CPP_VTK_TENSOR_LINES_HH
#define CPP_VTK_TENSOR_LINES_HH

#include "vtkAlgorithm.h"

class vtkPolyData;

class VTK_EXPORT vtkTensorLines : public vtkAlgorithm
{
public:
    vtkTypeMacro(vtkTensorLines,vtkAlgorithm)
    enum LineType : int
    {
        TensorTopology = 0,
        TensorCoreLines = 1,
        ParallelEigenvectors = 2
    };

    static vtkTensorLines* New();

    vtkTensorLines(const vtkTensorLines&) = delete;
    void operator=(const vtkTensorLines&) = delete;

    double GetTolerance() const
    {
        return _tolerance;
    }
    void SetTolerance(double value)
    {
        _tolerance = value;
        this->Modified();
    }

    double GetClusterEpsilon() const
    {
        return _cluster_epsilon;
    }
    void SetClusterEpsilon(double value)
    {
        _cluster_epsilon = value;
        this->Modified();
    }

    std::size_t GetMaxCandidates() const
    {
        return _max_candidates;
    }
    void SetMaxCandidates(int value)
    {
        _max_candidates = value;
        this->Modified();
    }

    int GetLineType() const
    {
        return _line_type;
    }
    void SetLineType(int lt)
    {
        _line_type = LineType(lt);
        this->Modified();
    }

    // Get the output data object for a port on this algorithm.
    vtkPolyData* GetOutput();
    vtkPolyData* GetOutput(int);

    // see vtkAlgorithm for details
    virtual int ProcessRequest(vtkInformation*,
                               vtkInformationVector**,
                               vtkInformationVector*) override;

protected:
    vtkTensorLines();
    ~vtkTensorLines() VTK_OVERRIDE {}

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
                                          vtkInformation* info) VTK_OVERRIDE;
    virtual int FillInputPortInformation(int port,
                                         vtkInformation* info) VTK_OVERRIDE;

private:
    //BTX
    double _tolerance = 1e-6;
    double _cluster_epsilon = 1e-4;
    std::size_t _max_candidates = 100;
    LineType _line_type = LineType::TensorCoreLines;
    //ETX
};

#endif

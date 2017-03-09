#ifndef CPP_VTK_PARALLEL_EIGENVECTORS_HH
#define CPP_VTK_PARALLEL_EIGENVECTORS_HH

#include "vtkAlgorithm.h"

class vtkPolyData;

class VTK_EXPORT vtkParallelEigenvectors: public vtkAlgorithm
{
public:
    static vtkParallelEigenvectors* New();
    vtkTypeMacro(vtkParallelEigenvectors,vtkAlgorithm);

    vtkParallelEigenvectors(const vtkParallelEigenvectors&) = delete;
    void operator= (const vtkParallelEigenvectors&) = delete;

    double GetSpatialEpsilon()
    {
        return _spatial_epsilon;
    }
    void SetSpatialEpsilon(double value)
    {
        _spatial_epsilon = value;
    }

    double GetDirectionEpsilon()
    {
        return _direction_epsilon;
    }
    void SetDirectionEpsilon(double value)
    {
        _direction_epsilon = value;
    }

    double GetClusterEpsilon()
    {
        return _cluster_epsilon;
    }
    void SetClusterEpsilon(double value)
    {
        _cluster_epsilon = value;
    }

    double GetParallelityEpsilon()
    {
        return _parallelity_epsilon;
    }
    void SetParallelityEpsilon(double value)
    {
        _parallelity_epsilon = value;
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
                                  vtkInformationVector* outputVector );

    // convenience method
    virtual int RequestInformation(vtkInformation* request,
                                   vtkInformationVector** inputVector,
                                   vtkInformationVector* outputVector );

    // Description:
    // This is called by the superclass.
    // This is the method you should override.
    virtual int RequestData(vtkInformation* request,
                            vtkInformationVector** inputVector,
                            vtkInformationVector* outputVector );

    // Description:
    // This is called by the superclass.
    // This is the method you should override.
    virtual int RequestUpdateExtent(vtkInformation*,
                                    vtkInformationVector**,
                                    vtkInformationVector* );

    virtual int FillOutputPortInformation(int port, vtkInformation* info) override;
    virtual int FillInputPortInformation(int port, vtkInformation* info) override;

private:
    double _spatial_epsilon = 1e-6;
    double _direction_epsilon = 1e-6;
    double _cluster_epsilon = 1e-4;
    double _parallelity_epsilon = 1e-6;
} ;

#endif

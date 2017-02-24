#ifndef CPP_VTK_PARALLEL_EIGENVECTORS_HH
#define CPP_VTK_PARALLEL_EIGENVECTORS_HH

#include "vtkAlgorithm.h"

class vtkPolyData;

class VTK_EXPORT vtkParallelEigenvectors: public vtkAlgorithm
{
public:
    static vtkParallelEigenvectors* New();
    vtkTypeMacro(vtkParallelEigenvectors,vtkAlgorithm);
    // void PrintSelf(ostream& os, vtkIndent indent) override;

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

    // Description:
    // Get the output data object for a port on this algorithm.
    vtkPolyData* GetOutput();
    vtkPolyData* GetOutput(int);
    // virtual void SetOutput(vtkDataObject* d);

    // Description:
    // see vtkAlgorithm for details
    virtual int ProcessRequest(vtkInformation*,
                               vtkInformationVector**,
                               vtkInformationVector*) override;

    // this method is not recommended for use, but lots of old style filters use it
    // vtkDataObject* GetInput();
    // vtkDataObject* GetInput(int port);

    // Description:
    // Set an input of this algorithm. You should not override these
    // methods because they are not the only way to connect a pipeline.
    // Note that these methods support old-style pipeline connections.
    // When writing new code you should use the more general
    // vtkAlgorithm::SetInputConnection().  These methods transform the
    // input index to the input port index, not an index of a connection
    // within a single port.
    // void SetInput(vtkDataObject*);
    // void SetInput(int, vtkDataObject*);

    // Description:
    // Add an input of this algorithm.  Note that these methods support
    // old-style pipeline connections.  When writing new code you should
    // use the more general vtkAlgorithm::AddInputConnection().  See
    // SetInput() for details.
    // void AddInput(vtkDataObject* );
    // void AddInput(int, vtkDataObject* );

    protected:
    vtkParallelEigenvectors();
    ~vtkParallelEigenvectors();

    // Description:
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
    vtkParallelEigenvectors(const vtkParallelEigenvectors&); // Not implemented.
    void operator= (const vtkParallelEigenvectors&);  // Not implemented.

    double _spatial_epsilon = 1e-6;
    double _direction_epsilon = 1e-6;
} ;

#endif

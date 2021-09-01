# Taken from master since 4.8 does not have most of these convenience functions
# MRML-numpy
#
from slicer.ScriptedLoadableModule import *
import os
import logging
import slicer
import vtk
import numpy as np

def array(pattern = "", index = 0):
    """Return the array you are "most likely to want" from the indexth
    MRML node that matches the pattern.
    .. warning::
      Meant to be used in the python console for quick debugging/testing.
    More specific API should be used in scripts to be sure you get exactly
    what you want, such as :py:meth:`arrayFromVolume`, :py:meth:`arrayFromModelPoints`,
    and :py:meth:`arrayFromGridTransform`.
    """
    node = getNode(pattern=pattern, index=index)
    import slicer
    if isinstance(node, slicer.vtkMRMLVolumeNode):
        return arrayFromVolume(node)
    elif isinstance(node, slicer.vtkMRMLModelNode):
        return arrayFromModelPoints(node)
    elif isinstance(node, slicer.vtkMRMLGridTransformNode):
        return arrayFromGridTransform(node)
    elif isinstance(node, slicer.vtkMRMLMarkupsNode):
        return arrayFromMarkupsControlPoints(node)
    elif isinstance(node, slicer.vtkMRMLTransformNode):
        return arrayFromTransformMatrix(node)

    # TODO: accessors for other node types: polydata (verts, polys...), colors
    raise RuntimeError("Cannot get node "+node.GetID()+" as array")

def arrayFromVolume(volumeNode):
    """Return voxel array from volume node as numpy array.
    Voxels values are not copied. Voxel values in the volume node can be modified
    by changing values in the numpy array.
    After all modifications has been completed, call :py:meth:`arrayFromVolumeModified`.
    .. warning:: Memory area of the returned array is managed by VTK, therefore
      values in the array may be changed, but the array must not be reallocated
      (change array size, shallow-copy content from other array most likely causes
      application crash). To allow arbitrary numpy operations on a volume array:
        1. Make a deep-copy of the returned VTK-managed array using :func:`numpy.copy`.
        2. Perform any computations using the copied array.
        3. Write results back to the image data using :py:meth:`updateVolumeFromArray`.
    """
    scalarTypes = ['vtkMRMLScalarVolumeNode', 'vtkMRMLLabelMapVolumeNode']
    vectorTypes = ['vtkMRMLVectorVolumeNode', 'vtkMRMLMultiVolumeNode', 'vtkMRMLDiffusionWeightedVolumeNode']
    tensorTypes = ['vtkMRMLDiffusionTensorVolumeNode']
    vimage = volumeNode.GetImageData()
    nshape = tuple(reversed(volumeNode.GetImageData().GetDimensions()))
    import vtk.util.numpy_support
    narray = None
    if volumeNode.GetClassName() in scalarTypes:
        narray = vtk.util.numpy_support.vtk_to_numpy(vimage.GetPointData().GetScalars()).reshape(nshape)
    elif volumeNode.GetClassName() in vectorTypes:
        components = vimage.GetNumberOfScalarComponents()
        if components > 1:
            nshape = nshape + (components,)
        narray = vtk.util.numpy_support.vtk_to_numpy(vimage.GetPointData().GetScalars()).reshape(nshape)
    elif volumeNode.GetClassName() in tensorTypes:
        narray = vtk.util.numpy_support.vtk_to_numpy(vimage.GetPointData().GetTensors()).reshape(nshape+(3,3))
    else:
        raise RuntimeError("Unsupported volume type: "+volumeNode.GetClassName())
    return narray

def arrayFromVolumeModified(volumeNode):
    """Indicate that modification of a numpy array returned by :py:meth:`arrayFromVolume` has been completed."""
    imageData = volumeNode.GetImageData()
    pointData = imageData.GetPointData() if imageData else None
    if pointData:
        if pointData.GetScalars():
            pointData.GetScalars().Modified()
        if pointData.GetTensors():
            pointData.GetTensors().Modified()
    volumeNode.Modified()

def arrayFromModelPoints(modelNode):
    """Return point positions of a model node as numpy array.
    Point coordinates can be modified by modifying the numpy array.
    After all modifications has been completed, call :py:meth:`arrayFromModelPointsModified`.
    .. warning:: Important: memory area of the returned array is managed by VTK,
      therefore values in the array may be changed, but the array must not be reallocated.
      See :py:meth:`arrayFromVolume` for details.
    """
    import vtk.util.numpy_support
    if modelNode.GetPolyData() is None:
        return np.asarray(())
    pointData = modelNode.GetPolyData().GetPoints().GetData()
    narray = vtk.util.numpy_support.vtk_to_numpy(pointData)
    return narray

def arrayFromModelPointsModified(modelNode):
    """Indicate that modification of a numpy array returned by :py:meth:`arrayFromModelPoints` has been completed."""
    if modelNode.GetPolyData():
        modelNode.GetPolyData().GetPoints().GetData().Modified()
    # Trigger re-render
    modelNode.GetDisplayNode().Modified()

def _vtkArrayFromModelPointData(modelNode, arrayName):
    """Helper function for getting VTK point data array that throws exception
    with informative error message if the data array is not found.
    """
    pointData = modelNode.GetPolyData().GetPointData()
    if not pointData or pointData.GetNumberOfArrays() == 0:
        raise ValueError("Input modelNode does not contain point data")
    arrayVtk = pointData.GetArray(arrayName)
    if not arrayVtk:
        availableArrayNames = [pointData.GetArrayName(i) for i in range(pointData.GetNumberOfArrays())]
        raise ValueError("Input modelNode does not contain data array '{0}'. Available array names: '{1}'".format(
            arrayName, "', '".join(availableArrayNames)))
    return arrayVtk

def arrayFromModelPointData(modelNode, arrayName):
    """Return point data array of a model node as numpy array.
    .. warning:: Important: memory area of the returned array is managed by VTK,
      therefore values in the array may be changed, but the array must not be reallocated.
      See :py:meth:`arrayFromVolume` for details.
    """
    import vtk.util.numpy_support
    arrayVtk = _vtkArrayFromModelPointData(modelNode, arrayName)
    narray = vtk.util.numpy_support.vtk_to_numpy(arrayVtk)
    return narray

def arrayFromModelPointDataModified(modelNode, arrayName):
    """Indicate that modification of a numpy array returned by :py:meth:`arrayFromModelPointData` has been completed."""
    arrayVtk = _vtkArrayFromModelPointData(modelNode, arrayName)
    arrayVtk.Modified()

def arrayFromMarkupsControlPointData(markupsNode, arrayName):
    """Return control point data array of a markups node as numpy array.
    .. warning:: Important: memory area of the returned array is managed by VTK,
      therefore values in the array may be changed, but the array must not be reallocated.
      See :py:meth:`arrayFromVolume` for details.
    """
    import vtk.util.numpy_support
    for measurementIndex in range(markupsNode.GetNumberOfMeasurements()):
        measurement = markupsNode.GetNthMeasurement(measurementIndex)
        doubleArrayVtk = measurement.GetControlPointValues()
        if doubleArrayVtk and doubleArrayVtk.GetName() == arrayName:
            narray = vtk.util.numpy_support.vtk_to_numpy(doubleArrayVtk)
            return narray

def arrayFromMarkupsControlPointDataModified(markupsNode, arrayName):
    """Indicate that modification of a numpy array returned by :py:meth:`arrayFromMarkupsControlPointData` has been completed."""
    for measurementIndex in range(markupsNode.GetNumberOfMeasurements()):
        measurement = markupsNode.GetNthMeasurement(measurementIndex)
        doubleArrayVtk = measurement.GetControlPointValues()
        if doubleArrayVtk and doubleArrayVtk.GetName() == arrayName:
            doubleArrayVtk.Modified()

def arrayFromModelPolyIds(modelNode):
    """Return poly id array of a model node as numpy array.
    These ids are the following format:
    [ n(0), i(0,0), i(0,1), ... i(0,n(00),..., n(j), i(j,0), ... i(j,n(j))...]
    where n(j) is the number of vertices in polygon j
    and i(j,k) is the index into the vertex array for vertex k of poly j.
    As described here:
    https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
    Typically in Slicer n(j) will always be 3 because a model node's
    polygons will be triangles.
    .. warning:: Important: memory area of the returned array is managed by VTK,
      therefore values in the array may be changed, but the array must not be reallocated.
      See :py:meth:`arrayFromVolume` for details.
    """
    import vtk.util.numpy_support
    arrayVtk = modelNode.GetPolyData().GetPolys().GetData()
    narray = vtk.util.numpy_support.vtk_to_numpy(arrayVtk)
    return narray

def arrayFromGridTransform(gridTransformNode):
    """Return voxel array from transform node as numpy array.
    Vector values are not copied. Values in the transform node can be modified
    by changing values in the numpy array.
    After all modifications has been completed, call :py:meth:`arrayFromGridTransformModified`.
    .. warning:: Important: memory area of the returned array is managed by VTK,
      therefore values in the array may be changed, but the array must not be reallocated.
      See :py:meth:`arrayFromVolume` for details.
    """
    transformGrid = gridTransformNode.GetTransformFromParent()
    displacementGrid = transformGrid.GetDisplacementGrid()
    nshape = tuple(reversed(displacementGrid.GetDimensions()))
    import vtk.util.numpy_support
    nshape = nshape + (3,)
    narray = vtk.util.numpy_support.vtk_to_numpy(displacementGrid.GetPointData().GetScalars()).reshape(nshape)
    return narray

def arrayFromVTKMatrix(vmatrix):
    """Return vtkMatrix4x4 or vtkMatrix3x3 elements as numpy array.
    The returned array is just a copy and so any modification in the array will not affect the input matrix.
    To set VTK matrix from a numpy array, use :py:meth:`vtkMatrixFromArray` or
    :py:meth:`updateVTKMatrixFromArray`.
    """
    from vtk import vtkMatrix4x4
    from vtk import vtkMatrix3x3
    import numpy as np
    if isinstance(vmatrix, vtkMatrix4x4):
        matrixSize = 4
    elif isinstance(vmatrix, vtkMatrix3x3):
        matrixSize = 3
    else:
        raise RuntimeError("Input must be vtk.vtkMatrix3x3 or vtk.vtkMatrix4x4")
    narray = np.eye(matrixSize)
    vmatrix.DeepCopy(narray.ravel(), vmatrix)
    return narray

def vtkMatrixFromArray(narray):
    """Create VTK matrix from a 3x3 or 4x4 numpy array.
    :param narray: input numpy array
    The returned matrix is just a copy and so any modification in the array will not affect the output matrix.
    To set numpy array from VTK matrix, use :py:meth:`arrayFromVTKMatrix`.
    """
    from vtk import vtkMatrix4x4
    from vtk import vtkMatrix3x3
    narrayshape = narray.shape
    if narrayshape == (4,4):
        vmatrix = vtkMatrix4x4()
        updateVTKMatrixFromArray(vmatrix, narray)
        return vmatrix
    elif narrayshape == (3,3):
        vmatrix = vtkMatrix3x3()
        updateVTKMatrixFromArray(vmatrix, narray)
        return vmatrix
    else:
        raise RuntimeError("Unsupported numpy array shape: "+str(narrayshape)+" expected (4,4)")

def updateVTKMatrixFromArray(vmatrix, narray):
    """Update VTK matrix values from a numpy array.
    :param vmatrix: VTK matrix (vtkMatrix4x4 or vtkMatrix3x3) that will be update
    :param narray: input numpy array
    To set numpy array from VTK matrix, use :py:meth:`arrayFromVTKMatrix`.
    """
    from vtk import vtkMatrix4x4
    from vtk import vtkMatrix3x3
    if isinstance(vmatrix, vtkMatrix4x4):
        matrixSize = 4
    elif isinstance(vmatrix, vtkMatrix3x3):
        matrixSize = 3
    else:
        raise RuntimeError("Output vmatrix must be vtk.vtkMatrix3x3 or vtk.vtkMatrix4x4")
    if narray.shape != (matrixSize, matrixSize):
        raise RuntimeError("Input narray size must match output vmatrix size ({0}x{0})".format(matrixSize))
    vmatrix.DeepCopy(narray.ravel())

def arrayFromTransformMatrix(transformNode, toWorld=False):
    """Return 4x4 transformation matrix as numpy array.
    :param toWorld: if set to True then the transform to world coordinate system is returned
      (effect of parent transform to the node is applied), otherwise transform to parent transform is returned.
    The returned array is just a copy and so any modification in the array will not affect the transform node.
    To set transformation matrix from a numpy array, use :py:meth:`updateTransformMatrixFromArray`.
    """
    from vtk import vtkMatrix4x4
    vmatrix = vtkMatrix4x4()
    if toWorld:
        success = transformNode.GetMatrixTransformToWorld(vmatrix)
    else:
        success = transformNode.GetMatrixTransformToParent(vmatrix)
    if not success:
        raise RuntimeError("Failed to get transformation matrix from node "+transformNode.GetID())
    return arrayFromVTKMatrix(vmatrix)

def updateTransformMatrixFromArray(transformNode, narray, toWorld = False):
    """Set transformation matrix from a numpy array of size 4x4 (toParent).
    :param world: if set to True then the transform will be set so that transform
      to world matrix will be equal to narray; otherwise transform to parent will be
      set as narray.
    """
    import numpy as np
    from vtk import vtkMatrix4x4
    narrayshape = narray.shape
    if narrayshape != (4,4):
        raise RuntimeError("Unsupported numpy array shape: "+str(narrayshape)+" expected (4,4)")
    if toWorld and transformNode.GetParentTransformNode():
        # thisToParent = worldToParent * thisToWorld = inv(parentToWorld) * toWorld
        narrayParentToWorld = arrayFromTransformMatrix(transformNode.GetParentTransformNode())
        thisToParent = np.dot(np.linalg.inv(narrayParentToWorld), narray)
        updateTransformMatrixFromArray(transformNode, thisToParent, toWorld = False)
    else:
        vmatrix = vtkMatrix4x4()
        updateVTKMatrixFromArray(vmatrix, narray)
        transformNode.SetMatrixTransformToParent(vmatrix)

def arrayFromGridTransformModified(gridTransformNode):
    """Indicate that modification of a numpy array returned by :py:meth:`arrayFromGridTransform` has been completed."""
    transformGrid = gridTransformNode.GetTransformFromParent()
    displacementGrid = transformGrid.GetDisplacementGrid()
    displacementGrid.GetPointData().GetScalars().Modified()
    displacementGrid.Modified()

def arrayFromSegment(segmentationNode, segmentId):
    """
    """
    import logging
    logging.warning("arrayFromSegment is deprecated! Binary labelmap representation may be shared between multiple segments!"
                    " Use arrayFromSegmentBinaryLabelmap to access a copy of the binary labelmap that will not modify the original labelmap."
                    " Use arrayFromSegmentInternalBinaryLabelmap to access a modifiable internal lablemap representation that may be shared"
                    " between multiple segments.")
    return arrayFromSegmentBinaryLabelmap(segmentationNode, segmentId)

def arrayFromSegmentInternalBinaryLabelmap(segmentationNode, segmentId):
    """Return voxel array of a segment's binary labelmap representation as numpy array.
    Voxels values are not copied.
    The labelmap containing the specified segment may be a shared labelmap containing multiple segments.
    To get and modify the array for a single segment, calling::
      segmentationNode->GetSegmentation()->SeparateSegment(segmentId)
    will transfer the segment from a shared labelmap into a new layer.
    Layers can be merged by calling::
      segmentationNode->GetSegmentation()->CollapseBinaryLabelmaps()
    If binary labelmap is the master representation then voxel values in the volume node can be modified
    by changing values in the numpy array. After all modifications has been completed, call::
      segmentationNode.GetSegmentation().GetSegment(segmentID).Modified()
    .. warning:: Important: memory area of the returned array is managed by VTK,
      therefore values in the array may be changed, but the array must not be reallocated.
      See :py:meth:`arrayFromVolume` for details.
    """
    vimage = segmentationNode.GetBinaryLabelmapInternalRepresentation(segmentId)
    nshape = tuple(reversed(vimage.GetDimensions()))
    import vtk.util.numpy_support
    narray = vtk.util.numpy_support.vtk_to_numpy(vimage.GetPointData().GetScalars()).reshape(nshape)
    return narray

def arrayFromSegmentBinaryLabelmap(segmentationNode, segmentId):
    """Return voxel array of a segment's binary labelmap representation as numpy array.
    Voxels values are copied.
    If binary labelmap is the master representation then voxel values in the volume node can be modified
    by changing values in the numpy array.
    After all modifications have been completed, call::
      segmentationNode.GetSegmentation().GetSegment(segmentID).Modified()
    .. warning:: Important: memory area of the returned array is managed by VTK,
      therefore values in the array may be changed, but the array must not be reallocated.
      See :py:meth:`arrayFromVolume` for details.
    """
    import slicer
    vimage = slicer.vtkOrientedImageData()
    segmentationNode.GetBinaryLabelmapRepresentation(segmentId, vimage)
    nshape = tuple(reversed(vimage.GetDimensions()))
    import vtk.util.numpy_support
    narray = vtk.util.numpy_support.vtk_to_numpy(vimage.GetPointData().GetScalars()).reshape(nshape)
    return narray

def arrayFromMarkupsControlPoints(markupsNode, world = False):
    """Return control point positions of a markups node as rows in a numpy array (of size Nx3).
    :param world: if set to True then the control points coordinates are returned in world coordinate system
      (effect of parent transform to the node is applied).
    The returned array is just a copy and so any modification in the array will not affect the markup node.
    To modify markup control points based on a numpy array, use :py:meth:`updateMarkupsControlPointsFromArray`.
    """
    numberOfControlPoints = markupsNode.GetNumberOfControlPoints()
    import numpy as np
    narray = np.zeros([numberOfControlPoints, 3])
    for controlPointIndex in range(numberOfControlPoints):
        if world:
            markupsNode.GetNthControlPointPositionWorld(controlPointIndex, narray[controlPointIndex,:])
        else:
            markupsNode.GetNthControlPointPosition(controlPointIndex, narray[controlPointIndex,:])
    return narray

def updateMarkupsControlPointsFromArray(markupsNode, narray, world = False):
    """Sets control point positions in a markups node from a numpy array of size Nx3.
    :param world: if set to True then the control point coordinates are expected in world coordinate system.
    All previous content of the node is deleted.
    """
    narrayshape = narray.shape
    if narrayshape == (0,):
        markupsNode.RemoveAllControlPoints()
        return
    if len(narrayshape) != 2 or narrayshape[1] != 3:
        raise RuntimeError("Unsupported numpy array shape: "+str(narrayshape)+" expected (N,3)")
    numberOfControlPoints = narrayshape[0]
    oldNumberOfControlPoints = markupsNode.GetNumberOfControlPoints()
    # Update existing control points
    for controlPointIndex in range(min(numberOfControlPoints, oldNumberOfControlPoints)):
        if world:
            markupsNode.SetNthControlPointPositionWorldFromArray(controlPointIndex, narray[controlPointIndex,:])
        else:
            markupsNode.SetNthControlPointPositionFromArray(controlPointIndex, narray[controlPointIndex,:])
    if numberOfControlPoints >= oldNumberOfControlPoints:
        # Add new points to the markup node
        from vtk import vtkVector3d
        for controlPointIndex in range(oldNumberOfControlPoints, numberOfControlPoints):
            if world:
                markupsNode.AddControlPointWorld(vtkVector3d(narray[controlPointIndex,:]))
            else:
                markupsNode.AddControlPoint(vtkVector3d(narray[controlPointIndex,:]))
    else:
        # Remove extra point from the markup node
        for controlPointIndex in range(oldNumberOfControlPoints, numberOfControlPoints, -1):
            markupsNode.RemoveNthControlPoint(controlPointIndex-1)

def arrayFromMarkupsCurvePoints(markupsNode, world = False):
    """Return interpolated curve point positions of a markups node as rows in a numpy array (of size Nx3).
    :param world: if set to True then the point coordinates are returned in world coordinate system
      (effect of parent transform to the node is applied).
    The returned array is just a copy and so any modification in the array will not affect the markup node.
    """
    import vtk.util.numpy_support
    if world:
        pointData = markupsNode.GetCurvePointsWorld().GetData()
    else:
        pointData = markupsNode.GetCurvePoints().GetData()
    narray = vtk.util.numpy_support.vtk_to_numpy(pointData)
    return narray

def updateVolumeFromArray(volumeNode, narray):
    """Sets voxels of a volume node from a numpy array.
    Voxels values are deep-copied, therefore if the numpy array
    is modified after calling this method, voxel values in the volume node will not change.
    Dimensions and data size of the source numpy array does not have to match the current
    content of the volume node.
    """

    vshape = tuple(reversed(narray.shape))
    if len(vshape) == 3:
        # Scalar volume
        vcomponents = 1
    elif len(vshape) == 4:
        # Vector volume
        vcomponents = vshape[0]
        vshape = vshape[1:4]
    else:
        # TODO: add support for tensor volumes
        raise RuntimeError("Unsupported numpy array shape: "+str(narray.shape))

    vimage = volumeNode.GetImageData()
    if not vimage:
        import vtk
        vimage = vtk.vtkImageData()
        volumeNode.SetAndObserveImageData(vimage)
    import vtk.util.numpy_support
    vtype = vtk.util.numpy_support.get_vtk_array_type(narray.dtype)

    # Volumes with "long long" scalar type are not rendered corectly.
    # Probably this could be fixed in VTK or Slicer but for now just reject it.
    if vtype == vtk.VTK_LONG_LONG:
        raise RuntimeError("Unsupported numpy array type: long long")

    vimage.SetDimensions(vshape)
    vimage.AllocateScalars(vtype, vcomponents)

    narrayTarget = arrayFromVolume(volumeNode)
    narrayTarget[:] = narray

    # Notify the application that image data is changed
    # (same notifications as in vtkMRMLVolumeNode.SetImageDataConnection)
    import slicer
    volumeNode.StorableModified()
    volumeNode.Modified()
    volumeNode.InvokeEvent(slicer.vtkMRMLVolumeNode.ImageDataModifiedEvent, volumeNode)

def addVolumeFromArray(narray, ijkToRAS=None, name=None, nodeClassName=None):
    """Create a new volume node from content of a numpy array and add it to the scene.
    Voxels values are deep-copied, therefore if the numpy array
    is modified after calling this method, voxel values in the volume node will not change.
    :param narray: numpy array containing volume voxels.
    :param ijkToRAS: 4x4 numpy array or vtk.vtkMatrix4x4 that defines mapping from IJK to RAS coordinate system (specifying origin, spacing, directions)
    :param name: volume node name
    :param nodeClassName: type of created volume, default: ``vtkMRMLScalarVolumeNode``.
      Use ``vtkMRMLLabelMapVolumeNode`` for labelmap volume, ``vtkMRMLVectorVolumeNode`` for vector volume.
    :return: created new volume node
    Example::
      # create zero-filled volume
      import numpy as np
      volumeNode = slicer.util.addVolumeFromArray(np.zeros((30, 40, 50)))
    Example::
      # create labelmap volume filled with voxel value of 120
      import numpy as np
      volumeNode = slicer.util.addVolumeFromArray(np.ones((30, 40, 50), 'int8') * 120,
        np.diag([0.2, 0.2, 0.5, 1.0]), nodeClassName="vtkMRMLLabelMapVolumeNode")
    """

    import slicer
    from vtk import vtkMatrix4x4

    if name is None:
        name = ""
    if nodeClassName is None:
        nodeClassName = "vtkMRMLScalarVolumeNode"

    volumeNode = slicer.mrmlScene.AddNewNodeByClass(nodeClassName, name)
    if ijkToRAS is not None:
        if not isinstance(ijkToRAS, vtkMatrix4x4):
            ijkToRAS = vtkMatrixFromArray(ijkToRAS)
        volumeNode.SetIJKToRASMatrix(ijkToRAS)
    updateVolumeFromArray(volumeNode, narray)
    volumeNode.CreateDefaultDisplayNodes()

    return volumeNode


def which(cmd, mode=os.F_OK | os.X_OK, path=None):
    """Given a command, mode, and a PATH string, return the path which
    conforms to the given mode on the PATH, or None if there is no such
    file.

    `mode` defaults to os.F_OK | os.X_OK. `path` defaults to the result
    of os.environ.get("PATH"), or can be overridden with a custom search
    path.

    """
    # Check that a given file can be accessed with the correct mode.
    # Additionally check that `file` is not a directory, as on Windows
    # directories pass the os.access check.
    def _access_check(fn, mode):
        return (os.path.exists(fn) and os.access(fn, mode)
                and not os.path.isdir(fn))

    # If we're given a path with a directory part, look it up directly rather
    # than referring to PATH directories. This includes checking relative to the
    # current directory, e.g. ./script
    if os.path.dirname(cmd):
        if _access_check(cmd, mode):
            return cmd
        return None

    if path is None:
        path = os.environ.get("PATH", os.defpath)
    if not path:
        return None
    path = path.split(os.pathsep)

    if sys.platform == "win32":
        # The current directory takes precedence on Windows.
        if not os.curdir in path:
            path.insert(0, os.curdir)

        # PATHEXT is necessary to check on Windows.
        pathext = os.environ.get("PATHEXT", "").split(os.pathsep)
        # See if the given file matches any of the expected path extensions.
        # This will allow us to short circuit when given "python.exe".
        # If it does match, only test that one, otherwise we have to try
        # others.
        if any([cmd.lower().endswith(ext.lower()) for ext in pathext]):
            files = [cmd]
        else:
            files = [cmd + ext for ext in pathext]
    else:
        # On other platforms you don't have things like PATHEXT to tell you
        # what file suffixes are executable, so just pass on cmd as-is.
        files = [cmd]

    seen = set()
    for dir in path:
        normdir = os.path.normcase(dir)
        if not normdir in seen:
            seen.add(normdir)
            for thefile in files:
                name = os.path.join(dir, thefile)
                if _access_check(name, mode):
                    return name
    return None


def logProcessOutput(proc):
    """Continuously write process output to the application log and the Python console.
    :param proc: process object.
    """
    import logging
    try:
        from slicer import app
        guiApp = app
    except ImportError:
        # Running from console
        guiApp = None
    for line in proc.stdout:
        if guiApp:
            logging.info(line.rstrip())
            guiApp.processEvents()  # give a chance the application to refresh GUI
        else:
            print(line.rstrip())
    proc.wait()
    retcode=proc.returncode
    if retcode != 0:
        logging.error("Error {}".format(retcode))


def _executePythonModule(module, args):
    """Execute a Python module as a script in Slicer's Python environment.
    Internally python -m is called with the module name and additional arguments.
    """
    # Determine pythonSlicerExecutablePath
    try:
        from slicer import app
        # If we get to this line then import from "app" is succeeded,
        # which means that we run this function from Slicer Python interpreter.
        # PythonSlicer is added to PATH environment variable in Slicer
        # therefore shutil.which will be able to find it.
        import shutil
        import subprocess
        pythonSlicerExecutablePath = which('python-real')
        if not pythonSlicerExecutablePath:
            raise RuntimeError("PythonSlicer executable not found")
    except ImportError:
        # Running from console
        import os
        import sys
        pythonSlicerExecutablePath = os.path.dirname(sys.executable)+"/python-real"
        if os.name == 'nt':
            pythonSlicerExecutablePath += ".exe"


    commandLine = [pythonSlicerExecutablePath, "-m", module]
    [commandLine.append(e) for e in args]
    logging.info(commandLine)
    proc = launchConsoleProcess(commandLine, useStartupEnvironment=False)
    logProcessOutput(proc)

def pip_install(requirements):
    """Install python packages.
    Currently, the method simply calls ``python -m pip install`` but in the future further checks, optimizations,
    user confirmation may be implemented, therefore it is recommended to use this method call instead of a plain
    pip install.
    :param requirements: requirement specifier, same format as used by pip (https://docs.python.org/3/installing/index.html)
    Example: calling from Slicer GUI
    .. code-block:: python
      pip_install("tensorflow keras scikit-learn ipywidgets")
    Example: calling from PythonSlicer console
    .. code-block:: python
      from slicer.util import pip_install
      pip_install("tensorflow")
    """
    args = 'install', requirements
    _executePythonModule('pip', args)

def launchConsoleProcess(args, useStartupEnvironment=True, cwd=None):
    """Launch a process. Hiding the console and captures the process output.
    The console window is hidden when running on Windows.
    :param args: executable name, followed by command-line arguments
    :param useStartupEnvironment: launch the process in the original environment as the original Slicer process
    :param cwd: current working directory
    :return: process object.
    """
    import subprocess
    logging.info("Cwd {}".format(cwd))
    startupEnv = None
    import os
    if os.name == 'nt':
        # Hide console window (only needed on Windows)
        info = subprocess.STARTUPINFO()
        info.dwFlags = 1
        info.wShowWindow = 0
        proc = subprocess.Popen(args, env=startupEnv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, startupinfo=info, cwd=cwd)
    else:
        proc = subprocess.Popen(args, env=startupEnv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, cwd=cwd)
    return proc

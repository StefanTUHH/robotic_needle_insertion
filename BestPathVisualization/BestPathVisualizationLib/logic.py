import sys
from itertools import compress
from typing import Optional
from numpy import linalg
import time
from multiprocessing import Process, cpu_count, Queue
from . import overlay
from .slicer_convenience_lib import *

try:
    import sklearn
    from scipy.spatial import ConvexHull
except ImportError:
    slicer.util.pip_install("scipy==1.5.2")
    slicer.util.pip_install("sklearn")
    import sklearn
    from scipy.spatial import ConvexHull

try:
    import pyvista as pv
except ImportError:
    slicer.util.pip_install("pyvista")
    import pyvista as pv


# BestPathVisualizationLogic
#

class BestPathVisualizationLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """
    T0, T1, T2, T3, T4 = 0, 0, 0, 0, 0

    def __init__(self, maxDistance, imageThreshold, colorbarValue, discreetStepsValue, socketReceiveSend, matrix):
        self.segmentEditorNode = None
        self.segmentationNode = None
        self.updateCallback = None
        self.doneCallback = None
        self.cancel = False
        self.outputPath = None
        self.useHoleFilling = False
        self.maxDistance = maxDistance
        self.imageThreshold = imageThreshold
        self.initTransform = True
        self.socketReceiveSend = socketReceiveSend
        self.matrix = matrix
        self.colorbarValue = colorbarValue
        self.discreetStepsValue = discreetStepsValue
        self.arrShape = None
        self.overlayTypeIndex = 0
        self.maxKernelSize = 0
        self.distanceWeighting = None
        self.angleWeighting = None
        self.gantry_pose = np.asarray(
            ((1., 0., 0., -1000.,), (0., 1., 0., -1000.,), (0., 0., 1., 1390.,), (0., 0., 0., 1.,)))
        self.scalarData = None
        self.scalarDataMoveIt = None
        self.targetGlobal = [0, 0, 0]
        self.targetPoint = [0, 0, 0]

    def runSegment(self, inputVolume, outputModel, targetNode):
        """
        Run the actual algorithm
        """
        if not self.isValidInputOutputData(inputVolume, outputModel, targetNode):
            return False

        logging.info('Processing started')
        npArrPoly = arrayFromModelPoints(outputModel)

        if len(npArrPoly) == 0:
            start_A = time.time()
            self.segmentSkin(inputVolume, outputModel)
            end_A = time.time()
            self.T0 = end_A - start_A
            npArrPoly = arrayFromModelPoints(outputModel)
        else:
            logging.info("Using previously segmented skin model")
            self.initWithPreviousModel(inputVolume, outputModel)

        ret = self.addOverlay(outputModel, targetNode)
        if ret is None:
            return False
        indices, insideTransformed = ret

        relevantPoints = np.asarray(self.scalarData)[indices] < self.globalMaxDensity
        if len(relevantPoints) > 0 and not self.applyMaxKernel(np.asarray(indices)[relevantPoints],
                                                               insideTransformed[relevantPoints, :]):
            return False

        combinedArray = np.zeros((len(npArrPoly) + 1, 4), dtype=np.float)
        combinedArray[0, :3] = self.targetGlobal
        combinedArray[1:, :3] = npArrPoly
        combinedArray[1:, 3] = self.scalarData

        combinedArrayMoveIt, foundInfeasiblePosition = self.applyReachability(combinedArray, npArrPoly)

        self.writeOutput(targetNode, combinedArray, combinedArrayMoveIt)

        self.displayResults(combinedArrayMoveIt is not None, npArrPoly, outputModel, foundInfeasiblePosition)

        logging.info('Processing completed')
        if self.waypoint(100):
            return False
        return True

    @staticmethod
    def np_matrix_from_vtk(vtk_matrix):
        result = np.eye(4)
        for r in range(4):
            for c in range(4):
                result[r, c] = vtk_matrix.GetElement(r, c)
        return result

    def applyReachability(self, combinedArray, npArrPoly):
        # Check if Points are reachable by MoveIt
        combinedArrayMoveItIn = np.delete(combinedArray, 0, 0)
        combinedArrayMoveIt = None
        foundInfeasiblePosition = False
        if self.socketReceiveSend is not None:
            if self.waypoint(95, "Checking Map with MoveIt"):
                return False

            tf_matrix = self.np_matrix_from_vtk(self.matrix)
            self.applyGantryMesh(tf_matrix)
            self.applyCollisionMesh(npArrPoly)
            start_E = time.time()
            combinedArrayMoveIt = self.checkMoveIt(combinedArrayMoveItIn)
            end_E = time.time()
            self.T4 = end_E - start_E
            moveItBool = combinedArrayMoveIt[:, 4]
            # Create VTK Color Map
            self.scalarDataMoveIt = vtk.vtkFloatArray()
            self.scalarDataMoveIt.SetNumberOfComponents(0)
            self.scalarDataMoveIt.SetNumberOfValues(len(npArrPoly))
            self.scalarDataMoveIt.SetName("density")

            toAdd = int(np.ceil((self.colorbarValue - self.globalMinDensity) / 256))

            not_reachable = moveItBool == 1
            np.asarray(self.scalarDataMoveIt)[~not_reachable] = combinedArray[
                np.add(np.where(~not_reachable), 1), 3].flatten()
            if np.any(not_reachable):
                foundInfeasiblePosition = True
                np.asarray(self.scalarDataMoveIt)[not_reachable] = self.globalMaxDensity + toAdd
            logging.info("Finished with MoveIT")
        return combinedArrayMoveIt, foundInfeasiblePosition

    def writeOutput(self, targetNode, combinedArray, combinedArrayMoveIt: Optional = None):
        if self.outputPath is not None and self.outputPath is not '':
            string_path = self.outputPath.split('.')
            targetName = targetNode.GetName()

            outputPath_ColormapMoveIt = string_path[0] + targetName + '_Rob.txt'
            outputPath_Colormap = string_path[0] + targetName + '.txt'
            outputPath_ComputationTime = string_path[0] + targetName + '_ComputationTime.txt'

            timeEstimation = "time" in outputPath_Colormap

            logging.info("Saving result to file {}".format(outputPath_Colormap))
            with open(outputPath_Colormap, "w") as f:
                np.savetxt(f, combinedArray)
            if timeEstimation:
                logging.info("Saving Computation Time.")
                with open(outputPath_ComputationTime, "w") as f:
                    np.savetxt(f, [self.T0, self.T1, self.T2, self.T3, self.T4])

            if combinedArrayMoveIt is not None:
                logging.info("Saving moveit result to file {}".format(outputPath_ColormapMoveIt))
                with open(outputPath_ColormapMoveIt, "w") as f:
                    np.savetxt(f, combinedArrayMoveIt)

    def displayResults(self, couldConnectToMoveIt, npArrPoly, outputModel, foundInfeasiblePosition):
        if couldConnectToMoveIt:
            min_point = npArrPoly[np.argmin(self.scalarDataMoveIt), :]
        else:
            min_point = npArrPoly[np.argmin(self.scalarData), :]

        fixedLRS = slicer.vtkMRMLMarkupsFiducialNode()
        fixedLRS.SetName('Opt_Surface')
        fixedLRS.AddFiducial(min_point[0], min_point[1], min_point[2])
        slicer.mrmlScene.AddNode(fixedLRS)
        fixedLRS.SetDisplayVisibility(True)
        outputModel.CreateDefaultDisplayNodes()

        print("Showing resulting model")
        if couldConnectToMoveIt:
            outputModel.GetPolyData().GetPointData().AddArray(self.scalarDataMoveIt)
        else:
            outputModel.GetPolyData().GetPointData().AddArray(self.scalarData)

        arrayFromModelPointDataModified(outputModel, "density")
        arrayFromModelPointsModified(outputModel)
        # Show pretty results
        modelDisplayNode = outputModel.GetDisplayNode()
        modelDisplayNode.SetActiveScalarName("density")
        modelDisplayNode.SetScalarRangeFlag(1)
        scalarRange = modelDisplayNode.GetScalarRange()
        # Fixes issue when no point is reachable
        scalarRange = (min(self.colorbarValue - 1, min(scalarRange[0], np.min(self.scalarData))),
                       max(scalarRange[1], np.max(self.scalarData)))
        newColorSize = int(
            round((scalarRange[1] - self.colorbarValue) / (self.colorbarValue - scalarRange[0]) * 256 + 256))
        densityColor = slicer.mrmlScene.AddNode(slicer.modules.colors.logic().CopyNode(
            slicer.mrmlScene.GetNodeByID("vtkMRMLColorTableNodeFileColdToHotRainbow.txt"), "densityColor"))
        densityColor.SetNumberOfColors(int(newColorSize))
        offset = 10
        for i in range(256, newColorSize - offset):
            densityColor.SetColor(i, 0.453125, 0, 0)
        for i in range(newColorSize - offset, newColorSize - 1):
            densityColor.SetColor(i, 0.875, 0.671875, 0.41015625)
        if couldConnectToMoveIt and foundInfeasiblePosition:
            densityColor.SetColor(newColorSize - 1, 0.35, 0.35, 0.35)
        else:
            densityColor.SetColor(newColorSize - 1, 0.875, 0.671875, 0.41015625)

        modelDisplayNode.SetAndObserveColorNodeID(densityColor.GetID())
        print("Displaying...")
        modelDisplayNode.SetScalarVisibility(True)

        modelDisplayNode.SetScalarRangeFlag(0)
        modelDisplayNode.SetScalarRange((scalarRange[0], scalarRange[1]))

    def addOverlay(self, outputModel, targetNode):
        targetNode.GetMarkupPoint(0, 0, self.targetGlobal)
        self.targetPoint = self.transformToCT(self.targetGlobal)
        logging.info("Surface point in CT indices: {}".format(self.targetPoint))
        npArrPoly = arrayFromModelPoints(outputModel)
        if self.waypoint(25, "Adding scalar overlay"):
            return None

        self.scalarData = vtk.vtkFloatArray()
        self.scalarData.SetNumberOfComponents(0)
        self.scalarData.SetNumberOfValues(len(npArrPoly))
        self.scalarData.SetName("density")

        # Calc density for all relevant surface points
        point_VolumeRas = vtk.vtkPoints()
        transformed = self.transformPointsToCT(outputModel.GetPolyData().GetPoints(), point_VolumeRas)

        distances = np.linalg.norm(npArrPoly - self.targetGlobal, axis=1)
        pointOutside = np.add(np.add(distances > self.maxDistance, ~self.pointInVolume(transformed, self.arrShape)),
                              npArrPoly[:, 1] < 50 + np.min(npArrPoly[:, 1]))
        pointInside = ~pointOutside

        np.asarray(self.scalarData)[pointOutside] = self.globalMaxDensity

        indices = list(compress(range(len(pointInside)), pointInside))
        insideTransformed = np.asarray(list(compress(transformed, pointInside)))

        self.maxIdx = len(indices) - 1
        logging.info(self.maxIdx)

        with np.errstate(divide='ignore', invalid='ignore'):
            if self.overlayTypeIndex == 7:
                start_B = time.time()
                self.overlayTypeIndex = 0
                if not calcDensityInThread(self, indices, insideTransformed):
                    return None
                if self.waypoint(40):
                    return None
                combinedData = np.copy(self.scalarData)
                relevant_indices = np.logical_and(combinedData < self.colorbarValue, pointInside)
                end_B = time.time()
                self.T1 = end_B - start_B

                if self.distanceWeighting > 0:
                    self.overlayTypeIndex = 4  # Distance
                    start_C = time.time()
                    if not calcDensityInThread(self, indices, insideTransformed):
                        return None
                    if self.waypoint(60):
                        return None
                    maxDistance = self.maxDistance

                    np.asarray(self.scalarData)[relevant_indices] = np.asarray(self.scalarData)[
                                                                        relevant_indices] / maxDistance * self.colorbarValue
                    combinedData[relevant_indices] = self.distanceWeighting * np.asarray(self.scalarData)[
                        relevant_indices]
                    end_C = time.time()
                    self.T2 = end_C - start_C
                else:
                    combinedData[relevant_indices] = 0

                if self.angleWeighting > 0:
                    start_D = time.time()
                    self.overlayTypeIndex = 5  # Angle
                    if not calcDensityInThread(self, indices, insideTransformed):
                        return None
                    maxAngle = np.pi / 2
                    np.asarray(self.scalarData)[relevant_indices] = np.asarray(self.scalarData)[
                                                                        relevant_indices] / maxAngle * self.colorbarValue
                    combinedData[relevant_indices] = combinedData[relevant_indices] + self.angleWeighting * \
                                                     np.asarray(self.scalarData)[relevant_indices]
                    end_D = time.time()
                    self.T3 = end_D - start_D

                np.asarray(self.scalarData)[:] = combinedData
                self.overlayTypeIndex = 7
            else:
                if not calcDensityInThread(self, indices, insideTransformed):
                    return None
            return indices, insideTransformed

    def initWithPreviousModel(self, inputVolume, outputModel):
        if self.arrShape is None:
            self.transformRasToVolumeRas = vtk.vtkGeneralTransform()
            slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, inputVolume.GetParentTransformNode(),
                                                                 self.transformRasToVolumeRas)
            # Get voxel coordinates from physical coordinates
            volumeRasToIjk = vtk.vtkMatrix4x4()
            inputVolume.GetRASToIJKMatrix(volumeRasToIjk)
            self.volumeRasToIjk = vtk.vtkTransform()
            self.volumeRasToIjk.SetMatrix(volumeRasToIjk)
            self.inputVolumeNPArray = np.asarray(slicer.util.arrayFromVolume(inputVolume))
            self.globalMaxDensity = np.max(self.inputVolumeNPArray) + 10
            self.globalMinDensity = np.min(self.inputVolumeNPArray)
            self.arrShape = np.asarray(np.shape(self.inputVolumeNPArray))
            self.spacing = inputVolume.GetSpacing()

        logging.info("Shape: " + str(self.arrShape))
        self.result = np.zeros(self.arrShape) + self.globalMaxDensity
        outputModel.GetPolyData().GetPointData().RemoveArray("density")

    def segmentSkin(self, inputVolume, outputModel):
        # If volume node is transformed, apply that transform to get volume's RAS coordinates
        self.transformRasToVolumeRas = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, inputVolume.GetParentTransformNode(),
                                                             self.transformRasToVolumeRas)
        # Get voxel coordinates from physical coordinates
        volumeRasToIjk = vtk.vtkMatrix4x4()
        inputVolume.GetRASToIJKMatrix(volumeRasToIjk)
        self.spacing = inputVolume.GetSpacing()
        self.volumeRasToIjk = vtk.vtkTransform()
        self.volumeRasToIjk.SetMatrix(volumeRasToIjk)
        self.inputVolumeNPArray = np.asarray(slicer.util.arrayFromVolume(inputVolume))
        self.globalMaxDensity = np.max(self.inputVolumeNPArray) + 10
        self.globalMinDensity = np.min(self.inputVolumeNPArray)
        self.arrShape = np.asarray(np.shape(self.inputVolumeNPArray))
        logging.info("Shape: " + str(self.arrShape))
        self.result = np.zeros(self.arrShape) + self.globalMaxDensity
        if self.waypoint(5, "Creating segmentation"):
            return False
        # Create segmentation
        if self.segmentationNode is not None:
            slicer.mrmlScene.RemoveNode(self.segmentationNode)

        addedSegmentID, segmentEditorWidget, segmentEditorNode = self.initSegmentationNode(inputVolume)

        self.applyThresholding(segmentEditorWidget, inputVolume)

        if self.waypoint(10, "Selecting largest island"):
            return False

        self.applyLargestIsland(segmentEditorWidget)

        if self.useHoleFilling:
            if self.waypoint(12, "Filling holes"):
                return False

            self.applySmoothing(segmentEditorWidget)

            if self.waypoint(18, "Inverting"):
                return False

            self.applyInverting(segmentEditorWidget)

            if self.waypoint(19, "Selecting largest island"):
                return False

            # Selecting largest island
            self.applyLargestIsland(segmentEditorWidget)

            if self.waypoint(20, "Inverting"):
                return False

            self.applyInverting(segmentEditorWidget)

        # Cleanup
        segmentEditorWidget.setActiveEffectByName(None)
        slicer.mrmlScene.RemoveNode(segmentEditorNode)

        if self.waypoint(21, "Creating closed surface"):
            return False

        outputPolyData = vtk.vtkPolyData()
        slicer.vtkSlicerSegmentationsModuleLogic.GetSegmentClosedSurfaceRepresentation(self.segmentationNode,
                                                                                       addedSegmentID, outputPolyData)
        outputModel.SetAndObservePolyData(outputPolyData)
        self.segmentationNode.GetDisplayNode().SetVisibility(False)

    def checkMoveIt(self, modelPoints):
        # Split Array
        positions = modelPoints[:, :3]
        color = modelPoints[:, 3]

        #  Only eval Points which have a feasible color
        relevantColor = color < self.globalMaxDensity
        positions_color = positions[relevantColor, :]

        # Round Positions
        positions_color_r = np.ceil(positions_color / self.discreetStepsValue) * self.discreetStepsValue

        # Find unique Positions
        positions_color_r_unique = np.unique(positions_color_r, axis=0)

        transformedPoints = self.transformToBase(
            np.append(np.expand_dims(self.targetGlobal, axis=0), positions_color_r_unique, axis=0))
        sendArray = np.append(np.array([len(positions_color_r_unique)]), transformedPoints.flatten())
        target_transformed = transformedPoints[0, :3]
        self.waypoint(96, 'Starting Checking Points with MoveIt')

        # print(sendArray)
        print('Points to evaluate:' + str(len(positions_color_r_unique)))

        # Convert to Float
        moveIt_result = self.sendInPeices(positions_color_r_unique, sendArray, target_transformed)

        # print("moveIt_result_boolean", moveIt_result)

        moveIt_result_idx = np.ones(len(moveIt_result))
        if np.any(np.asarray(moveIt_result)):
            moveIt_result_idx[np.asarray(moveIt_result)] = 2

        # 0 --> not in map, 1--> in colormap but not reachabel, 2--> in colormap and reachable
        self.waypoint(99, 'Finished Checking Points with MoveIt')

        # Update Color List
        color_array_moveIT = np.zeros(len(positions_color_r))
        for count, p in enumerate(positions_color_r_unique):
            color_array_moveIT[np.all(positions_color_r == p, axis=1)] = moveIt_result_idx[count]

        modelPointsOut = np.zeros((len(positions), 5), dtype=np.float)
        modelPointsOut[:, :3] = positions
        modelPointsOut[:, 3] = color
        modelPointsOut[relevantColor, 4] = color_array_moveIT
        return modelPointsOut

    def sendInPeices(self, positions_color_r_unique, sendArray, target_transformed):
        sent = 0
        socketsForSimulation = self.socketReceiveSend if len(self.socketReceiveSend) <= 1 else self.socketReceiveSend[1:]
        per_package = int(np.ceil(len(positions_color_r_unique) / len(socketsForSimulation)))
        for s in socketsForSimulation:
            if len(positions_color_r_unique) <= sent:
                break
            up_idx = min(sent + per_package, len(positions_color_r_unique))
            to_send = np.append(np.asarray((up_idx - sent)), target_transformed[:3])
            to_send = np.append(to_send, sendArray[(4 + sent * 3):(up_idx * 3 + 4)])
            sent = up_idx
            s.send(str(to_send.tolist()).encode())
        # Receive Answer
        moveIt_result = []
        sent = 0
        for s in socketsForSimulation:
            if len(positions_color_r_unique) <= sent:
                break
            up_idx = min(sent + per_package, len(positions_color_r_unique))
            sent = up_idx
            recv_data = s.recv(102400)
            msg = list(recv_data.decode('utf-8'))
            moveIt_result += [bool(int(i)) for i in msg]
        return moveIt_result

    def applyMaxKernel(self, indices, pointsTransformed):
        if self.maxKernelSize == 0:
            return True
        self.waypoint(75., "Applying max kernel")

        from sklearn.neighbors import radius_neighbors_graph
        values = np.copy(self.scalarData)
        values = values[indices]
        neighbours = radius_neighbors_graph(pointsTransformed * self.spacing, radius=self.maxKernelSize, n_jobs=-1,
                                            include_self=True)

        executeInPeices(lambda q, idx, sent: self.applyMaxKernelImpl(q, idx, neighbours, values, sent), indices, np.asarray(self.scalarData))

        if self.waypoint(95.):
            return False
        return True

    @staticmethod
    def applyMaxKernelImpl(q: Queue, indices, neighbours, values, idxOffset):
        result = np.zeros((len(indices)))
        for neighbourIdx, idx in enumerate(indices):
            finalIdx = np.asarray(neighbours.getrow(neighbourIdx + idxOffset).toarray(), dtype=np.bool).flatten()
            result[neighbourIdx] = np.max(values[finalIdx])
        q.put(result)

    def pointInVolume(self, point, maxDim):
        return np.logical_and(np.all(point > 0, axis=1), np.all(point < maxDim - 1, axis=1))

    def transformToCT(self, point):
        point_VolumeRas = self.transformRasToVolumeRas.TransformPoint(point[0:3])
        point_Ijk = self.volumeRasToIjk.TransformPoint(point_VolumeRas)
        return np.asarray(np.flip(point_Ijk[0:3], 0), dtype=np.int)

    def transformPointsToCT(self, points, point_VolumeRas):
        tmp = vtk.vtkPoints()
        self.transformRasToVolumeRas.TransformPoints(points, tmp)
        self.volumeRasToIjk.TransformPoints(tmp, point_VolumeRas)
        return np.asarray(np.flip(vtk.util.numpy_support.vtk_to_numpy(point_VolumeRas.GetData())[:, 0:3], 1),
                          dtype=np.int)

    def applyGantryMesh(self, tf_matrix: np.ndarray):
        IJK_T_RAS = np.eye(4)

        transformed_pose = tf_matrix.dot(IJK_T_RAS).dot(self.gantry_pose)
        size = 3000.
        points = np.asarray(((0., 0., 0., 1.), (0., size, 0., 1.), (size, 0., 0., 1.), (size, 0., 0., 1.),
                             (0., size, 0., 1.), (size, size, 0., 1.))).T
        points = transformed_pose.dot(points)[:3, :]
        max_v = np.max(points, axis=1)
        min_v = np.min(points, axis=1)
        center = (max_v - min_v) / 2 + min_v
        points = np.subtract(points.T, center)

        sendArray = np.append(np.append(np.asarray((-2., 1.)), center), points.flatten())
        for s in self.socketReceiveSend:
            s.send(str(sendArray.tolist()).encode())
        # Wait for bb to be applied
        for s in self.socketReceiveSend:
            s.recv(102400)

    def transformPoints(self, points, matrix):
        transformed = matrix.dot(np.append(points, np.ones((np.shape(points)[0], 1)), axis=1).T)
        return transformed[:3, :].T

    def transformToBase(self, points):
        tf_mat = self.np_matrix_from_vtk(self.matrix)
        IJK_T_RAS = np.eye(4)
        IJK_T_RAS[0, 0] = -1
        IJK_T_RAS[1, 1] = -1
        tf_mat = tf_mat.dot(IJK_T_RAS)
        return self.transformPoints(points, tf_mat)

    def applyCollisionMesh(self, points: np.ndarray):

        resolution = 30.
        # Round Positions
        positions_to_eval_subsampled = np.round(points / resolution) * resolution

        # Find unique Positions
        unique_points_eval = np.unique(positions_to_eval_subsampled, axis=0)
        if len(unique_points_eval) == 0:
            return

        points_arr = self.transformToBase(unique_points_eval)
        cloud = pv.PolyData(points_arr[:, :3])
        # cloud.plot()

        volume = cloud.delaunay_3d(alpha=resolution * 1.2)
        shell = volume.extract_geometry()
        # shell.plot()
        # Hull
        # hull = ConvexHull(points_arr[:, :3])
        indices = shell.faces.reshape((-1, 4))[:, 1:]
        vertices = points_arr[indices]
        # add table
        max_v = np.max(np.max(vertices, axis=0), axis=0)
        min_v = np.min(np.min(vertices, axis=0), axis=0)

        table_slack = 30.
        depth = 500.
        min_v[2] += table_slack
        min_v[:2] -= table_slack
        max_v[:2] += table_slack

        table = np.asarray(
            (((min_v[0], min_v[1], min_v[2]), (min_v[0], max_v[1], min_v[2]), (max_v[0], max_v[1], min_v[2])),
             ((min_v[0], min_v[1], min_v[2]), (max_v[0], max_v[1], min_v[2]), (max_v[0], min_v[1], min_v[2])),
             ((min_v[0], min_v[1], min_v[2] - depth), (min_v[0], max_v[1], min_v[2] - depth),
              (max_v[0], max_v[1], min_v[2] - depth)),
             ((min_v[0], min_v[1], min_v[2] - depth), (max_v[0], max_v[1], min_v[2] - depth),
              (max_v[0], min_v[1], min_v[2] - depth)),
             ((min_v[0], min_v[1], min_v[2]), (min_v[0], max_v[1], min_v[2]), (min_v[0], max_v[1], min_v[2] - depth)),
             ((min_v[0], max_v[1], min_v[2] - depth), (min_v[0], min_v[1], min_v[2] - depth),
              (min_v[0], min_v[1], min_v[2])),
             ((max_v[0], min_v[1], min_v[2]), (max_v[0], max_v[1], min_v[2]), (max_v[0], max_v[1], min_v[2] - depth)),
             ((max_v[0], max_v[1], min_v[2] - depth), (max_v[0], min_v[1], min_v[2] - depth),
              (max_v[0], min_v[1], min_v[2])),
             ((min_v[0], min_v[1], min_v[2]), (max_v[0], min_v[1], min_v[2]), (max_v[0], min_v[1], min_v[2] - depth)),
             ((min_v[0], min_v[1], min_v[2]), (max_v[0], min_v[1], min_v[2] - depth),
              (min_v[0], min_v[1], min_v[2] - depth)),
             ((min_v[0], max_v[1], min_v[2]), (max_v[0], max_v[1], min_v[2]), (max_v[0], max_v[1], min_v[2] - depth)),
             ((min_v[0], max_v[1], min_v[2]), (max_v[0], max_v[1], min_v[2] - depth),
              (min_v[0], max_v[1], min_v[2] - depth))))
        vertices = np.append(table, vertices, axis=0)

        max_v = np.max(np.max(vertices, axis=0), axis=0)
        min_v = np.min(np.min(vertices, axis=0), axis=0)
        center = (max_v - min_v) / 2 + min_v
        vertices -= center

        # Convert to Float
        sendArray = np.append(np.append(np.asarray((-2., 0.)), center), np.ndarray.flatten(vertices))
        for s in self.socketReceiveSend:
            s.send(str(sendArray.tolist()).encode())
        # Wait for bb to be applied
        for s in self.socketReceiveSend:
            s.recv(102400)

    @staticmethod
    def isValidInputOutputData(inputVolumeNode, outputModel, targetNode):
        """Validates if the output is not the same as input
        """
        if not inputVolumeNode:
            logging.debug('isValidInputOutputData failed: no input volume node defined')
            return False
        if not outputModel:
            logging.debug('isValidInputOutputData failed: no output volume node defined')
            return False
        if not targetNode:
            logging.debug('isValidInputOutputData failed: no target node defined')
            return False
        return True

    def waypoint(self, percentageDone=None, log=None):
        if log:
            logging.info(log)
        if self.updateCallback:
            self.updateCallback(percentageDone)
        if self.cancel:
            self.cancel = False
            return True
        return False

    def cleanUp(self):
        if self.segmentationNode is not None:
            slicer.mrmlScene.RemoveNode(self.segmentationNode)

    def initSegmentationNode(self, inputVolume):
        self.segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(inputVolume)
        addedSegmentID = self.segmentationNode.GetSegmentation().AddEmptySegment("skin")
        segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
        segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
        segmentEditorWidget.setSegmentationNode(self.segmentationNode)
        segmentEditorWidget.setMasterVolumeNode(inputVolume)
        return addedSegmentID, segmentEditorWidget, segmentEditorNode

    def applyThresholding(self, segmentEditorWidget, inputVolume):
        rangeHU = inputVolume.GetImageData().GetScalarRange()

        # Thresholding
        segmentEditorWidget.setActiveEffectByName("Threshold")
        effect = segmentEditorWidget.activeEffect()
        effect.setParameter("MinimumThreshold", self.imageThreshold)
        effect.setParameter("MaximumThreshold", rangeHU[1])
        effect.self().onApply()

    def applyLargestIsland(self, segmentEditorWidget):
        segmentEditorWidget.setActiveEffectByName("Islands")
        effect = segmentEditorWidget.activeEffect()
        effect.setParameter("Operation", "KEEP_LARGEST_ISLAND")
        effect.setParameter("MinimumSize", 1000)
        effect.self().onApply()

    def applySmoothing(self, segmentEditorWidget):
        segmentEditorWidget.setActiveEffectByName("Smoothing")
        effect = segmentEditorWidget.activeEffect()
        effect.setParameter("SmoothingMethod", "CLOSING")
        effect.setParameter("KernelSizeMm", 10)
        effect.self().onApply()

    def applyInverting(self, segmentEditorWidget):
        segmentEditorWidget.setActiveEffectByName("Logical operators")
        effect = segmentEditorWidget.activeEffect()
        effect.setParameter("Operation", "INVERT")
        effect.self().onApply()


def calcDensityInThread(logic: BestPathVisualizationLogic, indices: list, insideTransformed: np.ndarray):
    indices_per_process = int(np.ceil(len(indices) / cpu_count()))
    sent = 0
    processes = []
    original_stdin = sys.stdin
    sys.stdin = open(os.devnull)
    try:
        for _ in range(cpu_count()):
            if sent >= len(indices):
                break
            local_sent = min(sent + indices_per_process, len(indices))
            q = Queue()
            p = Process(target=calcDensityImpl, args=(
                q, logic.targetPoint, logic.arrShape, insideTransformed[sent:local_sent, :], indices[sent:local_sent],
                logic.imageThreshold, logic.globalMaxDensity, logic.inputVolumeNPArray, logic.overlayTypeIndex,
                logic.spacing))
            sent = local_sent
            p.start()
            processes.append((q, p))

        sent = 0
        for q, p in processes:
            if sent >= len(indices):
                break
            local_sent = min(sent + indices_per_process, len(indices))
            np.asarray(logic.scalarData)[indices[sent:local_sent]] = q.get()
            p.join()
            sent = local_sent
    finally:
        sys.stdin.close()
        sys.stdin = original_stdin
    return True

def calcDensityImpl(q: Queue, targetPoint: np.ndarray, arrShape: np.ndarray, insideTransformed: np.ndarray,
                    indices: list, imageThreshold: float,
                    globalMaxDensity: float, inputVolumeNPArray: np.ndarray, overlayTypeIndex: int,
                    spacing: np.ndarray):
    densities = np.zeros((len(indices)))
    for dispIdx in range(len(indices)):
        densities[dispIdx] = overlay.calcDensity(targetPoint, arrShape, insideTransformed[dispIdx], indices[dispIdx],
                                                 imageThreshold, globalMaxDensity, inputVolumeNPArray, overlayTypeIndex,
                                                 spacing)
    q.put(densities)

def executeInPeices(function, indices, outArray):
    indices_per_process = int(np.ceil(len(indices) / cpu_count()))
    sent = 0
    processes = []
    original_stdin = sys.stdin
    sys.stdin = open(os.devnull)
    try:
        for _ in range(cpu_count()):
            if sent >= len(indices):
                break
            local_sent = min(sent + indices_per_process, len(indices))
            q = Queue()
            p = Process(target=function,
                        args=(q, indices[sent:local_sent], sent))
            sent = local_sent
            p.start()
            processes.append((q, p))

        sent = 0
        for q, p in processes:
            if sent >= len(indices):
                break
            local_sent = min(sent + indices_per_process, len(indices))
            outArray[indices[sent:local_sent]] = q.get()
            p.join()
            sent = local_sent
    finally:
        sys.stdin.close()
        sys.stdin = original_stdin

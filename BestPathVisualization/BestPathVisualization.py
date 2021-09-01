# This Python file uses the following encoding: utf-8
from time import sleep

from BestPathVisualizationLib.slicer_convenience_lib import *
import BestPathVisualizationLib
import ctk
import qt
import socket
import json


#
# BestPathVisualization
#

class BestPathVisualization(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Best Path Visualization"
        self.parent.categories = ["Utilities"]
        self.parent.dependencies = []
        self.parent.contributors = ["Stefan Gerlach"]
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
It performs a simple thresholding on the input volume and optionally captures a screenshot.
"""
        self.parent.helpText += self.getDefaultModuleDocumentationLink()
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""  # replace with organization, grant and thanks.


#
# BestPathVisualizationWidget
#

class BestPathVisualizationWidget(ScriptedLoadableModuleWidget):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        self.connected = False
        # Instantiate and connect widgets ...

        #
        # Parameters Area
        #
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Parameters"
        self.layout.addWidget(parametersCollapsibleButton)

        # Layout within the dummy collapsible button
        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        #
        # input volume selector
        #
        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector.selectNodeUponCreation = True
        self.inputSelector.addEnabled = False
        self.inputSelector.removeEnabled = False
        self.inputSelector.noneEnabled = False
        self.inputSelector.showHidden = False
        self.inputSelector.showChildNodeTypes = False
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.inputSelector.setToolTip("Pick the input to the algorithm.")
        parametersFormLayout.addRow("Input Volume: ", self.inputSelector)

        #
        # output volume selector
        #
        self.outputSelector = slicer.qMRMLNodeComboBox()
        self.outputSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.outputSelector.selectNodeUponCreation = True
        self.outputSelector.addEnabled = True
        self.outputSelector.removeEnabled = True
        self.outputSelector.noneEnabled = True
        self.outputSelector.showHidden = False
        self.outputSelector.showChildNodeTypes = False
        self.outputSelector.setMRMLScene(slicer.mrmlScene)
        self.outputSelector.setToolTip("Pick the output to the algorithm.")
        parametersFormLayout.addRow("Output Model: ", self.outputSelector)


        label, self.imageThresholdSliderWidget = \
            self.createSliderWidgetAndDescription("Image threshold [HU]",
                                                  "Set threshold value for computing the segmentation. Voxels that "
                                                  "have intensities lower than this value will set to zero.",
                                                  -460, -1000, 1000, visible=True)
        parametersFormLayout.addRow(label, self.imageThresholdSliderWidget)

        label, self.distanceSliderWidget = \
            self.createSliderWidgetAndDescription("Maximum distance [mm]", "Set maximum distance value for insertions.",
                                                  160, 0, 1000, visible=True)
        parametersFormLayout.addRow(label, self.distanceSliderWidget)

        label, self.colorbarSliderWidget = \
            self.createSliderWidgetAndDescription("Maximum colorbar value [HU]", "Set colorbar value.", 500, 0, 3000,
                                                  visible=True)
        parametersFormLayout.addRow(label, self.colorbarSliderWidget)

        label, self.maxKernelSliderWidget = \
            self.createSliderWidgetAndDescription("Safety Margin [mm]",
                                                  "Set kernel size for adding safety margins to projections", 5, 0, 50,
                                                  1, True)
        parametersFormLayout.addRow(label, self.maxKernelSliderWidget)

        label, self.discreetStepsSliderWidget = \
            self.createSliderWidgetAndDescription("Robot reachability step size [mm]",
                                                  "Set descreet steps to evaluate robot reachability.", 30, 0, 50, 1,
                                                  True)
        parametersFormLayout.addRow(label, self.discreetStepsSliderWidget)
        #
        # use expensive model
        #
        self.fillHolesCheckBox = qt.QRadioButton()
        self.fillHolesCheckBox.text = "Use hole filling"
        self.fillHolesCheckBox.checked = True
        self.fillHolesCheckBox.setToolTip("Use hole filling (more complex model building but faster map generation).")
        parametersFormLayout.addRow(self.fillHolesCheckBox)

        self.conditionalWidgets = []
        self.distanceWeightLabel, self.distanceWeightSliderWidget = \
            self.createSliderWidgetAndDescription("Distance Weighting",
                                                  "Set weight of distance objective function.",
                                                  0.5)
        self.conditionalWidgets.append(self.distanceWeightLabel)
        self.conditionalWidgets.append(self.distanceWeightSliderWidget)
        parametersFormLayout.addRow(self.distanceWeightLabel, self.distanceWeightSliderWidget)

        self.angleWeightLabel, self.angleWeightSliderWidget = \
            self.createSliderWidgetAndDescription("Insertion Angle Weighting",
                                                  "Set weight of insertion angle objective function.",
                                                  0.5)
        self.conditionalWidgets.append(self.angleWeightLabel)
        self.conditionalWidgets.append(self.angleWeightSliderWidget)
        parametersFormLayout.addRow(self.angleWeightLabel, self.angleWeightSliderWidget)

        self.typeCombobox = qt.QComboBox()
        self.typeCombobox.setToolTip("Change the type of overlay projected onto the skin")
        self.typeCombobox.addItems(
            ["Max. Value", "Quantile Min. Value", "Mean Value", "Standard Deviation", "Distance", "Insertion angle",
             "All", "Optimal point"])
        self.typeCombobox.setEditable(False)
        parametersFormLayout.addRow("Overlay Type: ", self.typeCombobox)

        self.targetNameWidget = slicer.qMRMLNodeComboBox()
        self.targetNameWidget.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.targetNameWidget.selectNodeUponCreation = True
        self.targetNameWidget.addEnabled = False
        self.targetNameWidget.removeEnabled = False
        self.targetNameWidget.noneEnabled = False
        self.targetNameWidget.showHidden = False
        self.targetNameWidget.showChildNodeTypes = False
        self.targetNameWidget.setMRMLScene(slicer.mrmlScene)
        self.targetNameWidget.setToolTip("Pick the target to the algorithm.")
        parametersFormLayout.addRow("Target: ", self.targetNameWidget)

        self.outputFileSelector = ctk.ctkPathLineEdit()
        self.outputFileSelector.filters = ctk.ctkPathLineEdit().Writable | ctk.ctkPathLineEdit().Files
        self.outputFileSelector.nameFilters = ["*.txt"]
        self.outputFileSelector.settingKey = 'NumpyOutputFile'
        parametersFormLayout.addRow("Output density filename:", self.outputFileSelector)

        #
        # Segment Button
        #
        self.applySegmentButton = qt.QPushButton("Calculate Overlay")
        self.applySegmentButton.toolTip = "Run the algorithm."
        self.applySegmentButton.enabled = False
        self.cancelButton = qt.QPushButton("Cancel")
        self.cancelButton.toolTip = "Cancel the algorithm."
        self.cancelButton.enabled = False
        parametersFormLayout.addRow(self.applySegmentButton)
        parametersFormLayout.addRow(self.cancelButton)

        self.progress = qt.QProgressBar()
        self.progress.setGeometry(0, 0, 300, 25)
        self.progress.setMaximum(100)
        self.progress.value = 0
        parametersFormLayout.addRow(self.progress)

        # Input File for CT_T_Rob Pose
        self.inputFile = qt.QLabel(self)
        self.inputFile.enabled = True
        # self.inputFile.text = "/home/rechtsmedizin/data/experiments122020/T_base_ctimage/T_base_ctimage.txt"
        self.inputFile.text = "~/Documents/Base_T_CT.txt"
        self.inputFile = qt.QLineEdit(self.inputFile.text)
        parametersFormLayout.addRow("File Location of Rob_T_CT:", self.inputFile)

        # Load Transformation Button
        self.loadTransformButton = qt.QPushButton("Load New Transformation Matrix from Rob to CT")
        self.loadTransformButton.toolTip = "Load a Point from a specific file location."
        self.loadTransformButton.enabled = True
        parametersFormLayout.addRow(self.loadTransformButton)

        # IP input Field
        self.ConnectOnIP = qt.QLabel(self)
        self.ConnectOnIP.setText("IP of Server")
        self.ConnectOnIP.enabled = True
        self.ConnectOnIP.text = "127.0.0.1"
        self.ConnectOnIP = qt.QLineEdit(self.ConnectOnIP.text)
        parametersFormLayout.addRow("IP of Server:", self.ConnectOnIP)

        # Port input Field
        self.ConnectOnPort = qt.QLabel(self)
        self.ConnectOnPort.setText("Port of Server")
        self.ConnectOnPort.enabled = True
        self.ConnectOnPort.text = "6500:1:6514"
        self.ConnectOnPort = qt.QLineEdit(self.ConnectOnPort.text)
        parametersFormLayout.addRow("Port of Server:", self.ConnectOnPort)

        # TCPIP-Connection Button
        self.connectTCPIP = qt.QPushButton("Connect to KUKA via TCP/IP")
        self.connectTCPIP.toolTip = "Connect to the KUKA Server"
        self.connectTCPIP.enabled = True
        parametersFormLayout.addRow(self.connectTCPIP)

        # Drop Down Points -- Surface
        self.targetMoveItWidget = slicer.qMRMLNodeComboBox()
        self.targetMoveItWidget.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.targetMoveItWidget.selectNodeUponCreation = True
        self.targetMoveItWidget.addEnabled = False
        self.targetMoveItWidget.removeEnabled = False
        self.targetMoveItWidget.noneEnabled = False
        self.targetMoveItWidget.showHidden = False
        self.targetMoveItWidget.showChildNodeTypes = False
        self.targetMoveItWidget.setMRMLScene(slicer.mrmlScene)
        self.targetMoveItWidget.setToolTip("Pick a point at the surface.")
        parametersFormLayout.addRow("Surface Point: ", self.targetMoveItWidget)

        # Drop Down Points -- target
        self.surfaceMoveItWidget = slicer.qMRMLNodeComboBox()
        self.surfaceMoveItWidget.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.surfaceMoveItWidget.selectNodeUponCreation = True
        self.surfaceMoveItWidget.addEnabled = False
        self.surfaceMoveItWidget.removeEnabled = False
        self.surfaceMoveItWidget.noneEnabled = False
        self.surfaceMoveItWidget.showHidden = False
        self.surfaceMoveItWidget.showChildNodeTypes = False
        self.surfaceMoveItWidget.setMRMLScene(slicer.mrmlScene)
        self.surfaceMoveItWidget.setToolTip("Pick a target")
        parametersFormLayout.addRow("Target Point: ", self.surfaceMoveItWidget)

        # Eval Button
        self.checkPathMoveIt = qt.QPushButton("Execute Puncture.")
        self.checkPathMoveIt.toolTip = "Check if Robot can drive the desired path."
        self.checkPathMoveIt.enabled = True
        parametersFormLayout.addRow(self.checkPathMoveIt)

        # connections
        self.applySegmentButton.connect('clicked(bool)', self.onApplySegmentButton)
        self.cancelButton.connect('clicked(bool)', self.onCancelButton)
        self.loadTransformButton.connect('clicked(bool)', self.onloadTransformButton)
        self.connectTCPIP.connect('clicked(bool)', self.onconnectTCPIP)
        self.checkPathMoveIt.connect('clicked(bool)', self.oncheckPathMoveIt)
        self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
        self.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
        self.targetNameWidget.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
        self.targetMoveItWidget.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
        self.surfaceMoveItWidget.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
        self.typeCombobox.connect("currentIndexChanged(int)", self.onTypeChanged)

        # Add vertical spacer
        self.layout.addStretch(1)
        self.logic = None
        self.working = False
        # Refresh Apply button state
        self.onSelect()

        self.initTransform = True
        self.socketReceiveSend = None
        self.matrix = vtk.vtkMatrix4x4()
        self.matrix.Identity()
        self.matrix__IJK_T_RAS = vtk.vtkMatrix4x4()

    def createSliderWidgetAndDescription(self, name, tooltip, start=0., min=0., max=1., step=0.01, visible=False):
        widget = ctk.ctkSliderWidget()
        widget.singleStep = step
        widget.minimum = min
        widget.maximum = max
        widget.value = start
        widget.setToolTip(tooltip)
        widget.setVisible(visible)
        label = qt.QLabel(name)
        label.setVisible(visible)
        return label, widget

    def updateProgress(self, value=None):
        if value is None:
            slicer.app.processEvents()
        elif self.progress.value != value:
            self.progress.setValue(value)
            slicer.app.processEvents()

    def onDone(self):
        self.working = False
        self.cancelButton.enabled = False
        self.applySegmentButton.enabled = self.inputSelector.currentNode() and self.outputSelector.currentNode() and self.targetNameWidget.currentNode() \
                                          and not self.working

    def cleanup(self):
        if self.socketReceiveSend is not None:
            for s in self.socketReceiveSend:
                s.close()
        self.socketReceiveSend = None

    def onReload(self):
        self.cleanup()
        super().onReload()
        import importlib
        importlib.reload(BestPathVisualizationLib)

    def onTypeChanged(self, index):
        if index == self.typeCombobox.count - 1:
            for widget in self.conditionalWidgets:
                widget.setVisible(True)
        else:
            for widget in self.conditionalWidgets:
                widget.setVisible(False)

    def onSelect(self):
        self.applySegmentButton.enabled = self.inputSelector.currentNode() and self.outputSelector.currentNode() and self.targetNameWidget.currentNode() \
                                          and not self.working
        self.updateProgress(0)
        # self.applyFillButton.enabled = False

    # Define Action when TCPIP Connection Button is pressed
    def onconnectTCPIP(self):
        port_texts = self.ConnectOnPort.text.split(",")
        ips = self.ConnectOnIP.text.split(",")
        if len(ips) == 1:
            ips = ips*len(port_texts)
        if len(ips) != len(port_texts):
            raise Exception("Not the same number of ports and IPs specified! (Or 1 IP if all on the same machine)")
        addresses = []
        for ip, port_text in zip(ips, port_texts):
            ports_text = port_text.split(":")
            step = 1
            if len(ports_text) == 3:
                step = int(ports_text[1])
            addresses += [(ip, int(p)) for p in range(int(ports_text[0]), int(ports_text[-1]) + 1, step)]

        try:
            if self.socketReceiveSend is not None:
                for s in self.socketReceiveSend:
                    s.close()
                sleep(2)
            self.socketReceiveSend = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in addresses]
            for (ip, port), s in zip(addresses, self.socketReceiveSend):
                s.connect((ip, port))
            self.connected = True
            print('Connection Established!')
        except Exception as e:
            print(f'Server not reachable! ({e})')
            for s in self.socketReceiveSend:
                s.close()
            self.connected = False

    def onApplySegmentButton(self):
        if self.logic is None:
            self.logic = BestPathVisualizationLib.logic.BestPathVisualizationLogic(self.distanceSliderWidget.value,
                                                                                   self.imageThresholdSliderWidget.value,
                                                                                   self.colorbarSliderWidget.value,
                                                                                   self.discreetStepsSliderWidget.value,
                                                                                   self.socketReceiveSend, self.matrix)
            self.logic.updateCallback = self.updateProgress
            self.logic.doneCallback = self.onDone

        self.working = True
        self.onSelect()
        self.cancelButton.enabled = True
        slicer.app.processEvents()
        self.logic.maxDistance = self.distanceSliderWidget.value
        self.logic.imageThreshold = self.imageThresholdSliderWidget.value
        self.logic.socketReceiveSend = self.socketReceiveSend
        self.logic.matrix = self.matrix
        self.logic.outputPath = self.outputFileSelector.currentPath
        self.logic.useHoleFilling = self.fillHolesCheckBox.checked
        self.logic.colorbarValue = self.colorbarSliderWidget.value
        self.logic.discreetStepsValue = self.discreetStepsSliderWidget.value
        self.logic.maxKernelSize = self.maxKernelSliderWidget.value
        self.logic.distanceWeighting = self.distanceWeightSliderWidget.value
        self.logic.angleWeighting = self.angleWeightSliderWidget.value
        if self.typeCombobox.currentIndex != 6:
            self.logic.overlayTypeIndex = self.typeCombobox.currentIndex
            self.logic.runSegment(self.inputSelector.currentNode(), self.outputSelector.currentNode(),
                                  self.targetNameWidget.currentNode())
        else:
            path = self.logic.outputPath
            for idx in range(6):
                self.logic.overlayTypeIndex = idx
                if path is not '':
                    if ".txt" in path:
                        self.logic.outputPath = path.replace(".txt", "_" + str(idx) + ".txt")
                    else:
                        self.logic.outputPath = path + "_" + str(idx)
                if not self.logic.runSegment(self.inputSelector.currentNode(), self.outputSelector.currentNode(),
                                             self.targetNameWidget.currentNode()):
                    break
        self.onDone()

    def onCancelButton(self):
        self.logic.cancel = True
        self.onDone()

    # Define Action when Load Button is pressed
    def onloadTransformButton(self):
        # transformMatrixNP = np.loadtxt("/home/rechtsmedizin/data/experiments122020/T_base_ctimage/T_base_ctimage.txt", delimiter=',')
        transformMatrixNP = np.loadtxt(os.path.expanduser(self.inputFile.text), delimiter=',')
        print("Loaded Transfomation Matrix Rob_T_CT")
        print(transformMatrixNP)
        for i in range(0, 4):
            for j in range(0, 4):
                self.matrix.SetElement(i, j, transformMatrixNP[i, j])
        if self.initTransform == True:
            self.transformNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode', 'Rob_T_CT')
            self.transformNode.SetMatrixTransformToParent(self.matrix)
            self.initTransform = False
        self.transformNode.SetMatrixTransformToParent(self.matrix)

    def oncheckPathMoveIt(self):
        if self.connected:
            targetMoveIt = self.targetMoveItWidget.currentNode()
            self.targetGlobalMoveIt = [0, 0, 0]
            try:
                targetMoveIt.GetMarkupPoint(0, 0, self.targetGlobalMoveIt)
                self.validPoint1 = True
            except:
                print('No Target Selected!')
                self.validPoint1 = False
            surfaceMoveIt = self.surfaceMoveItWidget.currentNode()
            self.surfaceGlobalMoveIt = [0, 0, 0]
            try:
                surfaceMoveIt.GetMarkupPoint(0, 0, self.surfaceGlobalMoveIt)
                self.validPoint2 = True
            except:
                print('No Surface Point Selected!')
                self.validPoint2 = False
        if self.validPoint1 & self.validPoint2:
            MoveItSuccess = self.checkTwoPointsInMoveItDisplay(self.surfaceGlobalMoveIt, self.targetGlobalMoveIt)
            print(MoveItSuccess)
            print(f"Moveit Result for Trajectory is {MoveItSuccess}")
            if MoveItSuccess[0]:
                try:
                    print("Waiting for completion")
                    recv_data = self.socketReceiveSend[0].recv(102400)
                    msg = recv_data.decode('utf-8')
                    print(f"Received: {msg}")
                    if "Aborting" in msg:
                        print("Aborted by user in Control Node")
                    else:
                        # msg = list(recv_data.decode('utf-8'))
                        log_file_name = os.path.expanduser('~/Documents/Robot_Data/slicer_data/') + ''.join([msg])
                        print(log_file_name)
                        node = self.inputSelector.currentNode()
                        storageNode = node.GetStorageNode()
                        if storageNode is not None:
                            filepath = storageNode.GetFullNameFromFileName()
                        else:
                            UIDs = node.GetAttribute("DICOM.instanceUIDs").split()
                            filepath = slicer.dicomDatabase.fileForInstance(UIDs[0])
                        item = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(
                            slicer.mrmlScene).GetItemByDataNode(node)
                        patient = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(
                            slicer.mrmlScene).GetItemAncestorAtLevel(
                            item, 'Patient')
                        patientID = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(
                            slicer.mrmlScene).GetItemAttribute(
                            patient, 'DICOM.PatientID')
                        matrix_array = np.eye(4)
                        for r in range(4):
                            for c in range(4):
                                matrix_array[r, c] = self.matrix.GetElement(r, c)
                        log_dict = {
                            'Path': filepath,
                            'PatientID': patientID,
                            'TMatrix': matrix_array.tolist(),
                            'PointA': self.surfaceGlobalMoveIt,
                            'PointB': self.targetGlobalMoveIt

                        }
                        dump_file = open(log_file_name, "w")
                        json.dump(log_dict, dump_file)
                        dump_file.close()
                except:
                    print('Did not receive answer from server!')
        if not self.connected:
            print("No Server Connection!")

    def checkTwoPointsInMoveItDisplay(self, PointA, PointB):
        # Get Transforms
        narray = np.eye(4)
        for r in range(4):
            for c in range(4):
                narray[r, c] = self.matrix.GetElement(r, c)

        IJK_T_RAS = np.eye(4)
        IJK_T_RAS[0, 0] = -1
        IJK_T_RAS[1, 1] = -1

        # Apply Transforms to Points
        Point_A_RAS = IJK_T_RAS.dot(np.append(PointA, 1))
        Point_B_RAS = IJK_T_RAS.dot(np.append(PointB, 1))

        Point_A_Rob = narray.dot(Point_A_RAS)
        Point_B_Rob = narray.dot(Point_B_RAS)

        sendArray = np.ones((7))
        sendArray[0] = 0  # 0 -> Real Perform, 1 -> Simulate
        sendArray[1:4] = Point_A_Rob[0:3]
        sendArray[4:7] = Point_B_Rob[0:3]

        # Convert to Float
        print(sendArray)

        self.socketReceiveSend[0].send(str(sendArray.tolist()).encode())
        # Receive Answer
        try:
            # Receive echoed data
            # recv_data = self.socketReceiveSend.recv(102400)
            # if np.all(np.array(eval(recv_data.decode('utf-8'))) == sendArray):
            #  print('Transfer OK')

            # Receive list of valid points
            recv_data = self.socketReceiveSend[0].recv(102400)
            msg = list(recv_data.decode('utf-8'))
            moveIt_result_boolean = [bool(int(i)) for i in msg]
            return moveIt_result_boolean
        except:
            print('Did not receive answer from server!')
            return 3


class BestPathVisualizationTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        # self.test_BestPathVisualization1()

    def test_BestPathVisualization1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")
        #
        # first, get some data
        #
        import urllib
        downloads = (
            ('http://slicer.kitware.com/midas3/download?items=5767', 'FA.nrrd', slicer.util.loadVolume),
        )

        for url, name, loader in downloads:
            filePath = slicer.app.temporaryPath + '/' + name
            if not os.path.exists(filePath) or os.stat(filePath).st_size == 0:
                logging.info('Requesting download %s from %s...\n' % (name, url))
                urllib.urlretrieve(url, filePath)
            if loader:
                logging.info('Loading %s...' % (name,))
                loader(filePath)
        self.delayDisplay('Finished with download and loading')

        volumeNode = slicer.util.getNode(pattern="FA")
        logic = BestPathVisualizationLib.logic.BestPathVisualizationLogic()
        self.delayDisplay('Test passed!')

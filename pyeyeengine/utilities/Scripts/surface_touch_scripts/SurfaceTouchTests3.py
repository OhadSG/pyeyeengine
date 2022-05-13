import vtk
import numpy as np
from numpy import random
import pyeyeengine.utilities.global_params as Globals
from pyeyeengine.utilities.Scripts.camera_scripts.LocalFrameManager import LocalFrameManager
from pyeyeengine.utilities.Scripts.general_scripts.LocalCalibration import LocalCalibration
from pyeyeengine.point_cloud.point_cloud import PointCloud
from pyeyeengine.utilities.astra_orbbec_conf import AstraOrbbec

class VtkPointCloud:

    def __init__(self, zMin=0.0, zMax=1000.0, maxNumPoints=1e6):
        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        self.clearPoints()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(zMin, zMax)
        mapper.SetScalarVisibility(1)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.GetProperty().SetPointSize(1)
        self.vtkActor.SetMapper(mapper)

    def addPoint(self, point):
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            self.vtkDepth.InsertNextValue(point[2])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
        else:
            r = random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()

    def clearPoints(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')

class AddPointCloudTimerCallback():
    def __init__(self, renderer, iterations):
        self.iterations = iterations
        self.renderer = renderer
        self.frame_manager = LocalFrameManager(rgb_resolution=Globals.RGB_MEDIUM_QUALITY)
        self.test_counter = 0

    def execute(self, iren, event):
        self.test_counter += 1
        print(self.test_counter)

        for actor in self.renderer.GetActors():
            self.renderer.RemoveActor(actor)

        point_cloud_raw = self.get_point_cloud()
        pointCloud = VtkPointCloud()
        self.renderer.AddActor(pointCloud.vtkActor)
        pointCloud.clearPoints()

        # if self.test_counter < 10:
        for point_data in point_cloud_raw:
            pointCloud.addPoint(point_data)

        iren.GetRenderWindow().Render()
        self.renderer.ResetCamera()

    def get_point_cloud(self):
        depth_frame = self.frame_manager.get_depth_frame()
        conf = AstraOrbbec(scale_factor=Globals.CAMERA_SCALE_FACTOR)
        pcl = PointCloud(depth_frame, camera_conf=conf)

        filtered_points = pcl.point_cloud[
            np.sign(np.dot(pcl.point_cloud, (0, 0, 0)) + (100 - 50)) !=
            np.sign(np.dot(pcl.point_cloud, (0, 0, 0)) + (100 - 100))]

        # fig2, ax2 = plt.subplots()
        # ax2.scatter(filtered_points[:, 0], filtered_points[:, 2], c='g', cmap="Greens")
        # fig2.show()

        return pcl.point_cloud

if __name__ == "__main__":
    # Renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(.0, .0, .0)
    renderer.ResetCamera()

    camera = vtk.vtkCamera()
    camera.SetPosition(0, 250, -200)
    camera.SetFocalPoint(0, 1, 0)
    camera.SetViewUp(1, 1, 1)
    renderer.SetActiveCamera(camera)

    # Render Window
    renderWindow = vtk.vtkRenderWindow()

    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(2000, 2000)

    # Interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderWindowInteractor.Initialize()

    # Initialize a timer for the animation
    addPointCloudTimerCallback = AddPointCloudTimerCallback(renderer, 2)
    renderWindowInteractor.AddObserver('TimerEvent', addPointCloudTimerCallback.execute)
    timerId = renderWindowInteractor.CreateRepeatingTimer(16)
    addPointCloudTimerCallback.timerId = timerId

    # Begin Interaction
    renderWindow.Render()
    renderWindowInteractor.Start()
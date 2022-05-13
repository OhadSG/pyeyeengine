import cv2
import numpy as np
import vtk
import matplotlib.pyplot as plt
import pyeyeengine.utilities.global_params as Globals
from pyeyeengine.utilities.Scripts.camera_scripts.LocalFrameManager import LocalFrameManager
from pyeyeengine.utilities.Scripts.general_scripts.LocalCalibration import LocalCalibration
from pyeyeengine.point_cloud.point_cloud import PointCloud
from pyeyeengine.utilities.astra_orbbec_conf import AstraOrbbec

def depth_map_2_binary(depth_map, lower=0, higher=255):
    return cv2.inRange(depth_map, lower, higher)  # -255, -5)

def filter_noise(binary_image):
    return cv2.dilate(cv2.medianBlur(cv2.erode(binary_image, None, 3), 3), None, 5)

def get_display_corners_on_cam(needed_resolution=(1280, 800)):
    factor = 1280 / needed_resolution[0]
    top_right = np.expand_dims(self.transfrom_points_display_to_cam(np.array([[1280, 0]])), axis=1)
    top_left = np.expand_dims(self.transfrom_points_display_to_cam(np.array([[0, 0]])), axis=1)
    bottom_right = np.expand_dims(
        self.transfrom_points_display_to_cam(np.array([[1280, 800]])), axis=1)
    bottom_left = np.expand_dims(
        self.transfrom_points_display_to_cam(np.array([[0, 800]])), axis=1)
    return bottom_left / factor, bottom_right / factor, top_left / factor, top_right / factor

def adjusted_depth(depth_image):
    # from https://orbbec3d.com/product-astra/
    # Astra S has a depth in the range 0.35m to 2.5m
    maxDepth = 2500
    minDepth = 350 # in mm

    result_image = depth_image

    for j in range(0, result_image.shape[0]):
        for i in range(0, result_image.shape[1]):
            if result_image[j][i] is not None:
                result_image[j][i] = maxDepth - (result_image[j][i] - minDepth)

    return result_image

def process_depth():
    while True:
        images = {}

        depth_frame = frame_manager.get_depth_frame()
        images['depth'] = depth_frame

        inverted = (255 - depth_frame)
        images['inverted'] = inverted

        binary = depth_map_2_binary(depth_frame, 0, 10)
        images['binary'] = binary

        filtered = filter_noise(binary)
        images['filtered'] = filtered

        adapted = cv2.adaptiveThreshold(inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        np_horiz1 = np.hstack((depth_frame, inverted))
        np_horiz2 = np.hstack((adapted, filtered))
        np_grid = np.vstack((np_horiz1, np_horiz2))

        cv2.imshow("Images", np_grid)
        cv2.waitKey(1)

        # for key in images.keys():
        #     cv2.imshow(key, images[key])
        #     cv2.waitKey(1)

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def process_depth_deep():
    while True:
        images = {}

        depth_frame = frame_manager.get_depth_frame()
        rgb_frame = frame_manager.get_rgb_frame()

        blended = cv2.addWeighted(rgb_frame, 1.0, depth_frame, 1.0, 0)

        cv2.imshow("Depth Deep", blended)
        cv2.waitKey(1)

class SurfaceTouchTester():
    def __init__(self):
        self.frame_manager = LocalFrameManager(rgb_resolution=Globals.RGB_MEDIUM_QUALITY)
        self.calibrator = LocalCalibration()
        self.points = None
        self.vertices = None
        self.point = None
        self.mapper = None
        self.actor = None
        self.renderer = None
        self.renderWindow = None
        self.renderWindowInteractor = None

    def calibrate(self):
        result = self.calibrator.calibrate(frame_manager=frame_manager)
        print("Calibration Result: {}".format(result))
        return result

    def init_vtk(self):
        self.point = vtk.vtkPolyData()
        self.mapper = vtk.vtkPolyDataMapper()

        if vtk.VTK_MAJOR_VERSION <= 5:
            self.mapper.SetInput(self.point)
        else:
            self.mapper.SetInputData(self.point)

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetPointSize(1)
        self.renderer = vtk.vtkRenderer()
        self.renderWindow = vtk.vtkRenderWindow()
        self.renderWindow.AddRenderer(self.renderer)
        self.renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        self.renderWindowInteractor.SetRenderWindow(self.renderWindow)
        self.renderWindow.SetSize(1000, 1000)
        self.renderer.AddActor(self.actor)
        self.renderWindowInteractor.Initialize()
        self.renderWindowInteractor.AddObserver("TimerEvent", self.render_frame)
        dt = 30  # ms
        timer_id = self.renderWindowInteractor.CreateRepeatingTimer(dt)
        self.renderWindowInteractor.Start()

    def render_frame(self, obj = None, event = None):
        print("Update")
        point_cloud = self.get_point_cloud()

        self.points = vtk.vtkPoints()
        self.vertices = vtk.vtkCellArray()

        for point_data in point_cloud:
            id = self.points.InsertNextPoint(point_data)
            self.vertices.InsertNextCell(1)
            self.vertices.InsertCellPoint(id)

        self.point.SetPoints(self.points)
        self.point.SetVerts(self.vertices)
        self.points.Modified()
        self.vertices.Modified()
        self.renderWindow.Render()

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

if __name__ == '__main__':
    # engine = SurfaceTouchTester()
    # engine.init_vtk()


    print("Done")

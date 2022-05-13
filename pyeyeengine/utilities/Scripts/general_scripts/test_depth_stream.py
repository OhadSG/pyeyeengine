from pyeyeengine.utilities.file_uploader import FileUploader
from pyeyeengine.camera_utils.camera_manager import CameraManager
from pyeyeengine.camera_utils.frame_manager import FrameManager
import pyeyeengine.utilities.global_params as Globals

if __name__ == '__main__':
    # camera = CameraManager()
    # depth = camera.get_depth(res_xy=(640, 480))
    # FileUploader.upload_image(depth, "camera_manager_frame.png")

    FrameManager.getInstance().start()
    FrameManager.set_depth_resolution(Globals.DEPTH_HIGH_QUALITY)
    depth = FrameManager.getInstance().get_depth_frame()
    FileUploader.upload_image(depth, "frame_manager_frame.png")
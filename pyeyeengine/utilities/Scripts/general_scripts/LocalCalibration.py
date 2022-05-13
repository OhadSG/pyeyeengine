import pyeyeengine.utilities.global_params as Globals
from pyeyeengine.utilities.Scripts.camera_scripts.LocalFrameManager import LocalFrameManager

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
SAVE_RESULT = True

class LocalCalibration():
    def __init__(self):
        self.calibration_scale_factor = Globals.CALIBRATION_SCALE_FACTOR
        self.warp_mat_cam_2_displayed = None

    def compare_projection_with_template(self, projected, template, factor=1):
        projected_gray = cv2.cvtColor(projected, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        orbBF = cv2.ORB_create(MAX_MATCHES)
        kp1, des1 = orbBF.detectAndCompute(projected_gray, None)
        kp2, des2 = orbBF.detectAndCompute(template_gray, None)

        bfMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, True)
        matchesBF = bfMatcher.match(des1, des2)
        matchesBF = sorted(matchesBF, key=lambda x: x.distance)
        numGoodMatches = int(len(matchesBF) * GOOD_MATCH_PERCENT)
        matchesBF = matchesBF[:numGoodMatches]

        points1 = np.zeros((len(matchesBF), 2), dtype=np.float32)
        points2 = np.zeros((len(matchesBF), 2), dtype=np.float32)

        for i, match in enumerate(matchesBF):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        try:
            homography, status = cv2.findHomography(points1, points2, method=cv2.RANSAC, ransacReprojThreshold=7, maxIters=10000, confidence=0.99)
        except:
            return 0.0, None, None

        if CALIBRATION_DEBUG:
            print("Homography:\n{}".format(homography))

        height, width, channels = template.shape
        warped_image = cv2.warpPerspective(projected, homography, (width, height))

        template_confidence = cv2.matchTemplate(template, warped_image, method=cv2.TM_CCOEFF_NORMED)
        ransac_confidence = np.sum(status) / len(matchesBF)

        points1 /= factor
        homography, status = cv2.findHomography(points1, points2, method=cv2.RANSAC, ransacReprojThreshold=7,
                                                maxIters=10000, confidence=0.99)

        if SAVE_RESULT:
            matches_image = cv2.drawMatches(projected, kp1, template, kp2, matchesBF, None, flags=2)
            cv2.imwrite(FOUND_MATHCHES_IMAGE_NAME, matches_image)
            pheight, pwidth, _ = projected.shape
            projected_resized = cv2.resize(projected, (int(pwidth / factor), int(pheight / factor)))
            result_image = cv2.warpPerspective(projected_resized, homography, (width, height))
            new_path = "{}/local_calibration_images/homography.jpg".format(BASE_PATH)
            cv2.imwrite(new_path, result_image)
            print("Homography Confidence: [RANSAC: {} Template: {}]".format(ransac_confidence * 100, template_confidence * 100))

        return template_confidence[0][0], warped_image, homography

    def find_homography(self, cam_image, template):
        template_confidence, final_warp, final_homography = self.compare_projection_with_template(cam_image, template, index=1, factor=self.calibration_scale_factor)
        return template_confidence, final_homography, final_warp

    def calibrate(self, frame_manager=None, template=None):
        if frame_manager is None or not isinstance(frame_manager, LocalFrameManager):
            return False

        template_image = template if template is not None else BASE_PATH + "../../../calibration/ORB_SINGLE_IMAGE/calibration_template.jpg"

        frame = frame_manager.get_rgb_frame()

        cv2.imwrite(BASE_PATH + "/local_calibration_images/reference_rgb_frame.png", frame)

        template_confidence, homography, warped = self.find_homography(frame, template_image)

        if template_confidence > 0.35:
            Log.d("Final Homography:\n{}".format(homography))

            np.save(HOMOGRAPHY_FILE_NAME, homography)
            self.warp_mat_cam_2_displayed = homography

            Log.i("[CALIBRATION] New Calibration Successful", extra_details={"confidence":"{:.1f}".format(template_confidence * 100)})
            return True
        else:
            Log.e("[CALIBRATION] New Calibration Failed", extra_details={"confidence":"{:.1f}".format(template_confidence * 100)})
            return False
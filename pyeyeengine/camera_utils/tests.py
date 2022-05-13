import cv2
import numpy as np
from scipy import ndimage

## shape detector
from pyeyeengine.calibration.find_max_playing_mask import find_optimal_playing_screen
from pyeyeengine.object_detection.shape_detector import ShapeDetector

shapeDetector = ShapeDetector()
# 1) circle radius is properly calculated :
theta = np.arange(360).reshape((360, 1))/360*(2*3.14)
contour = np.int32(np.round(np.expand_dims( np.concatenate([np.cos(theta), np.sin(theta)], axis=1)*10+np.array([10,10]), axis=1)))
contour_center, r = shapeDetector.fit_circle_to_contour(contour)
results = contour_center[0,0]==10 and contour_center[0,1]==10 and round(r)==10
assert(results)


## export calibration
# test  find_optimal_playing_screen
results = True
for angle in range(10,360,10):
    img = np.zeros((200,200))
    img[50:150, 50:100] = 1
    rotated = cv2.dilate(np.uint8(ndimage.rotate(img, angle))*255, None, None)
    if (rotated.sum() / (img.sum()*255)) > .9:
        _, contours, _ = cv2.findContours(rotated, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        bounding_box = find_optimal_playing_screen(contours[0])
        playing_mask = np.repeat( np.expand_dims(np.zeros_like(rotated), axis=2), 3, axis=2)
        playing_mask = cv2.rectangle(playing_mask, (int(bounding_box[0]), int(bounding_box[1])),
                                     (int(bounding_box[0] + bounding_box[2]), int(bounding_box[1] + bounding_box[3])),
                                     (255, 255, 255), -1)[:, :, 0]


        results = results * ((playing_mask*rotated).sum() <= playing_mask.sum()) * ((playing_mask*(rotated==0)).sum() <= 255*4)# is playing mask fully fitted in rotated?
        print(results)
        cv2.imshow("out_of_rotates", playing_mask * (rotated == 0))
        rotated[playing_mask > 0] = 0
        cv2.imshow("playing_mask", playing_mask)
        cv2.imshow("rotated", rotated)
        cv2.waitKey(0)
assert(results)

#
# # contour_line_intersection
# img = np.zeros((200,200))
# img[50:100, 50:200] = 1
# rotated = cv2.dilate(np.uint8(ndimage.rotate(img, 25))*255, None, None)
# _, contours, _ = cv2.findContours(rotated, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
#
# rect = cv2.minAreaRect(contours[0])
# box = cv2.boxPoints(rect)
# box = np.int0(box)
# rotated_rgb = cv2.drawContours(np.tile(np.expand_dims(rotated, axis=2), (1,1,3)),[box],0,(0,0,255),2)
# cv2.imshow("rotated", rotated_rgb)
# cv2.waitKey(0)
# hi=5
#



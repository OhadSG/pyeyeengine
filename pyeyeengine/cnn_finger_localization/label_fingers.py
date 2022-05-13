import glob
import csv

import cv2

ix,  iy = -1, -1
def mouse_callback(event,x,y,flags,param):
    global ix,iy
    if event == cv2.EVENT_LBUTTONDBLCLK:
        ix,iy = x,y
        print("x: ", x,", y: ",y)



image_paths = glob.glob("./finger_images/*png")
cv2.namedWindow('select_finger')
cv2.setMouseCallback('select_finger',mouse_callback)
with open('example4.csv', 'w') as csvfile:
    fieldnames = ['img_path', 'x', 'y']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for img_path in  image_paths:
        img = cv2.imread(img_path)
        cv2.imshow("select_finger", cv2.resize(img,(0,0), fx=5, fy=5))
        cv2.waitKey(0)
        writer.writerow({'img_path': img_path, 'x': ix/5, 'y': iy/5})
        ix, iy = -1, -1









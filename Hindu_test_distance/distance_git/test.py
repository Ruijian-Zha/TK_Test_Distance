'''
-------------------------------------------
-    Author: Asadullah Dal                -
-    =============================        -
-    Company Name: AiPhile                -
-    =============================        -
-    Purpose : Youtube Channel            -
-    ============================         -
-    Link: https://youtube.com/c/aiphile  -
-------------------------------------------
'''

import cv2
from object_detector import *
import numpy as np

# variables
# distance from camera to object(face) measured
KNOWN_DISTANCE = 20.1 # or 20.5 centimeter 
KNOWN_DISTANCE_2 = 20.4 # or 20.5 centimeter 
# width of face in the real world or Object Plane
KNOWN_WIDTH = 2.5  # centimeter

# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
fonts = cv2.FONT_HERSHEY_COMPLEX

# cap = cv2.VideoCapture(0)
cap = cv2.imread("Ref_1.bmp")

# face detector object
# face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# focal length finder function
def focal_length(measured_distance, real_width, width_in_rf_image):
    '''
    This Function Calculate the Focal Length(distance between lens to CMOS sensor), it is simple constant we can find by using 
    MEASURED_DISTACE, REAL_WIDTH(Actual width of object) and WIDTH_OF_OBJECT_IN_IMAGE 
    :param1 Measure_Distance(int): It is distance measured from object to the Camera while Capturing Reference image
    :param2 Real_Width(int): It is Actual width of object, in real world (like My face width is = 14.3 centimeters)
    :param3 Width_In_Image(int): It is object width in the frame /image in our case in the reference image(found by Face detector) 
    :retrun focal_length(Float):
    '''

    focal_length_value = (width_in_rf_image * measured_distance) / real_width
    return focal_length_value

# distance estimation function
def distance_finder(focal_length, real_face_width, face_width_in_frame):
    '''
    This Function simply Estimates the distance between object and camera using arguments(focal_length, Actual_object_width, Object_width_in_the_image)
    :param1 focal_length(float): return by the focal_length_Finder function
    :param2 Real_Width(int): It is Actual width of object, in real world (like My face width is = 5.7 Inches)
    :param3 object_Width_Frame(int): width of object in the image(frame in our case, using Video feed)
    :return Distance(float) : distance Estimated  
    '''

    distance = (real_face_width * focal_length)/face_width_in_frame
    return distance

# face detector function 
def face_data(image):
    '''
    This function Detect the face 
    :param Takes image as argument.
    :returns face_width in the pixels
    '''

    # face_width = 0
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    # for (x, y, h, w) in faces:
    #     cv2.rectangle(image, (x, y), (x+w, y+h), WHITE, 1)
    #     face_width = w

    parameters = cv2.aruco.DetectorParameters_create()
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)


    # Load Object Detector
    detector = HomogeneousBgDetector()

    # Load Image
    # img = cv2.imread("phone_aruco_marker.jpg")
    img = cv2.imread(image)
    img = img[1025:2058, 400:1800]
    # print(img.shape)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv2.imshow("img",img)
    # cv2.waitKey(0)

    # Get Aruco marker
    corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

    # Draw polygon around the marker
    int_corners = np.int0(corners)
    cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

    # Aruco Perimeter
    # aruco_perimeter = cv2.arcLength(corners[0], True)

    # Pixel to cm ratio
    # pixel_cm_ratio = aruco_perimeter / 20

    contours = detector.detect_objects(img)

    result = 100000

    # Draw objects boundaries
    for cnt in contours:
        # Get rect
        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), angle = rect

        # skip small things
        if w <= 200 or h <=200 :
            continue

        result = min(result, h)
        result = min(result, w)

        # Get Width and Height of the Objects by applying the Ratio pixel to cm
        # object_width = w / pixel_cm_ratio
        # object_height = h / pixel_cm_ratio

        # Display rectangle
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.polylines(img, [box], True, (255, 0, 0), 2)
        cv2.putText(img, "Width {} px".format(w), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 1, (100, 200, 0), 1)
        cv2.putText(img, "Height {} px".format(h), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 1, (100, 200, 0), 1)

    return result


# reading reference image from directory
# ref_image = cv2.imread("Ref_0.bmp")

ref_image_face_width = face_data("Ref_0.bmp")
ref_image_face_width_2 = face_data("Ref_1.bmp")
focal_length_found = focal_length(
    KNOWN_DISTANCE, KNOWN_WIDTH, ref_image_face_width)
print(focal_length_found)
focal_length_found_2 = focal_length(
    KNOWN_DISTANCE_2, KNOWN_WIDTH, ref_image_face_width_2)
print(focal_length_found_2)
focal_length_found = (focal_length_found + focal_length_found_2)/2
# cv2.imshow("ref_image", ref_image)

# while True:

# _, frame = cap.read()

# calling face_data function
face_width_in_frame = face_data("Test_2.bmp")

# finding the distance by calling function Distance
if face_width_in_frame != 0:
    Distance = distance_finder(
        focal_length_found, KNOWN_WIDTH, face_width_in_frame)

print(Distance)
        
# Drwaing Text on the screen
    # cv2.putText(
    #     frame, f"Distance = {round(Distance,2)} CM", (50, 50), fonts, 1, (WHITE), 2)

# cv2.imshow("frame", frame)
# if cv2.waitKey(1) == ord("q"):
#     break

# cap.release()    




cv2.destroyAllWindows()

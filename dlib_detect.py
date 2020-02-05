#Import required modules
import cv2
import dlib

#Dlib positions
#  ("mouth", (48, 68)),
#	("right_eyebrow", (17, 22)),
#	("left_eyebrow", (22, 27)),
#	("right_eye", (36, 42)),
#	("left_eye", (42, 48)),
#	("nose", (27, 35)),
#	("jaw", (0, 17))
#Set up some required objects
video_capture = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()  #Face detector
#Landmark identifier. Set the filename to whatever you named the downloaded file
predictor = dlib.shape_predictor(
    'classifiers/shape_predictor_68_face_landmarks.dat')

while True:
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 180)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)

    detections = detector(clahe_image, 1)  #Detect the faces in the image

    for k, d in enumerate(detections):  #For each detected face
        shape = predictor(clahe_image, d)  #Get coordinates
        for i in range(1, 68):  #There are 68 landmark points on each face
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y),
                       1, (0, 255, 0),
                       thickness=-1)
    cv2.imshow("image", frame)
    #Exit program when the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
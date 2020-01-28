import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# Blue color
    low_blue = np.array([94, 80, 2])
    high_blue = np.array([126, 255, 255])
    blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
    blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

    # Show every color except white
    low = np.array([0, 65, 0])
    high = np.array([255, 255, 255])
    mask = cv2.inRange(hsv_frame, low, high)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Frame", frame)
    cv2.imshow("Blue", blue)
    cv2.imshow("Result", result)

    #Sair do programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
    	break
        
video_capture.release()
cv2.destroyAllWindows()

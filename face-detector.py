import cv2

# loading Pre-trained data on face front face from opencv it uses haar cascade algo
trained_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


webcam_image = cv2.VideoCapture(0)

while True:

    frame_confirm, frame = webcam_image.read()

    # converting to grayscale
    grayscaled_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # Detexting face coordinates
    face_coordinates = trained_data.detectMultiScale(grayscaled_image)

    # # forming a rectacngle around the face
    for (x,y,width,height) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+width,y+height),(0,255,0),2)

    cv2.imshow("Face Detector", frame)
    
    # assigning key pressed so that we can quit using specific key
    key = cv2.waitKey(1)

    if key==81 or key == 113:
        break
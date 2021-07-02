# importing OpenCV library if not present then run
# pip install opencv-python
import cv2

# loading Pre-trained data on face front face from opencv it uses haar cascade algo
Trained_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# for an image file
# image = cv2.imread("image_file_name.jpg/png")

# for Real time Video
webcam_image = cv2.VideoCapture(0)

# for a video file
# webcam_image = cv2.VideoCapture("video_file_name.mp4")

# while loop so that video capture runs continuously
while True:

    # Read the current frame
    frame_confirm, frame = webcam_image.read()

    # converting to grayscale (important step)
    grayscaled_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detexting face coordinates
    face_coordinates = Trained_data.detectMultiScale(grayscaled_image)

    # forming a rectacngle around the face/faces
    for (x, y, width, height) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)

    # Viewing the Video Feed
    cv2.imshow("Akash's Face Detector", frame)

    # assigning key pressed so that we can quit using specific key
    key = cv2.waitKey(1)

    # press q or Q to quit from the video capture window
    if key == 81 or key == 113:
        break

# releasing webcam as it's a good practice
webcam_image.release()
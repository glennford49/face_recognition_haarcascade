# make sure to give name of sample in line 7
import cv2, sys, numpy, os
size = 4
classifier = 'haarcascade_frontalface_default.xml'
image_dir = 'images'
try:
    name_class = "glenn" # name of person for recognition
except:
    print("You must provide a name")
    sys.exit(0)
path = os.path.join(image_dir, name_class)
if not os.path.isdir(path):
    os.mkdir(path)
(im_width, im_height) = (112, 92)
haar_cascade = cv2.CascadeClassifier(classifier)
webcam = cv2.VideoCapture(0)

# Generate name for image file
pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path)
     if n[0]!='.' ]+[0])[-1] + 1

# Beginning message
print("\n\033[94mThe program will save 20 samples. \
Move your head around to increase while it runs.\033[0m\n")

# The program loops until it has 20 images of the face.
count = 0
pause = 0
count_max = 20   # desired number of sample per class
while count < count_max:

    # Loop until the camera is working
    rval = False
    while(not rval):
        # Put the image from the webcam into 'frame'
        (rval, frame) = webcam.read()
        if(not rval):
            print("Failed to open webcam. Trying again...")

    # Get image size
    height, width, channels = frame.shape

    # Flip frame
    frame = cv2.flip(frame, 1, 0)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Scale down for speed
    mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

    # Detect faces
    faces = haar_cascade.detectMultiScale(mini)

    # We only consider largest face
    faces = sorted(faces, key=lambda x: x[3])
    if faces:
        face_i = faces[0]
        (x, y, w, h) = [v * size for v in face_i]

        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))

        # Draw rectangle and write name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, name_class, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,
            1,(0, 255, 0))

        # Remove false positives
        if(w * 6 < width or h * 6 < height):
            print("Face too small")
        else:

            # To create diversity, only save every fith detected image
            if(pause == 0):

                print("Saving training sample "+str(count+1)+"/"+str(count_max))

                # Save image file
                cv2.imwrite('%s/%s.png' % (path, pin), face_resize)

                pin += 1
                count += 1

                pause = 1

    if(pause > 0):
        pause = (pause + 1) % 5
    cv2.imshow('Sampling', frame)
    key = cv2.waitKey(10)
    if key == 27:
        break

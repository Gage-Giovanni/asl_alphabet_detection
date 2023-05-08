import os
import time
import string
from dotenv import load_dotenv
import cv2

load_dotenv()

# load webcam id from .env into cv2 capture
webcam_id = os.getenv('WEBCAM_ID')
webcam = cv2.VideoCapture(int(webcam_id))

# create all needed image directories
image_folder = 'Tensorflow\\workspace\\images\\take_images'
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

if not os.path.exists(image_folder):
    os.mkdir(image_folder)

for label in labels:
    path = os.path.join(image_folder, label)
    if not os.path.exists(path):
        os.mkdir(path)

# gather images based on user label and number input
label = ''
while label != '!':
    label = input('Enter letter to be captured: ')
    numImgs = input('Number of images to be captured: ')
    for i in range(int(numImgs)):
        time.sleep(3)
        ret, frame = webcam.read()
        imagename = label + '{}.jpg'.format(str(i))
        image = os.path.join(image_folder, label, imagename)
        cv2.imwrite(image, frame)
        cv2.imshow('frame', frame)
        time.sleep(2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

webcam.release()
cv2.destroyAllWindows()
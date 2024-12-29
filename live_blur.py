import cv2
import cv2.data
import numpy as np
import scipy
import scipy.ndimage
from scipy import signal

PATH_XML = 'haarcascade_frontalface_default.xml'
video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + PATH_XML)


def generate_kernel(kernel_len=5, desvio_padrao=5):

  generate_kernel1d = signal.windows.gaussian(kernel_len, std=desvio_padrao).reshape(kernel_len, 1)
  generate_kernel2d = np.outer(generate_kernel1d, generate_kernel1d)

  return generate_kernel2d

kernel = generate_kernel()

# kernel_title = np.tile(kernel, (3,1,1))

kernel_sum = kernel.sum()
kernel =  kernel / kernel_sum

while True:
  ret, frame = video.read()

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1, 
    minNeighbors=5, 
    minSize=(30,30),
    flags=cv2.CASCADE_SCALE_IMAGE
  )

  for x, y, w, h in faces:
    frame[y:y+h, x:x+w] = cv2.GaussianBlur(frame[y:y+h, x:x+w], (63,63), sigmaX=10, sigmaY=10)
    # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # roi_gray = gray[y:y+h, x:x+w]

    cv2.imshow('frame', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

video.release()
cv2.destroyAllWindows()

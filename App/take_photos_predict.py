import cv2
import numpy as np
import pickle
from skimage import io
from skimage.transform import resize
import time
from CNN.layers import predict

def take_photos():
 cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
 suc, img=cam.read()

 face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
 faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

 if (len(faces_detected) == 0):
  print("No face detected")
  eye_state = 0

 else:
  (x, y, w, h) = faces_detected[0]
  cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1);

  for (x, y, w, h) in faces_detected:
     cv2.imwrite('App/Images/crop.jpg', img[y + 1:y + h , x + 1:x + w ])

  img = cv2.imread('App/Images/crop.jpg')

  height, weight, channels = img.shape

  y=height//5
  x=weight//10
  w = weight//2 - x
  h = height//2 - y
  cv2.imwrite('App/Images/1.jpg', img[y:y + h, x:x + w])
  cv2.imwrite('App/Images/2.jpg', img[y: y + h, x + w:(weight - weight//10)])
  cam.release()
  cv2.destroyAllWindows()

  save_path = 'App/parameters.pkl'
  params, cost = pickle.load(open(save_path, 'rb'))
  [f1, f2, w3, w4, b1, b2, b3, b4] = params

  img = io.imread('App/Images/1.jpg', as_gray=True)
  img = resize(img, (32, 32))
  np.array(img)
  img = img.reshape(1,32,32)
  digit, probability = predict(img, params)
  print("first eye: ")
  eye1 = int(digit)
  p = int(probability * 100)

  if (eye1 == 1):
      print("open")
  elif (eye1 == 0):
      print("closed")

  print("probability:")
  print(p,"%")

  img = io.imread('App/Images/2.jpg', as_gray=True)
  img = resize(img, (32, 32))
  np.array(img)
  img = img.reshape(1,32,32)
  digit, probability = predict(img, params)
  print("second eye: ")

  eye2 = int(digit)
  p = int(probability * 100)

  if (eye2 == 1):
      print("open")
  elif (eye2 == 0):
      print("closed")

  print("probability:")
  print(p,"%")

  if((eye1 == 0)and(eye2 == 0)):
   eye_state = 0
  elif((eye1 == 1)and(eye2 == 1)):
   eye_state = 1
  else:
   eye_state = 2

 return eye_state

def assume(time_interval):
 counter = 0
 photo_num = 0
 while(counter < 3):
  photo_num = photo_num + 1
  print("Sleeping counter: ",counter)
  print()
  print("Photo number: ", photo_num)
  current_state = take_photos()
  if(current_state == 0):
   counter = counter + 1
   print("Both closed, counter++")
  elif(current_state == 1):
   counter = 0
   print("Both open, counter is nulled")
  else:
   print("different prediction, counter stays the same")

  time.sleep(time_interval)



# import cv2
# import numpy as np
# from keras.models import load_model
#
# model = load_model('glassesDetector.h5')
#
# results = {0: 'without glasses', 1: 'glasses'}
# GR_dict = {0: (0, 0, 255), 1: (0, 255, 0)}
#
# rect_size = 4
# cap = cv2.VideoCapture(0)
#
# haarcascade = cv2.CascadeClassifier(
#     'C:/Users/Romio/AppData/Local/Programs/Python/Python38/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
#
#
# while True:
#     (rval, im) = cap.read()
#     im = cv2.flip(im, 1, 1)
#
#     rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
#     faces = haarcascade.detectMultiScale(rerect_size)
#     for f in faces:
#         (x, y, w, h) = [v * rect_size for v in f]
#
#         face_img = im[y:y + h, x:x + w]
#         rerect_sized = cv2.resize(face_img, (160, 160))
#         normalized = rerect_sized / 255.0
#         reshaped = np.reshape(normalized, (1, 160, 160, 3))
#         reshaped = np.vstack([reshaped])
#         result = model.predict(reshaped)
#
#         label = np.argmax(result, axis=1)[0]
#
#         cv2.rectangle(im, (x, y), (x + w, y + h), GR_dict[label], 2)
#         cv2.rectangle(im, (x, y - 40), (x + w, y), GR_dict[label], -1)
#         cv2.putText(im, results[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#
#     cv2.imshow('LIVE', im)
#     key = cv2.waitKey(10)
#
#     if key == 27:
#         break
#
# cap.release()
#
# cv2.destroyAllWindows()



#
# from __future__ import print_function
# import cv2 as cv
# import argparse
# def detectAndDisplay(frame):
#     frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     frame_gray = cv.equalizeHist(frame_gray)
#     #-- Detect faces
#     faces = face_cascade.detectMultiScale(frame_gray)
#     for (x,y,w,h) in faces:
#         center = (x + w//2, y + h//2)
#         frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
#         faceROI = frame_gray[y:y+h,x:x+w]
#         #-- In each face, detect eyes
#         eyes = eyes_cascade.detectMultiScale(faceROI)
#         for (x2,y2,w2,h2) in eyes:
#             eye_center = (x + x2 + w2//2, y + y2 + h2//2)
#             radius = int(round((w2 + h2)*0.25))
#             frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
#     cv.imshow('Capture - Face detection', frame)
# parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
# parser.add_argument('--face_cascade', help='Path to face cascade.', default='data/haarcascades/haarcascade_frontalface_alt.xml')
# parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
# parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
# args = parser.parse_args()
# face_cascade_name = args.face_cascade
# eyes_cascade_name = args.eyes_cascade
# face_cascade = cv.CascadeClassifier()
# eyes_cascade = cv.CascadeClassifier()
#
# if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
#     print('--(!)Error loading face cascade')
#     exit(0)
# if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
#     print('--(!)Error loading eyes cascade')
#     exit(0)
# camera_device = args.camera
# cap = cv.VideoCapture(camera_device)
# if not cap.isOpened:
#     print('--(!)Error opening video capture')
#     exit(0)
# while True:
#     ret, frame = cap.read()
#     if frame is None:
#         print('--(!) No captured frame -- Break!')
#         break
#     detectAndDisplay(frame)
#     if cv.waitKey(10) == 27:
#         break




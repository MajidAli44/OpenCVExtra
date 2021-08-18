import cv2 as cv
import os
import numpy as np

people = ["Ben", "Elton", "Jerry", "Madonna", "Mindy"]

DIR = r'/home/majid/Pictures/Face'

cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR,person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_gray = cv.imread(img_path)
            gray = cv.cvtColor(img_gray, cv.COLOR_BGR2GRAY)

            face_rect = cascade.detectMultiScale(gray, 1.1,3)

            for (x,y,w,h) in face_rect:
                face_region = gray[y:y+h, x:x+w]
                features.append(face_region)
                labels.append(label)


create_train()
print("Training Done --------------")


features = np.array(features, dtype="object")
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features,labels)
face_recognizer.save("face_trained.yml")

np.save("features.npy", features)
np.save("labels.npy", labels)



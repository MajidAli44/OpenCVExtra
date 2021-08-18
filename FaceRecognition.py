import cv2 as cv

people = ["Ben", "Elton", "Jerry", "Madonna", "Mindy"]

cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.read("face_trained.yml")

img = cv.imread("/home/majid/Pictures/val/madonna/4.jpg")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)


face_rect = cascade.detectMultiScale(gray,1.1,3)

for (x,y,w,h) in face_rect:
    face_region = gray[y:y+h, x:x+w]

    label , confidence = face_recognizer.predict(face_region)
    print(f"label = {people[label]} with the confidence of {confidence}")

    cv.putText(img, str(people[label]), (20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),3)
    cv.rectangle(img, (x,y),(x+w, y+h),(0,255,255),3)

cv.imshow("face detect", img)
cv.waitKey(0)
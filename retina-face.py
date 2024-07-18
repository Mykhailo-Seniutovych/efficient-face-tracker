from retinaface import RetinaFace
import cv2
import time

# RetinaFace.build_model()
faces = RetinaFace.detect_faces(img_path="data/small-one-face.jpg")
# face = faces['face_1']['facial_area']
# print(faces)
faces = RetinaFace.detect_faces(img_path="data/small-one-face.jpg")
start = time.time()
faces = RetinaFace.detect_faces(img_path="data/small-one-face.jpg")
end = time.time()
print((end - start) * 1000)
img = cv2.imread("data/small-one-face.jpg")
cv2.rectangle(img, (243, 151), (256, 167), (255, 0, 0), 1, 1)
cv2.imshow("lol", img)
cv2.waitKey()

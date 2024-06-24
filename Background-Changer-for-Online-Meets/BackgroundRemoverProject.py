import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import mediapipe as mp
import time

# Ganti dengan jalur absolut ke folder BackgroundRemoverImages
absolute_path = r'D:\MATKUL IK-1A\SEMESTER 6\PENGOLAHAN CITRA\BACKROUND-REMOVAL\Background-Changer-for-Online-Meets\BackgroundRemoverImages'

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
segmentor = SelfiSegmentation()

# Menggunakan metode manual untuk menghitung FPS
pTime = 0

listImg = os.listdir(absolute_path)
print(listImg)

imgList = []

# Membaca dan mengubah ukuran gambar latar belakang agar sesuai dengan ukuran kamera
for imgPath in listImg:
    img = cv2.imread(f'{absolute_path}/{imgPath}')
    img = cv2.resize(img, (640, 480))  # Ubah ukuran gambar latar belakang ke 640x480
    imgList.append(img)
print(len(listImg))

indexImg = 1
while True:
    success, img = cap.read()
    if not success:
        break
    
    # Menghapus latar belakang tanpa argumen threshold
    imgOut = segmentor.removeBG(img, imgList[indexImg])

    # Menghitung FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    imgStacked = cvzone.stackImages([img, imgOut], 2, 1)
    cv2.putText(imgStacked, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    print(indexImg)
    cv2.imshow("Image", imgStacked)

    key = cv2.waitKey(1)
    if key == ord('a'):
        if indexImg > 0:
            indexImg -= 1
    elif key == ord('d'):
        if indexImg < len(imgList) - 1:
            indexImg += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
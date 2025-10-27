import cv2
import os
import imutils

# Definir la identificacion de los rostros

personName = 'Uvuwev'

# Definir la carpeta donde se almacenaran los rostros

dataPath = 'Data'
personPath = dataPath + '/' + personName

if not os.path.exists(personPath):
    print(f'Carpeta creada: {personPath}...!')
    os.makedirs(personPath)

# Definir la captura de video desde una camara

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Definir el modelo

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Definir la identificacion de los rostros des de la caputra

count = 0

while True:

    ret, frame = cap.read()
    if ret == False: break
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count), rostro)
        count += 1

    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27 or count >= 300:
        break

cap.release()
cv2.destroyAllWindows()
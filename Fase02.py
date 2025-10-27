# Definir las librerias

import cv2
import os
import numpy as np

# Definir la ruta donde se almacenan las imagenes para el entrenamiento
dataPath = 'Data'
peopleList = os.listdir(dataPath)
print(f'Lista de personas: {peopleList}.')

# Definir las variables donde se almacenaran las imagenes y las etiquetas

labels = []
facesData = []
label = 0

# Definir el entranamiento del modelo

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print(f'Leyendo las imagenes...')

    for fileName in os.listdir(personPath):
        print(f'Rostro: {nameDir}/{fileName}')
        labels.append(label)
        facesData.append(cv2.imread(personPath + '/' + fileName, 0))

    label += 1

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

print('Entrenando el modelo...')

face_recognizer.train(facesData, np.array(labels))

face_recognizer.write('modeloLBPHFace.xml')

print('Modelo entrenado con exito...!')

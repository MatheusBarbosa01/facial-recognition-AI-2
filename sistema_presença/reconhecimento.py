import kivy
kivy.require('1.11.1')

from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.floatlayout import FloatLayout
from kivy.core.window import Window
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# COR DA JANELA E TAMANHO
Window.clearcolor = (0, 0.1, 0, 1)
Window.size = (1000, 720)

# DIRETÓRIO DAS IMAGENS
data_path = 'L:/python-recognition-opencv-main/python-recognition-opencv-main/faces/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# TREINAMENTO DE FACES
Training_Data, Labels = [], []
for i, file in enumerate(onlyfiles):
    image_path = join(data_path, onlyfiles[i])
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if images is None:
        continue
    Training_Data.append(images)
    Labels.append(i)

# Verifica se há dados de treinamento
if len(Labels) == 0:
    print("Nenhuma imagem de treinamento encontrada.")
    exit()

# Convertendo para numpy arrays
Labels = np.asarray(Labels, dtype=np.int32)

# Treinamento do modelo LBPH
model = cv2.face.LBPHFaceRecognizer_create()
model.train(Training_Data, Labels)
print("TREINAMENTO EFETUADO")

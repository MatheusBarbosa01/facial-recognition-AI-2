from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.textinput import TextInput
from kivy.graphics import Color, RoundedRectangle
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import os

# Configurações iniciais da janela Kivy
Window.clearcolor = (1, 1, 1, 1)
Window.size = (980, 720)

# Widget Kivy para exibir a câmera
class KivyCV(Image):
    def __init__(self, capture, fps, **kwargs):
        super().__init__(**kwargs)
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faceCascade = cv2.CascadeClassifier(r'L:\python-recognition-opencv-main\.venv\Lib\haarcascade_frontalface_default.xml')
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            buf = cv2.flip(frame, 0).tostring()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.texture = image_texture

# Aplicação principal Kivy
class Sistema(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(TelaInicial(name='telaInicial'))
        sm.add_widget(TelaFuncao(name='telaFuncao'))
        return sm
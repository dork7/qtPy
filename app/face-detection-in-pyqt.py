import sys
from os import path

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui

import os


class RecordVideo(QtCore.QObject):
    image_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, camera_port=0, parent=None):
        super().__init__(parent)
        self.camera = cv2.VideoCapture(camera_port)

        self.timer = QtCore.QBasicTimer()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    def start_recording(self):
        self.timer.start(0, self)

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return

        read, data = self.camera.read()
        if read:
            self.image_data.emit(data)


class FaceDetectionWidget(QtWidgets.QWidget):
    def __init__(self, haar_cascade_filepath, parent=None):
        super().__init__(parent)
        maskNet = load_model("../mask_detector.model")
        self.classifier = cv2.CascadeClassifier(haar_cascade_filepath)
        self.image = QtGui.QImage()
        self._green = (0, 255, 0)
        self._width = 2
        self._min_size = (30, 30)

    def detect_faces(self, frame):
        # frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25,
        #                    interpolation=cv2.INTER_AREA)
        (h, w) = frame.shape[:2]
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.equalizeHist(gray_image)

        det_faces = self.classifier.detectMultiScale(gray_image,
                                                     scaleFactor=1.3,
                                                     minNeighbors=4,
                                                     flags=cv2.CASCADE_SCALE_IMAGE,
                                                     minSize=self._min_size)
        # print(faces)
        # return faces
        faces = []
        faces_tf = []
        locs = []
        preds = [[0, 0]]
        print(det_faces)
        for (x, y, w, h) in det_faces:

            startX = x
            startY = y
            endX = w
            endY = h
            locs.append((startX, startY, endX, endY))
            face = frame[startY:startY + endY, startX:startX+endX]
            faces.append(face)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces_tf.append(face)

        return locs, det_faces, faces_tf

    def image_data_slot(self, image_data):
        locs, faces, faces_tf = self.detect_faces(image_data)
        for (x, y, w, h) in faces:
            cv2.rectangle(image_data,
                          (x, y),
                          (x+w, y+h),
                          self._green,
                          self._width)
            preds_tf = self.detect_mask(faces_tf, self.maskNet)

        self.image = self.get_qimage(image_data)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()

    def get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage

        image = QImage(image.data,
                       width,
                       height,
                       bytesPerLine,
                       QImage.Format_RGB888)

        image = image.rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()

    def detect_mask(faces, maskNet):
        preds = None
        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=1)[0]
        return preds


class MainWidget(QtWidgets.QWidget):
    def __init__(self, haarcascade_filepath, parent=None):
        super().__init__(parent)
        fp = haarcascade_filepath
        self.face_detection_widget = FaceDetectionWidget(fp)

        # TODO: set video port
        self.record_video = RecordVideo()

        image_data_slot = self.face_detection_widget.image_data_slot
        self.record_video.image_data.connect(image_data_slot)

        layout = QtWidgets.QVBoxLayout()

        layout.addWidget(self.face_detection_widget)
        self.run_button = QtWidgets.QPushButton('Start')
        layout.addWidget(self.run_button)

        self.run_button.clicked.connect(self.record_video.start_recording)
        self.setLayout(layout)


def main(haar_cascade_filepath):
    app = QtWidgets.QApplication(sys.argv)

    main_window = QtWidgets.QMainWindow()
    main_widget = MainWidget(haar_cascade_filepath)
    main_window.setCentralWidget(main_widget)
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    script_dir = path.dirname(path.realpath(__file__))
    cascade_filepath = path.join(script_dir,
                                 '..',
                                 'data',
                                 'haarcascade_frontalface_default.xml')

    cascade_filepath = path.abspath(cascade_filepath)
    main(cascade_filepath)

import cv2 as cv
import numpy as np
import time
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from typing import Callable


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, cap: cv.VideoCapture, processing: Callable = None):
        super().__init__()
        self.cap = cap
        self.stp = False
        self.processing = processing
        self.timeByFrame = 1./cap.get(cv.CAP_PROP_FPS)

    def setProcessing(self, processing):
        self.processing = processing

    def stop(self):
        self.stp = True

    def run(self):
        while True:
            if self.stp:
                break
            ret, cv_img = self.cap.read()
            if ret:
                if self.processing is not None:
                    cv_img = self.processing(cv_img)
                self.change_pixmap_signal.emit(cv_img)
            time.sleep(self.timeByFrame)
        self.cap.release()


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.createCentralWidget()
        self.createMenuBar()

        self.image = None
        self.videoThread = None

    def createCentralWidget(self):
        self.setCentralWidget(QLabel())

    def createMenuBar(self):
        m1 = self.menuBar().addMenu('File')
        a = m1.addAction('Open image')
        a.triggered.connect(self.open_image)
        m1.addAction('Open video').triggered.connect(self.open_video)
        m1 = self.menuBar().addMenu('Filter')
        m1.addAction('Clear').triggered.connect(self.clear_filter)
        m2 = m1.addMenu('Blur')
        a = m2.addAction('Gaussian')
        a.triggered.connect(self.gaussian_blur)

    def clear_filter(self):
        if self.videoThread is not None:
            self.videoThread.setProcessing(None)

    def gaussian_blur(self):
        if VideoThread is None:
            self.image = cv.GaussianBlur(self.image, (5, 5), 0, 0)
            self.update_image()
        else:
            self.videoThread.setProcessing(self.video_gaussian_blur)

    def video_gaussian_blur(self, img: np.ndarray) -> np.ndarray:
        return cv.GaussianBlur(img, (5, 5), 0, 0)

    def open_video(self):
        fn, format = QFileDialog.getOpenFileName(
            self, 'Open Video', '.', 'Videos (*.avi *.mkv *.mp4)')
        cap = cv.VideoCapture(fn)
        if self.videoThread is not None:
            self.videoThread.stop()
        self.videoThread = VideoThread(cap)
        self.videoThread.change_pixmap_signal.connect(self.update_video)
        self.videoThread.start()

    def open_image(self):
        fn, format = QFileDialog.getOpenFileName(
            self, 'Open File', '.', 'Images (*.png *.xpm *.jpg)')
        self.image = cv.imread(fn)
        self.update_image()

    def update_video(self, image):
        self.image = image
        self.update_image()

    def update_image(self):
        im = QImage(self.image.data, self.image.shape[1],
                    self.image.shape[0], QImage.Format_RGB888).rgbSwapped()
        self.centralWidget().setPixmap(QPixmap.fromImage(im))

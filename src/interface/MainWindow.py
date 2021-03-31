import cv2 as cv
import numpy as np
import time
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from typing import Callable

from src.filter.unsharp_masking import *


def current_milli_time():
    return round(time.time() * 1000)


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, cap: cv.VideoCapture, processing: Callable = None):
        super().__init__()
        self.cap = cap
        self.stp = False
        self.processing_steps = []
        self.timeByFrame = 1./cap.get(cv.CAP_PROP_FPS)

    def clear_processing_steps(self):
        self.processing_steps = []

    def push_processing_step(self, ps: Callable):
        self.processing_steps.append(ps)

    def stop(self):
        self.stp = True

    def run(self):
        while True:
            if self.stp:
                break
            start = current_milli_time()
            ret, cv_img = self.cap.read()
            if ret:
                for ps in self.processing_steps:
                    cv_img = ps(cv_img)
                self.change_pixmap_signal.emit(cv_img)

            elapsed = current_milli_time() - start
            sl = max(0, self.timeByFrame - (elapsed/1000.))
            time.sleep(sl)
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
        m1.addAction('Connected components').triggered.connect(self.connected_components)
        m2 = m1.addMenu('Blur')
        a = m2.addAction('Gaussian')
        a.triggered.connect(self.gaussian_blur)
        m2 = m1.addMenu('Sharpening')
        a = m2.addAction('Unsharp masking')
        a.triggered.connect(self.unsharp_masking)
        m1.addAction('Background removal').triggered.connect(self.background_removal)

    def clear_filter(self):
        if self.videoThread is not None:
            self.videoThread.clear_processing_steps()

    def connected_components(self):
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        cv.threshold(self.image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU, self.image)
        cv.dilate(self.image, np.ones((5,5), np.uint8), self.image)
        n, self.image = cv.connectedComponents(self.image, self.image, 4, cv.CV_16U)
        self.image *= 5120
        self.update_image(QImage.Format_Grayscale16)

    def gaussian_blur(self):
        if self.videoThread is None:
            self.image = cv.GaussianBlur(self.image, (5, 5), 0, 0)
            self.update_image()
        else:
            self.videoThread.push_processing_step(self.video_gaussian_blur)

    def background_removal(self):
        self.fgbg = cv.createBackgroundSubtractorKNN()
        if self.videoThread is None:
            fgmask = self.fgbg.apply(self.image)
            self.image = cv.bitwise_and(self.image, self.image, mask=fgmask)
            self.update_image(cv.Format_RGB32)
        else:
            self.videoThread.push_processing_step(self.video_background_removal)

    def video_background_removal(self, img):
        fgmask = self.fgbg.apply(img)
        return cv.bitwise_and(img, img, mask=fgmask)

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

    def unsharp_masking(self):
        if self.videoThread is None:
            unsharp_masking(self.image)
            self.update_image()
        else:
            self.videoThread.push_processing_step(unsharp_masking)

    def update_video(self, image: np.ndarray):
        self.image = image
        self.update_image()

    def update_image(self, format = QImage.Format_RGB888):
        im = QImage(self.image.data, self.image.shape[1],
                    self.image.shape[0], format).rgbSwapped()
        self.centralWidget().setPixmap(QPixmap.fromImage(im))

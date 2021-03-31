import cv2 as cv
import numpy as np
import time
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from typing import Callable

from src.filter.unsharp_masking import *
from src.filter.background_removal import BackgroundRemoval


def current_milli_time():
    return round(time.time() * 1000)


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, path: str, processing: Callable = None):
        super().__init__()
        self.filename = path
        self.cap = cv.VideoCapture(path)
        self.stp = False
        self.processing_steps = []
        self.timeByFrame = 1./self.cap.get(cv.CAP_PROP_FPS)
        self.psa = False

    def clear_processing_steps(self):
        self.processing_steps = []

    def push_processing_step(self, ps: Callable):
        self.processing_steps.append(ps)

    def stop(self):
        self.stp = True

    def pause(self):
        self.psa = not self.psa

    def restart(self):
        self.cap = cv.VideoCapture(self.filename)

    def run(self):
        while True:
            if self.stp:
                break
            if self.psa:
                time.sleep(0.1)
            else:
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
        self.background = None

    
    def contextMenuEvent(self, event):
        contextMenu = QMenu(self) 
        pause = contextMenu.addAction("pause")
        restart = contextMenu.addAction("restart")
        action = contextMenu.exec_(self.mapToGlobal(event.pos()))
        if action == pause and self.videoThread is not None:
            self.videoThread.pause()
        if action == restart and self.videoThread is not None:
            self.videoThread.restart()

    def createCentralWidget(self):
        self.setCentralWidget(QLabel())

    def createMenuBar(self):
        m1 = self.menuBar().addMenu('File')
        a = m1.addAction('Open image')
        a.triggered.connect(self.open_image)
        m1.addAction('Open video').triggered.connect(self.open_video)
        m1.addAction('Open image as reference').triggered.connect(
            self.open_image_reference)
        m1.addAction('Open video as reference').triggered.connect(
            self.open_video_reference)
        m1 = self.menuBar().addMenu('Filter')
        m1.addAction('Clear').triggered.connect(self.clear_filter)
        m2 = m1.addMenu('Blur')
        a = m2.addAction('Gaussian')
        a.triggered.connect(self.gaussian_blur)
        m2 = m1.addMenu('Sharpening')
        a = m2.addAction('Unsharp masking')
        a.triggered.connect(self.unsharp_masking)
        m1.addAction('Background removal 1').triggered.connect(
            self.background_removal)
        m1.addAction('Background removal 2').triggered.connect(
            self.background_removal2)

    def clear_filter(self):
        if self.videoThread is not None:
            self.videoThread.clear_processing_steps()

    def gaussian_blur(self):
        if self.videoThread is None:
            self.image = cv.GaussianBlur(self.image, (5, 5), 0, 0)
            self.update_image()
        else:
            self.videoThread.push_processing_step(self.video_gaussian_blur)

    def background_removal(self):
        self.fgbg = cv.createBackgroundSubtractorKNN()
        if self.videoThread is not None:
            self.videoThread.push_processing_step(
                self.video_background_removal)

    def background_removal2(self):
        if self.background is not None:
            bck = BackgroundRemoval(self.background)
            if self.videoThread is None:
                self.image = bck.background_removal(self.image)
                self.update_image()
            else:
                self.videoThread.push_processing_step(bck.background_removal)
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Must select a background as reference")
            msg.exec_()


    def video_background_removal(self, img):
        fgmask = self.fgbg.apply(img)
        return cv.bitwise_and(img, img, mask=fgmask)

    def video_gaussian_blur(self, img: np.ndarray) -> np.ndarray:
        return cv.GaussianBlur(img, (5, 5), 0, 0)

    def open_video(self):
        fn, format = QFileDialog.getOpenFileName(
            self, 'Open Video', '.', 'Videos (*.avi *.mkv *.mp4)')
        if self.videoThread is not None:
            self.videoThread.stop()
        self.videoThread = VideoThread(fn)
        self.videoThread.change_pixmap_signal.connect(self.update_video)
        self.videoThread.start()

    def open_image(self):
        fn, format = QFileDialog.getOpenFileName(
            self, 'Open File', '.', 'Images (*.png *.xpm *.jpg)')
        self.image = cv.imread(fn)
        self.update_image()

    def open_image_reference(self):
        fn, format = QFileDialog.getOpenFileName(
            self, 'Open File', '.', 'Images (*.png *.xpm *.jpg)')
        self.background = cv.imread(fn)

    def open_video_reference(self):
        fn, format = QFileDialog.getOpenFileName(
            self, 'Open Video', '.', 'Videos (*.avi *.mkv *.mp4)')
        x, self.background = cv.VideoCapture(fn).read()

    def unsharp_masking(self):
        if self.videoThread is None:
            unsharp_masking(self.image)
            self.update_image()
        else:
            self.videoThread.push_processing_step(unsharp_masking)

    def update_video(self, image: np.ndarray):
        self.image = image
        self.update_image()

    def update_image(self):
        im = QImage(self.image.data, self.image.shape[1],
                    self.image.shape[0], QImage.Format_RGB888).rgbSwapped()
        self.centralWidget().setPixmap(QPixmap.fromImage(im))

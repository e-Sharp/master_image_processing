import time
import cv2 as cv
import numpy as np

from typing import Callable

from PyQt5.QtCore import QThread, pyqtSignal


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
                        try:
                            cv_img = ps(cv_img)
                        except:
                            print("Error on ", ps)
                    self.change_pixmap_signal.emit(cv_img)

                elapsed = current_milli_time() - start
                sl = max(0, self.timeByFrame - (elapsed/1000.))
                time.sleep(sl)
        self.cap.release()

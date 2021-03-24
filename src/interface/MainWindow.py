import cv2 as cv
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.createCentralWidget()
        self.createMenuBar()

        self.image = None
        
    def createCentralWidget(self):
        self.setCentralWidget(QLabel())

    def createMenuBar(self):
        m1 = self.menuBar().addMenu('File')
        a = m1.addAction('Open')
        a.triggered.connect(self.open_image)
        m1 = self.menuBar().addMenu('Filter')
        m2 = m1.addMenu('Blur')
        a = m2.addAction('Gaussian')
        a.triggered.connect(self.gaussian_blur)

    def gaussian_blur(self):
        self.image = cv.GaussianBlur(self.image, (5, 5), 0, 0)
        self.update_image()

    def open_image(self):
        fn, format = QFileDialog.getOpenFileName(self, 'Open File', '.', 'Images (*.png *.xpm *.jpg)')
        self.image = cv.imread(fn)
        self.update_image()

    def update_image(self):
        im = QImage(self.image.data, self.image.shape[1], self.image.shape[0], QImage.Format_RGB888).rgbSwapped()
        self.centralWidget().setPixmap(QPixmap.fromImage(im))

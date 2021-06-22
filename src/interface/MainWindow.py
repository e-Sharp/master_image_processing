import copy
import cv2 as cv
import numpy as np
import time
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt

from typing import Callable

from src.filter.unsharp_masking import *
from src.filter.background_removal import BackgroundRemoval
from src.interface.RegionGrowing import RegionGrowing
from src.interface.VideoThread import VideoThread
from src.utils.groupType import GroupType


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.references = {}
        self.buffers = {}
        self.selectedBuffer = 'src'
        
        self.createCentralWidget()
        self.createMenuBar()

        self.image = None
        self.videoThread = None
        self.background = None

    def contextMenuEvent(self, event):
        contextMenu = QMenu(self)
        pause = contextMenu.addAction("Pause")
        restart = contextMenu.addAction("Restart")
        stopPlacing = contextMenu.addAction("Stop Placing")
        clearSeeds = contextMenu.addAction("Clear Seeds")
        action = contextMenu.exec_(self.mapToGlobal(event.pos()))
        if action == pause and self.videoThread is not None:
            self.videoThread.pause()
        if action == restart and self.videoThread is not None:
            self.videoThread.restart()
        if action == stopPlacing:
            QApplication.restoreOverrideCursor()
            self.placing = False
        if action == clearSeeds:
            self.mapSeeds.clear()


    def createCentralWidget(self):
        self.setCentralWidget(QLabel())
        self.centralWidget().mousePressEvent = self.getPos

    def createMenuBar(self):
        m1 = self.menuBar().addMenu('File')
        a = m1.addAction('Open image')
        a.triggered.connect(self.open_image)
        m1.addAction('Open video').triggered.connect(self.open_video)
        m1.addAction('Open image as reference').triggered.connect(
            self.open_image_reference)
        m1.addAction('Open video as reference').triggered.connect(
            self.open_video_reference)
        self.references['viewMenu'] = self.menuBar().addMenu('View')
        m1 = self.menuBar().addMenu('Filter')
        m1.addAction('Clear').triggered.connect(self.clear_filter)
        m2 = m1.addMenu('Blur')
        a = m2.addAction('Gaussian')
        a.triggered.connect(self.gaussian_blur)
        m2 = m1.addMenu('Sharpening')
        a = m2.addAction('Unsharp masking')
        a.triggered.connect(self.unsharp_masking)
        m2 = m1.addMenu('Noise')
        a = m2.addAction('Add_noise')
        a.triggered.connect(self.add_noise)
        a = m2.addAction('Denoise')
        a.triggered.connect(self.denoising)

        m3 = m1.addMenu('Background removal')
        m3.addAction('KNN').triggered.connect(
            self.background_removal)
        m3.addAction('Perso').triggered.connect(
            self.background_removal2)
        m1 = self.menuBar().addAction('Region Growing').triggered.connect(self.regionGrowing)

        m10 = self.menuBar().addMenu('Body Parts')
        m11 = m10.addMenu('Place seeds')
        m11.addAction('Left arm').triggered.connect(lambda: self.placeSeeds(GroupType.LEFT_ARM))
        m11.addAction('Right arm').triggered.connect(lambda: self.placeSeeds(GroupType.RIGHT_ARM))
        m11.addAction('Head').triggered.connect(lambda: self.placeSeeds(GroupType.HEAD))
        m11.addAction('Torso').triggered.connect(lambda: self.placeSeeds(GroupType.TORSO))
        m11.addAction('Left leg').triggered.connect(lambda: self.placeSeeds(GroupType.LEFT_LEG))
        m11.addAction('Right leg').triggered.connect(lambda: self.placeSeeds(GroupType.RIGHT_LEG))
        self.placing = False
        self.currentGroup = None
        self.mapSeeds = {}

    def placeSeeds(self, idGroup):
        print("Place seeds : ", idGroup)
        self.placing = True
        self.currentGroup = idGroup
        QApplication.setOverrideCursor(Qt.CrossCursor)

    def getPos(self, event):
        if event.buttons() & Qt.LeftButton:
            if self.placing:
                x = event.pos().x()
                y = event.pos().y()
                if self.currentGroup not in self.mapSeeds:
                    self.mapSeeds[self.currentGroup] = []
                self.mapSeeds[self.currentGroup].append((x,y))
                print(self.mapSeeds)

    def clear_filter(self):
        if self.videoThread is not None:
            self.videoThread.clear_processing_steps()

    def connected_components(self):
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        cv.threshold(self.image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU, self.image)
        cv.dilate(self.image, np.ones((5, 5), np.uint8), self.image)
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

    def denoising(self):
        self.image = cv.fastNlMeansDenoising(self.image, None, 17, 21, 7)
        self.update_image()

    def add_noise(self):
        #speckle noise
        gauss = np.random.normal(0, 0.5, self.image.size)
        gauss = gauss.reshape(self.image.shape[0], self.image.shape[1], self.image.shape[2]).astype('uint8')
        self.image = cv.add(self.image, gauss)

        '''gauss = np.random.normal(0, 0.5, self.image.size)
        gauss = gauss.reshape(self.image.shape[0], self.image.shape[1], self.image.shape[2]).astype('uint8')
        self.image = cv.add(self.image, gauss)'''
        '''
        for _ in range(20):
            img1 = self.image.copy()
            cv.randn(img1, (1, 1, 1), (2, 2, 2))
            self.image += img1
        # For averaging create an empty array, then add images to this array.
        img_avg = np.zeros((self.image.shape[0], self.image.shape[1], self.image.shape[2]), np.float32)
        for im in self.image:
            img_avg = img_avg + im / 20
            print(im)
        # Round the float values. Always specify the dtype
        #self.image = np.array(np.round(img_avg), dtype=np.uint8)
        print(self.image)
        '''
        self.update_image()

    def video_gaussian_blur(self, img: np.ndarray) -> np.ndarray:
        return cv.GaussianBlur(img, (5, 5), 0, 0)

    def open_video(self):
        fn, format = QFileDialog.getOpenFileName(
            self, 'Open Video', '.', 'Videos (*.avi *.mkv *.mp4)')
        if fn is not None and fn != "":
            if self.videoThread is not None:
                self.videoThread.stop()
            self.videoThread = VideoThread(fn)
            self.videoThread.change_pixmap_signal.connect(self.update_video)
            self.videoThread.start()

    def open_image(self):
        fn, format = QFileDialog.getOpenFileName(
            self, 'Open File', '.', 'Images (*.png *.xpm *.jpg)')
        if fn is not None and fn != "":
            self.buffers['src'] = {
                'data': cv.imread(fn),
                'format': QImage.Format_RGB888}
            self.image = self.buffers['src']['data']
            self.update_image()
        self.updateViewMenu()

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
        self.buffers['src'] = {
            'data'  : image,
            'format': QImage.Format_RGB888}
        self.selectBuffer('src')

    def updateViewMenu(self):
        m = self.references['viewMenu']
        m.clear()
        for (id, buffer) in self.buffers.items():
            def make(id):
                def action():
                    self.selectBuffer(id)
                return action
            m.addAction(id).triggered.connect(make(id))

    def selectBuffer(self, id):
        self.image = self.buffers[id]['data']
        self.selectedBuffer = id
        self.update_image()

    def update_image(self, format=QImage.Format_RGB888):
        im = QImage(self.image.data, self.image.shape[1],
                    self.image.shape[0], self.buffers[self.selectedBuffer]['format']).rgbSwapped()
        self.centralWidget().setPixmap(QPixmap.fromImage(im))

    def regionGrowing(self):
        rg = RegionGrowing(
            self.buffers['src']['data'],
            np.ones(self.buffers['src']['data'].shape[:2]),
            self.mapSeeds)
        rg.saturate()
        print(rg.result())
        
        # self.buffers['boundaries'] = {
        #     'data': np.zeros((w, h), np.uint8),
        #     'format': QImage.Format_Mono}
        # self.buffers['regions'] = {
        #     'data': np.zeros((w, h), np.uint8),
        #     'format': QImage.Format_Mono}
        # self.buffers['visited'] = {
        #     'data': np.zeros((w, h), np.bool_),
        #     'format': QImage.Format_Mono}
        # self.updateViewMenu()
        
            

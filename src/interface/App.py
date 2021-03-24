from src.interface.MainWindow import MainWindow

from PyQt5.QtWidgets import QApplication
import sys

class App:
    def __init__(self):
        app = QApplication(sys.argv)
        mw = MainWindow()
        mw.show()
        sys.exit(app.exec_())

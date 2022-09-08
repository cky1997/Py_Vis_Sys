from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QLabel


class MyLabel(QLabel):
    clicked = pyqtSignal(object)
    mouseDoubleClickSignal = pyqtSignal(object)

    # def mousePressEvent(self, ev):
    #     self.clicked.emit(self)
    def mouseDoubleClickEvent(self, ev):
        self.mouseDoubleClickSignal.emit(self)
        # print('mouse double clicked')
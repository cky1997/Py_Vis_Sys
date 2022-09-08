from functools import reduce

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QTableWidget, QTableWidgetItem, QMessageBox, \
    QAbstractItemView, QVBoxLayout
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap, QIntValidator
import pandas as pd
import numpy as np
import sys
from mainwin import Ui_Form
import os
import random
import tools
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import MiniBatchKMeans
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
# from wcwidget import WcWidget

class MatplotlibWidget(QWidget):

    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.axis("off")
        self.figure.set_facecolor("#EDECEC")


        vertical_layout = QVBoxLayout(self)
        vertical_layout.addWidget(self.canvas)

        # self.canvas.axes = self.canvas.figure.add_subplot(5,1,random.randint(1, 6))
        # self.setLayout(vertical_layout)




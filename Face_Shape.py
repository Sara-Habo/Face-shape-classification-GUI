# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Face_Shape.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


import style
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class MplCanvas(FigureCanvas):

    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1287, 930)
        MainWindow.setStyleSheet("background-color: rgb(252, 242, 255);")
        self.centralwidget = QtWidgets.QLabel(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(900, 10, 321, 321))
        self.label_2.setStyleSheet(
            "background-image: url(:/newPrefix/S5.png);")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 20, 601, 121))
        self.label.setStyleSheet("background-image: url(:/newPrefix/13.png);")
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(30, 160, 781, 121))
        self.label_3.setStyleSheet(
            "background-image: url(:/newPrefix/14.png);")
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.Face_Shape_Label = QtWidgets.QLabel(self.centralwidget)
        self.Face_Shape_Label.setGeometry(QtCore.QRect(50, 300, 751, 571))
        self.Face_Shape_Label.setText("")
        self.Face_Shape_Label.setObjectName("Face_Shape_Label")
        #font = QtGui.QFont()
        #font.setFamily("Perpetua Titling MT")
        #font.setPointSize(18)
        #font.setBold(True)
        #font.setWeight(75)
        #self.label_4 = QtWidgets.QLabel(self.centralwidget)
        #self.label_4.setGeometry(QtCore.QRect(840, 440, 200, 150))
        #self.label_4.setStyleSheet("background-image: url(:/newPrefix/6.png);")
        #self.label_4.setText("Results")
        #self.label_4.setFont(font)
        #self.label_4.setObjectName("label_4")
        #self.label_5 = QtWidgets.QLabel(self.centralwidget)
        #self.label_5.setGeometry(QtCore.QRect(830, 610, 101, 111))
        #self.label_5.setStyleSheet("background-image: url(:/newPrefix/8.png);")
        #self.label_5.setText("")
        #self.label_5.setObjectName("label_5")
        self.label_Model1 = QtWidgets.QLabel(self.centralwidget)
        self.label_Model1.setGeometry(QtCore.QRect(960, 460, 301, 81))
        self.label_Model1.setText("")
        self.label_Model1.setObjectName("label_Model1")
        #self.label_Model2 = QtWidgets.QLabel(self.centralwidget)
        #self.label_Model2.setGeometry(QtCore.QRect(960, 630, 291, 81))
        #self.label_Model2.setText("")
        #self.label_Model2.setObjectName("label_Model2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1287, 26))
        self.menubar.setObjectName("menubar")
        self.menuOPEN = QtWidgets.QMenu(self.menubar)
        self.menuOPEN.setObjectName("menuOPEN")
        MainWindow.setMenuBar(self.menubar)
        self.actionOPEN = QtWidgets.QAction(MainWindow)
        self.actionOPEN.setObjectName("actionOPEN")
        self.menuOPEN.addAction(self.actionOPEN)
        self.menubar.addAction(self.menuOPEN.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menuOPEN.setTitle(_translate("MainWindow", "FILE"))
        self.actionOPEN.setText(_translate("MainWindow", "OPEN"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

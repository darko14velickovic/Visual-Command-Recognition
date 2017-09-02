# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created: Sun Jun 18 16:06:40 2017
#      by: pyside-uic 0.2.15 running on PySide 1.2.4
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(998, 786)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.graphicsView = QtGui.QGraphicsView(self.centralwidget)
        self.graphicsView.setEnabled(True)
        self.graphicsView.setMouseTracking(True)
        self.graphicsView.setAutoFillBackground(False)
        self.graphicsView.setInteractive(True)
        self.graphicsView.setSceneRect(QtCore.QRectF(0.0, 0.0, 480.0, 640.0))
        self.graphicsView.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.graphicsView.setTransformationAnchor(QtGui.QGraphicsView.NoAnchor)
        self.graphicsView.setObjectName("graphicsView")
        self.verticalLayout.addWidget(self.graphicsView)
        self.status_label = QtGui.QLabel(self.centralwidget)
        self.status_label.setObjectName("status_label")
        self.verticalLayout.addWidget(self.status_label)
        self.fov_slider = QtGui.QSlider(self.centralwidget)
        self.fov_slider.setMinimum(32)
        self.fov_slider.setMaximum(200)
        self.fov_slider.setSingleStep(2)
        self.fov_slider.setOrientation(QtCore.Qt.Horizontal)
        self.fov_slider.setObjectName("fov_slider")
        self.verticalLayout.addWidget(self.fov_slider)
        self.learn_button = QtGui.QPushButton(self.centralwidget)
        self.learn_button.setObjectName("learn_button")
        self.verticalLayout.addWidget(self.learn_button)
        self.start_button = QtGui.QPushButton(self.centralwidget)
        self.start_button.setObjectName("start_button")
        self.verticalLayout.addWidget(self.start_button)
        self.stop_button = QtGui.QPushButton(self.centralwidget)
        self.stop_button.setObjectName("stop_button")
        self.verticalLayout.addWidget(self.stop_button)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 998, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionNew_video = QtGui.QAction(MainWindow)
        self.actionNew_video.setObjectName("actionNew_video")
        self.menuFile.addAction(self.actionNew_video)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", "MainWindow", None, QtGui.QApplication.UnicodeUTF8))
        self.status_label.setText(QtGui.QApplication.translate("MainWindow", "TextLabel", None, QtGui.QApplication.UnicodeUTF8))
        self.learn_button.setText(QtGui.QApplication.translate("MainWindow", "Learn", None, QtGui.QApplication.UnicodeUTF8))
        self.start_button.setText(QtGui.QApplication.translate("MainWindow", "Start", None, QtGui.QApplication.UnicodeUTF8))
        self.stop_button.setText(QtGui.QApplication.translate("MainWindow", "Stop", None, QtGui.QApplication.UnicodeUTF8))
        self.menuFile.setTitle(QtGui.QApplication.translate("MainWindow", "File", None, QtGui.QApplication.UnicodeUTF8))
        self.actionNew_video.setText(QtGui.QApplication.translate("MainWindow", "New video", None, QtGui.QApplication.UnicodeUTF8))


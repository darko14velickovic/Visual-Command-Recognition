import sys
from PySide import QtGui, QtCore

import cv2 as cv2
import numpy as np
from datetime import datetime

# import tflearn as tf

from gui.pyui.Ui_MainWindow import Ui_MainWindow


class MainWindow(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.Trainer = None
        self.view_capture = None
        self.currentFrame = None

        self.graphics_scene = QtGui.QGraphicsScene()
        self.graphics_scene.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 255)))

        self.graphicsView.setScene(self.graphics_scene)

        self.pix_item = None
        self.polySnipet = None
        self.start_flag = False
        self.drawing_pen = QtGui.QPen("White")
        self.event_hookup()
        # timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.fov_size = 32

    def window_stack(self, a, stepsize, width):
        n = a.shape[0]
        return np.hstack(a[i:1 + n + i - width:stepsize] for i in range(0, width))

    def FOV(self, ev):
        position = ev.pos()
        xPos = position.x()
        yPos = position.y()
        #self.status_label.setText("X: " + str(xPos) + " Y: " + str(yPos))

        if (self.polySnipet != None):
            try:
                self.graphics_scene.removeItem(self.polySnipet)
            except Exception:
                print (Exception)
        half = self.fov_size / 2
        self.polySnipet = QtGui.QGraphicsRectItem(xPos - half, yPos - half, self.fov_size, self.fov_size)
        # self.polySnipet.setBrush(QtGui.QBrush(QtCore.Qt.green))

        self.polySnipet.setPen(self.drawing_pen)
        self.graphics_scene.addItem(self.polySnipet)

    def fov_size_change(self, val):
        # print val
        self.status_label.setText("FOV size is: " + str(val) + "x" + str(val))
        self.fov_size = val
    def event_hookup(self):

        self.start_button.clicked.connect(lambda: self.start())
        self.stop_button.clicked.connect(lambda: self.stop())
        self.learn_button.clicked.connect(lambda: self.learn())
        self.actionNew_video.triggered.connect(lambda: self.open())

        self.graphicsView.mouseMoveEvent = self.FOV
        self.graphics_scene.mousePressEvent = self.take_a_snapshot
        self.fov_slider.valueChanged.connect(self.fov_size_change)

    def QImageToCvMat(self, incomingImage):
        '''  Converts a QImage into an opencv MAT format  '''

        incomingImage = incomingImage.convertToFormat(QtGui.QImage.Format.Format_RGB32)

        width = incomingImage.width()
        height = incomingImage.height()

        ptr = incomingImage.constBits()
        arr = np.array(ptr).reshape(height, width, 4)  # Copies the data
        return arr

    def take_a_snapshot(self, position):
        # self.showCoords(position.scenePos())
        pos = position.scenePos()
        xCord = int(pos.x())
        yCord = int(pos.y())

        if self.pix_item is not None:
            half = self.fov_size / 2
            pixMap = self.pix_item.pixmap()
            qimg = pixMap.toImage()
            matrix = self.QImageToCvMat(qimg)
            part = matrix[yCord - half: yCord + half, xCord - half: xCord + half]
            now = datetime.now()
            name = str(now.year) + "" + str(now.month) + "" + str(now.day) + "" + str(now.hour) + "" + \
                   str(now.minute) + str(now.second) + ".png"
            cv2.imwrite(name, part)

    def start(self):
        self.timer.start(10)
        # self.actionAutoplay.setText("Stop autoplay")
        print("Timer started!")

    def stop(self):
        self.timer.stop()
        # self.actionAutoplay.setText("Autoplay")
        print("Timer stopped!")

    def learn(self):
        print("MMMM LEARNING")
        inting = 1
        return inting

    def open(self):
        file_name, _ = QtGui.QFileDialog.getOpenFileName(self, "Open File", QtCore.QDir.currentPath())
        if file_name:
            self.view_capture = cv2.VideoCapture(file_name)
            if not self.view_capture.isOpened():
                return

            flag, self.currentFrame = self.view_capture.read()
            if flag:
                self.show_frame(self.currentFrame)

    def next_frame(self):
        if self.view_capture is not None:
            flag, self.currentFrame = self.view_capture.read()
            if flag:
                self.show_frame(self.currentFrame)
            else:
                self.timer.stop()
                print("Timer stopped!")

    def show_frame(self, frame):

        frames = self.window_stack(frame, 8, 60)
        # TODO: Test moving window function and implement ROI (Region of interest) and procesing of the ROI window_stack
        for window in frames:
            print window


        frame = cv2.cvtColor(frame, cv2.cv.CV_BGR2RGB)



        image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_RGB888)
        self.graphics_scene.clear()
        self.graphics_scene.update()

        img = QtGui.QPixmap.fromImage(image)
        self.pix_item = QtGui.QGraphicsPixmapItem(img)

        self.graphics_scene.addItem(self.pix_item)

        self.graphics_scene.update()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    MainWindow = MainWindow()
    MainWindow.show()

    sys.exit(app.exec_())

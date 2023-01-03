
from PyQt5 import QtCore, QtWidgets
from PyQt5 import QtGui
import sys
from PyQt5.QtWidgets import QFileDialog, QLabel
from PyQt5.QtGui import QPixmap
import pyqtgraph
from pyqtgraph import *
import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import numpy as np
from Face_Shape import Ui_MainWindow, MplCanvas
import matplotlib as mpl
import pyqtgraph.exporters
#from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt

from math import sqrt
from PIL import Image
import pyqtgraph.exporters
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from keras.models import load_model
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import dlib
import imutils
from imutils import face_utils
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
matplotlib.use('QT5Agg')


class MatplotlibCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, dpi=120):
        fig = Figure(dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(fig)
        fig.tight_layout()


class mainApp(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(mainApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.actionOPEN.triggered.connect(lambda: self.browseAnImg())
        self.logHistory = []

    def saveLocation(self):
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
                                                  "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*) ")

        return filePath

    def model1(self, path1):

        #new_model = load_model('face-shape-recognizer (2).h5')
        model_inspection=load_model('face-shape-recognizer_inception64.h5')
        model_cnn1=load_model('face-shape-recognizer (2).h5')
        model_cnn2=load_model('face-shape-recognizer_72.8_olddata.h5')


        class_names = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
        class_names_label = {class_name: i for i,
                             class_name in enumerate(class_names)}
        test_datagen = ImageDataGenerator(
            rescale=1./255,
            brightness_range=(0.8, 1.2),
            horizontal_flip=True,
        )
        path = os.path.normpath(path1).split(os.path.sep)
        img = tf.keras.utils.load_img(
            path1,
            color_mode='grayscale',
            target_size=(250, 190),
            interpolation='nearest',
            keep_aspect_ratio=False
        )
        img1 = image.img_to_array(img)
        img1 = img1/255.
        img1 = np.expand_dims(img1, axis=0)
        images = np.vstack([img1])

        img_150 = tf.keras.utils.load_img(
            path1,
            target_size=(150, 150),
            interpolation='nearest',
            keep_aspect_ratio=False
        )
        img2 = image.img_to_array(img_150)
        img2 = img2/255.
        img2 = np.expand_dims(img2, axis=0)

       # x = new_model.predict(img1, verbose=0)
        predictions_inspection = model_inspection.predict_generator(img2, verbose=0)
        predictions_cnn1 = model_cnn1.predict_generator(img1, verbose=0)
        predictions_cnn2 = model_cnn2.predict_generator(img1, verbose=0)

        w_inspection=0.7
        w_cnn1=0.74
        w_cnn2=0.74
        avg_predection=(w_inspection*predictions_inspection+w_cnn1*predictions_cnn1+w_cnn2*predictions_cnn2)/(w_inspection+w_cnn1+w_cnn2)
        y_pred = np.argmax(avg_predection, axis = 1)[0] # We take the highest probability
        #y_pred = np.argmax(new_model.predict(img1, verbose=0), axis=1)[0]

        self.write1(class_names[y_pred])
    """""
    def Model2(self, path2):
        model2 = load_model('face11-shape-recognizer.h5')
        class_names = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
        class_names_label = {class_name: i for i,
                             class_name in enumerate(class_names)}
        IMAGE_SIZE = (150, 150)
        image2 = cv2.imread(path2)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image3 = cv2.resize(image2, IMAGE_SIZE)
        image3 = np.reshape(image3, [1, 150, 150, 3])

        image3 = image3 / 255.0
        predictions = model2.predict(image3)     # Vector of probabilities
        # We take the highest probability
        pred_labels = np.argmax(predictions, axis=1)[0]
        print(pred_labels)
        print(class_names[pred_labels])
        self.write2(class_names[pred_labels])
    """
    def landmark2(self, path4):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(
            'shape_predictor_68_face_landmarks.dat')
        image = cv2.imread(path4)
        image = imutils.resize(image, width=500)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rects = detector(image, 1)
        for (i, rect) in enumerate(rects):
            shape = predictor(image, rect)
            shape = face_utils.shape_to_np(shape)

            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 255, 0), 3)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.savefig('img5.jpg')
        pixmap1 = QPixmap('img5.jpg').scaled(750, 850)
        self.ui.Face_Shape_Label.setPixmap(pixmap1)

    def write_equation(self, mathTex):
        fig = mpl.figure.Figure()
        fig.patch.set_facecolor('none')
        fig.set_canvas(FigureCanvasAgg(fig))
        renderer = fig.canvas.get_renderer()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.patch.set_facecolor('none')
        t = ax.text(0, 0, r"$%s$" % (mathTex),
                    ha='left', va='bottom', fontsize=30)

    # ---- fit figure size to text artist ----

        fwidth, fheight = fig.get_size_inches()
        fig_bbox = fig.get_window_extent(renderer)
        text_bbox = t.get_window_extent(renderer)
        tight_fwidth = text_bbox.width * fwidth / fig_bbox.width
        tight_fheight = text_bbox.height * fheight / fig_bbox.height
        fig.set_size_inches(tight_fwidth, tight_fheight)

# ---- convert mpl figure to QPixmap ----

        buf, size = fig.canvas.print_to_buffer()
        qimage = QtGui.QImage.rgbSwapped(QtGui.QImage(buf, size[0], size[1],
                                                      QtGui.QImage.Format_ARGB32))
        qpixmap = QtGui.QPixmap(qimage)
        return qpixmap
    """
    def write2(self, text):
        self.c1 = "{}".format(text)
        self.ui.label_Model2.setPixmap(
            self.write_equation(self.c1))
    """
    def write1(self, text):
        self.c1 = "{}".format(text)
        self.ui.label_Model1.setPixmap(
            self.write_equation(self.c1))

    # The logging function

    def logging(self, text):
        f = open("logHistory.txt", "w+")
        self.logHistory.append(text)
        for i in self.logHistory:
            f.write("=> %s\r\n" % (i))
        f.close()

    def browseAnImg(self):
        self.logging("browseAnImg function was called")
        image = QFileDialog.getOpenFileName()
        self.logging("Image path was chosen from the dialog box")
        self.imagePath = image[0]
        print(self.imagePath)
        self.logging("image path is set to "+self.imagePath)
        self.landmark2(self.imagePath)
        #self.Model2(self.imagePath)
        self.model1(self.imagePath)
        return self.imagePath


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = mainApp()
    main.show()
    sys.exit(app.exec_())

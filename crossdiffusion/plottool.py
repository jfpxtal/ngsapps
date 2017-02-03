from __future__ import unicode_literals
import sys
import os
import random
import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPainter, QStaticText
from PyQt5.QtWidgets import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from itertools import product

progname = os.path.basename(sys.argv[0])
progversion = "0.1"

valGrid = []
valGrid.append([1, 2, 3])
valGrid.append([i*0.02 for i in range(3, 14)])
valGrid.append([20, 25, 30, 50, 100, 200, 1000])

params = ['order', 'maxh', 'eta']

types = ['Entropy', 'L2^2 distance to equilibrium (species red)', 'L2^2 distance to equilibrium (species blue)']


class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.subplots_adjust(left=0.05, right = 0.95)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class TickLabelSlider(QWidget):
    def __init__(self, labels):
        super().__init__()
        self.labels = labels
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, len(labels))
        self.slider.setTickInterval(1)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        vlay = QVBoxLayout(self)
        vlay.setContentsMargins(0, self.fontMetrics().height()-8, 0, 0)
        vlay.addWidget(self.slider)

    def paintEvent(self, e):
        p = QPainter(self)
        sty = self.slider.style()
        opt = QStyleOptionSlider()
        self.slider.initStyleOption(opt)
        p.translate(opt.rect.x(), opt.rect.y())
        p.setPen(opt.palette.windowText().color())
        # tickOffset = sty.pixelMetric(QStyle.PM_SliderTickmarkOffset, opt, self.slider)
        available = sty.pixelMetric(QStyle.PM_SliderSpaceAvailable, opt, self.slider)
        slen = sty.pixelMetric(QStyle.PM_SliderLength, opt, self.slider)
        fudge = slen / 2
        for i in range(len(self.labels)):
            v = opt.minimum + i
            pos = sty.sliderPositionFromValue(opt.minimum, opt.maximum, v, available) + fudge
            # p.drawLine(pos, 0, pos, tickOffset - 2)
            # p.drawStaticText(pos, 0, QStaticText(str(i+1)))
            # p.drawText(pos, 0, str(i+1))
            br = p.fontMetrics().boundingRect(self.labels[i])
            # p.drawText(QPoint(pos - br.width()/2, p.fontMetrics().ascent()-2), str(i+1))
            pos = min(max(pos, br.width()/2), available+3-br.width()/2)
            p.drawStaticText(pos - br.width()/2, 0, QStaticText(self.labels[i]))



class ApplicationWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Crossdiffusion PlotTool')

        menu = QMenu('&Plot Type', self)
        menu.addAction(types[0], self.typeEntropy, QtCore.Qt.Key_E)
        menu.addAction(types[1], self.typeL2R, QtCore.Qt.Key_R)
        menu.addAction(types[2], self.typeL2B, QtCore.Qt.Key_B)
        self.menuBar().addMenu(menu)

        self.main_widget = QWidget(self)

        vlayout = QVBoxLayout(self.main_widget)
        self.dc = MyMplCanvas(self.main_widget, width=5, height=4, dpi=100)
        vlayout.addWidget(self.dc, 10)
        group = QGroupBox()
        vlayout.addWidget(group, 3)
        grid = QGridLayout(group)
        self.sOrder = TickLabelSlider([str(i) for i in valGrid[0]])
        self.sMaxH = TickLabelSlider([str(i) for i in valGrid[1]])
        self.sEta = TickLabelSlider([str(i) for i in valGrid[2]])
        self.sliders = [self.sOrder, self.sMaxH, self.sEta]
        grid.addWidget(self.sOrder, 0, 1, 1, 8)
        grid.addWidget(self.sMaxH, 1, 1, 1, 8)
        grid.addWidget(self.sEta, 2, 3, 1, 6)
        for s in self.sliders:
            s.slider.valueChanged.connect(self.updatePlot)

        self.cbConv = QCheckBox('Conv')
        self.cbDG = QCheckBox('DG')
        self.cbConv.setChecked(False)
        self.cbDG.setChecked(True)
        grid.addWidget(self.cbConv, 2, 0,)
        grid.addWidget(self.cbDG, 2, 1)
        self.cbConv.toggled.connect(self.updatePlot)
        self.cbDG.toggled.connect(self.updatePlot)

        self.cbSimul = []
        for i in range(3):
            cb = QCheckBox()
            grid.addWidget(cb, i, 9)
            cb.toggled.connect(self.updatePlot)
            self.cbSimul.append(cb)

        l0 = QLabel(params[0])
        l1 = QLabel(params[1])
        l2 = QLabel(params[2])
        grid.addWidget(l0, 0, 0)
        grid.addWidget(l1, 1, 0)
        grid.addWidget(l2, 2, 2)

        self.iType = 0

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.updatePlot()

    def updatePlot(self):
        vals = [valGrid[i][self.sliders[i].slider.value()-1] for i in range(len(self.sliders))]
        conv = self.cbConv.isChecked()
        DG = self.cbDG.isChecked()

        prodList = []
        for i in range(3):
            if self.cbSimul[i].isChecked():
                prodList.append(valGrid[i])
            else:
                prodList.append([vals[i]])

        self.dc.axes.clear()
        self.dc.axes.set_title(types[self.iType], fontsize=10)

        for prodVals in product(*prodList):
            fname = '../data/crossdiff/topf/order' + str(prodVals[0]) + '_maxh' + str(prodVals[1])
            if DG:
                fname += '_formDG_eta' + str(prodVals[2])
            else:
                fname += '_formCG'
            fname += '_conv' + str(conv) + '.csv'

            ts = []
            ents = []
            try:
                with open(fname) as f:
                    f.readline() # skip first
                    for l in f.readlines():
                        lvals = [float(v) for v in l.split(',')]
                        ts.append(lvals[0])
                        ents.append(lvals[1+self.iType])
                self.dc.axes.plot(ts, ents,
                                  label=', '.join([params[i]+'='+str(prodVals[i]) for i in range(3)
                                             if self.cbSimul[i].isChecked()]))
                self.statusBar().showMessage(fname)
                self.dc.axes.legend()
            except FileNotFoundError:
                self.statusBar().showMessage(fname + ' not found')

        self.dc.draw()

    def typeEntropy(self):
        self.iType = 0
        self.updatePlot()

    def typeL2R(self):
        self.iType = 1
        self.updatePlot()

    def typeL2B(self):
        self.iType = 2
        self.updatePlot()


qApp = QApplication(sys.argv)

aw = ApplicationWindow()
aw.setWindowTitle("%s" % progname)
aw.show()
sys.exit(qApp.exec_())

from __future__ import unicode_literals
import sys
import os
import random
import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QPoint, QSize
from PyQt5.QtGui import QPainter, QStaticText
from PyQt5.QtWidgets import *

from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas,
                                                NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from matplotlib.widgets import Cursor
from itertools import product

progname = os.path.basename(sys.argv[0])
progversion = "0.1"

valGrid = []
valGrid.append([1, 2, 3])
valGrid.append([i*0.02 for i in range(3, 14)])
valGrid.append([20, 25, 30, 50, 100, 200, 1000, 'CG'])

params = ['order', 'maxh', 'eta']

types = ['Entropy', 'L2^2 distance to equilibrium (species red)', 'L2^2 distance to equilibrium (species blue)']


class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.subplots_adjust(left=0.05, right = 0.975)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class TickLabelSlider(QSlider):
    def __init__(self, labels):
        super().__init__(Qt.Horizontal)
        self.labels = labels
        self.setRange(1, len(labels))
        self.setTickInterval(1)
        # self.setTickPosition(QSlider.TicksAbove)

    def paintEvent(self, e):
        p = QPainter(self)
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        p.translate(opt.rect.x(), opt.rect.y())
        p.setPen(opt.palette.windowText().color())
        # tickOffset = sty.pixelMetric(QStyle.PM_SliderTickmarkOffset, opt, self.slider)
        available = self.style().pixelMetric(QStyle.PM_SliderSpaceAvailable, opt, self)
        slen = self.style().pixelMetric(QStyle.PM_SliderLength, opt, self)
        fudge = slen / 2
        for i in range(len(self.labels)):
            v = opt.minimum + i
            pos = self.style().sliderPositionFromValue(opt.minimum, opt.maximum, v, available) + fudge
            # p.drawLine(pos, 0, pos, tickOffset - 2)
            br = p.fontMetrics().boundingRect(self.labels[i])
            pos = min(max(pos, br.width()/2), available+fudge-br.width()/2)
            p.drawStaticText(pos - br.width()/2, 0, QStaticText(self.labels[i]))

        super().paintEvent(e)

    def sizeHint(self):
        self.ensurePolished()
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        thick = self.style().pixelMetric(QStyle.PM_SliderThickness, opt, self)
        return QSize(84, thick+self.fontMetrics().height())



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
        navtool = NavigationToolbar(self.dc, self)
        navtool.setMovable(True)
        self.addToolBar(navtool)

        vlayout.addWidget(self.dc, 10)

        grid = QGridLayout()
        self.sOrder = TickLabelSlider([str(i) for i in valGrid[0]])
        self.sMaxH = TickLabelSlider([str(i) for i in valGrid[1]])
        self.sEta = TickLabelSlider([str(i) for i in valGrid[2]])
        self.sliders = [self.sOrder, self.sMaxH, self.sEta]
        grid.addWidget(self.sOrder, 1, 2, 1, 2)
        grid.addWidget(self.sMaxH, 2, 2, 1, 2)
        grid.addWidget(self.sEta, 3, 2, 1, 2)
        for s in self.sliders:
            s.valueChanged.connect(self.updatePlot)
        vlayout.addLayout(grid, 3)

        self.cbConv = QCheckBox('Conv')
        self.cbDG = QCheckBox('DG')
        self.cbConv.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.cbDG.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.cbConv.setChecked(False)
        self.cbDG.setChecked(True)
        grid.addWidget(self.cbConv, 2, 0,)
        grid.addWidget(self.cbDG, 3, 0)
        self.cbConv.toggled.connect(self.updatePlot)
        self.cbDG.toggled.connect(self.updatePlot)

        self.cbSimul = []
        for i in range(3):
            cb = QCheckBox()
            # cb.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            grid.addWidget(cb, 1+i, 4)
            cb.toggled.connect(self.updatePlot)
            self.cbSimul.append(cb)
        lblSimul = QLabel('all')
        lblSimul.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        grid.addWidget(lblSimul, 0, 4)

        for i in range(3):
            lbl = QLabel(params[i])
            lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            grid.addWidget(lbl, 1+i, 1)

        self.iType = 0

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.updatePlot()

    def updatePlot(self):
        vals = [valGrid[i][self.sliders[i].value()-1] for i in range(len(self.sliders))]
        conv = self.cbConv.isChecked()
        DG = self.cbDG.isChecked()

        prodList = []
        haveSimul = False
        for i in range(3):
            if self.cbSimul[i].isChecked():
                prodList.append(valGrid[i])
                haveSimul = True
            else:
                prodList.append([vals[i]])

        self.dc.axes.clear()
        self.dc.axes.set_title(types[self.iType], fontsize=10)

        for prodVals in product(*prodList):
            fname = '../data/crossdiff/topf/order' + str(prodVals[0]) + '_maxh' + str(prodVals[1])
            if prodVals[2] == 'CG':
                fname += '_formCG'
            else:
                fname += '_formDG_eta' + str(prodVals[2])
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
                if haveSimul:
                    self.dc.axes.legend()
            except FileNotFoundError:
                self.statusBar().showMessage(fname + ' not found')

        self.dc.cursor = Cursor(self.dc.axes, useblit=True, linewidth=0.5, color='k')
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

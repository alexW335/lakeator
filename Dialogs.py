from matplotlib.backends.qt_compat import QtWidgets, QtGui, QtCore
# from PyQt5.QtCore import Horizontal

class NegativeDistanceError(Exception):
    pass

class GPSError(Exception):
    pass

class EPSGError(Exception):
    pass


class GPSPopUp(QtWidgets.QDialog):
    """"""
    # Class for the set GPS information dialog box
    def __init__(self, coords=(0., 0.), EPSG=4326, pEPSG=2193, tEPSG=3857, parent=None):
        """"""
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle('Set GPS Array Information')
        self.setWindowIcon(QtGui.QIcon('kiwi.png'))
        self.setMinimumWidth(400)

        self.coordsLabel = QtWidgets.QLabel("Coordinates")
        self.name1 = QtWidgets.QLineEdit(self)
        self.name1.setText(str(coords)[1:-1])

        self.EPSGLabel = QtWidgets.QLabel("EPSG")
        self.name2 = QtWidgets.QLineEdit(self)
        self.name2.setText(str(EPSG))

        self.EPSGLabel2 = QtWidgets.QLabel("Local Projected EPSG")
        self.name3 = QtWidgets.QLineEdit(self)
        self.name3.setText(str(pEPSG))

        self.EPSGLabel3 = QtWidgets.QLabel("Target EPSG")
        self.name4 = QtWidgets.QLineEdit(self)
        self.name4.setText(str(tEPSG))
        self.name4.setStatusTip("Save the current display to a PNG file.")

        self.activate = QtWidgets.QPushButton("Set")

        Box = QtWidgets.QVBoxLayout()
        Box.addWidget(self.coordsLabel)
        Box.addWidget(self.name1)
        Box.addWidget(self.EPSGLabel)
        Box.addWidget(self.name2)
        Box.addWidget(self.EPSGLabel2)
        Box.addWidget(self.name3)
        Box.addWidget(self.EPSGLabel3)
        Box.addWidget(self.name4)
        Box.addWidget(self.activate)

        # Now put everything into the frame
        self.setLayout(Box)

    def getValues(self):
        """"""
        instr = self.name1.text().strip().split(',')
        try:
            assert len(instr) == 2
            lat, long = float(instr[0]), float(instr[1])
        except:
            raise GPSError
        try:
            EPSG = int(self.name2.text())
            pEPSG = int(self.name3.text())
            tEPSG = int(self.name4.text())
        except ValueError:
            raise EPSGError
        return [lat, long, EPSG, pEPSG, tEPSG]


class MicPositionPopUp(QtWidgets.QDialog):
    """"""
    # Class for the set GPS information dialog box
    def __init__(self, cur_locs=[[0, 0], [0, 1], [1, 0], [1, 1]], parent=None):
        """"""
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle('Set Array Configuration Information')
        self.setWindowIcon(QtGui.QIcon('kiwi.png'))
        self.setMinimumWidth(400)

        self.coordsLabel = QtWidgets.QLabel("Coordinates")
        self.name1 = QtWidgets.QPlainTextEdit(self)
        self.name1.setPlainText(self.arrayToStr(cur_locs))

        self.activate = QtWidgets.QPushButton("Set")

        Box = QtWidgets.QVBoxLayout()
        Box.addWidget(self.coordsLabel)
        Box.addWidget(self.name1)
        Box.addWidget(self.activate)

        # Now put everything into the frame
        self.setLayout(Box)

    def arrayToStr(self, arr):
        """"""
        strout = ""
        for row in range(len(arr)):
            strout += "{}, {}\n".format(arr[row][0], arr[row][1])
        return strout

    def getValues(self):
        """"""
        tmp = []
        # print(r"{}".format(self.name1.toPlainText()))
        instr = self.name1.toPlainText().strip().split('\n')
        for ind in instr:
            a = ind.split(',')
            tmp.append([float(a[0].strip()), float(a[1].strip())])
        return tmp

class HeatmapBoundsPopUp(QtWidgets.QDialog):
    """"""
    # Class for the set GPS information dialog box
    def __init__(self, l, r, u, d, parent=None):
        """"""
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle('Set Heatmap Domain')
        self.setWindowIcon(QtGui.QIcon('kiwi.png'))
        self.setMinimumWidth(400)

        self.header = QtWidgets.QLabel("All distances are relative to the array center.")

        self.leftLabel = QtWidgets.QLabel("Left/West:")
        self.lledit = QtWidgets.QLineEdit(self)
        self.lledit.setText(str(l))

        self.rightLabel = QtWidgets.QLabel("Right/East:")
        self.rledit = QtWidgets.QLineEdit(self)
        self.rledit.setText(str(r))

        self.upLabel = QtWidgets.QLabel("Up/North:")
        self.uledit = QtWidgets.QLineEdit(self)
        self.uledit.setText(str(u))

        self.downLabel = QtWidgets.QLabel("Down/South:")
        self.dledit = QtWidgets.QLineEdit(self)
        self.dledit.setText(str(d))

        self.activate = QtWidgets.QPushButton("Set")

        Box = QtWidgets.QVBoxLayout()

        Box.addWidget(self.header)

        left = QtWidgets.QHBoxLayout()
        left.addWidget(self.leftLabel)
        left.addWidget(self.lledit)

        right = QtWidgets.QHBoxLayout()
        right.addWidget(self.rightLabel)
        right.addWidget(self.rledit)

        up = QtWidgets.QHBoxLayout()
        up.addWidget(self.upLabel)
        up.addWidget(self.uledit)

        down = QtWidgets.QHBoxLayout()
        down.addWidget(self.downLabel)
        down.addWidget(self.dledit)

        Box.addLayout(left)
        Box.addLayout(right)
        Box.addLayout(up)
        Box.addLayout(down)

        Box.addWidget(self.activate)

        # Now put everything into the frame
        self.setLayout(Box)

    def getValues(self):
        """"""
        a,b,c,d = float(self.lledit.text()), float(self.rledit.text()), \
        float(self.uledit.text()), float(self.dledit.text())
        if a >= b or c <= d:
            raise NegativeDistanceError
        return a, b, c, d


class AlgorithmSettingsPopUp(QtWidgets.QDialog):
    """"""
    # Class for the set GPS information dialog box
    def __init__(self, algSettingsDict, parent=None):
        """"""
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle('Change Algorithm-Specific Settings')
        self.setWindowIcon(QtGui.QIcon('kiwi.png'))
        self.setMinimumWidth(400)
        self.cur = algSettingsDict["current"]

        # Set the font to be used for section headings
        fnt = QtGui.QFont()
        fnt.setBold(True)

        # AF-MUSIC
        self.afTitle = QtWidgets.QLabel("AF-MUSIC:")
        self.afTitle.setFont(fnt)

        self.fmax_l = QtWidgets.QLabel(u"\u0192_max (Hz):")
        self.fmax = QtWidgets.QLineEdit(self)
        self.fmax.setText(str(algSettingsDict["AF-MUSIC"]["f_max"]))

        self.fmin_l = QtWidgets.QLabel(u"\u0192_min (Hz):")
        self.fmin = QtWidgets.QLineEdit(self)
        self.fmin.setText(str(algSettingsDict["AF-MUSIC"]["f_min"]))

        self.f0_l = QtWidgets.QLabel("Focusing Frequency (Hz):")
        self.f0 = QtWidgets.QLineEdit(self)
        self.f0.setText(str(algSettingsDict["AF-MUSIC"]["f_0"]))

        # GCC
        self.GCCTitle = QtWidgets.QLabel("GCC:")
        self.GCCTitle.setFont(fnt)

        # Available processors: PHAT, p-PHAT, CC, RIR, SCOT, HB
        self.GCC_l = QtWidgets.QLabel("Processor:")
        # self.GCC.setText(algSettingsDict["GCC"]["processor"])

        self.possible_processors = ["PHAT", "p-PHAT", "CC", "RIR", "SCOT", "HB"]
        self.cb = QtWidgets.QComboBox()
        self.cb.addItems(self.possible_processors)
        self.cb.setCurrentIndex(self.possible_processors.index(algSettingsDict["GCC"]["processor"]))


        self.def_rho = algSettingsDict["GCC"]["rho"]
        self.sl_l = QtWidgets.QLabel(u"0 \u2265 \u03C1={} \u2265 1:".format(self.def_rho))
        self.sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl.setMinimum(0)
        self.sl.setMaximum(100)
        self.sl.setTickInterval(10)
        self.sl.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.sl.setValue(int(self.def_rho*100))
        self.sl.valueChanged.connect(self.changeRho)

        # MUSIC
        self.MUSICTitle = QtWidgets.QLabel("MUSIC:")
        self.MUSICTitle.setFont(fnt)

        self.MUSIC_l = QtWidgets.QLabel("Frequency:")
        self.MUSIC = QtWidgets.QLineEdit(self)
        self.MUSIC.setText(str(algSettingsDict["MUSIC"]["freq"]))

        self.activate = QtWidgets.QPushButton("Set")

        Box = QtWidgets.QVBoxLayout()

        Box.addWidget(self.afTitle)
        f_min = QtWidgets.QHBoxLayout()
        f_min.addWidget(self.fmin_l)
        f_min.addWidget(self.fmin)
        Box.addLayout(f_min)

        f_max = QtWidgets.QHBoxLayout()
        f_max.addWidget(self.fmax_l)
        f_max.addWidget(self.fmax)
        Box.addLayout(f_max)

        f_0 = QtWidgets.QHBoxLayout()
        f_0.addWidget(self.f0_l)
        f_0.addWidget(self.f0)
        Box.addLayout(f_0)

        Box.addSpacerItem(QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum,
                                                QtWidgets.QSizePolicy.Expanding))

        Box.addWidget(self.GCCTitle)
        procBox = QtWidgets.QHBoxLayout()
        procBox.addWidget(self.GCC_l)
        procBox.addWidget(self.cb)
        Box.addLayout(procBox)

        rhoBox = QtWidgets.QHBoxLayout()
        rhoBox.addWidget(self.sl_l)
        rhoBox.addWidget(self.sl)
        Box.addLayout(rhoBox)

        Box.addSpacerItem(QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum,
                                                QtWidgets.QSizePolicy.Expanding))

        Box.addWidget(self.MUSICTitle)
        MUSICBox = QtWidgets.QHBoxLayout()
        MUSICBox.addWidget(self.MUSIC_l)
        MUSICBox.addWidget(self.MUSIC)
        Box.addLayout(MUSICBox)

        Box.addWidget(self.activate)

        # Now put everything into the frame
        self.setLayout(Box)

    def changeRho(self):
        self.sl_l.setText(u"0 \u2265 \u03C1={} \u2265 1:".format(self.sl.value()/100.0))

    def getValues(self):
        """"""
        retDict = {
            "AF-MUSIC": {
                "f_0": float(self.f0.text()),
                "f_max": float(self.fmax.text()),
                "f_min": float(self.fmin.text())
            },
            "GCC": {
                "processor": self.cb.currentText(),
                "rho": self.sl.value()/100.0
            },
            "MUSIC": {
                "freq": float(self.MUSIC.text())
            },
            "current": self.cur
        }
        return retDict

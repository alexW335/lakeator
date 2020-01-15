from matplotlib.backends.qt_compat import QtCore, QtWidgets, QtGui
from numpy import array

class GPSPopUp(QtWidgets.QDialog):
    # Class for the set GPS information dialog box
    def __init__(self, coords=(0., 0.), EPSG=4326, pEPSG=2193, tEPSG=3857, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle('Set GPS Array Information')
        self.setWindowIcon(QtGui.QIcon('kiwi.png'))
        # self.setWindowFlags((self.windowFlags() ^ QtCore.Qt.WindowContextHelpButtonHint) | QtCore.Qt.WindowStaysOnTopHint)
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
        instr = self.name1.text().strip().split(',')
        assert len(instr) == 2
        lat, long = float(instr[0]), float(instr[1])
        EPSG = int(self.name2.text())
        pEPSG = int(self.name3.text())
        tEPSG = int(self.name4.text())
        return [lat, long, EPSG, pEPSG, tEPSG]

    
class MicPositionPopUp(QtWidgets.QDialog):
    # Class for the set GPS information dialog box
    def __init__(self, cur_locs=array([[0,0],[0,1],[1,0],[1,1]]), parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle('Set GPS Array Information')
        self.setWindowIcon(QtGui.QIcon('kiwi.png'))
        # self.setWindowFlags((self.windowFlags() ^ QtCore.Qt.WindowContextHelpButtonHint) | QtCore.Qt.WindowStaysOnTopHint)
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
        strout = ""
        for row in range(arr.shape[0]):
            strout += "{}, {}\n".format(arr[row, 0], arr[row, 1])
        return strout
    
    def getValues(self):
        tmp = []
        print(r"{}".format(self.name1.toPlainText()))
        instr = self.name1.toPlainText().strip().split('\n')
        for ind in instr:
            a = ind.split(',')
            tmp.append([float(a[0].strip()), float(a[1].strip())])
        tmp = array(tmp)
        return tmp
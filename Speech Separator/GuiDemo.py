from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QDesktopWidget
from PyQt5.QtWidgets import QToolTip
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QLabel
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QAction
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QStatusBar
from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtWidgets import QDesktopWidget

import sys
import pyaudio
import wave
import vlc
import os.path
from audio_test_mp import out_put
from spectrogram import spectrogram
from scipy.signal import resample
import numpy as np
import soundfile as sf

class SpeechEnhancer(QMainWindow, QWidget):

    def __init__(self):
        super(SpeechEnhancer, self).__init__()
        self.initUI() 
        
    def record_audio(self):
        self.statusBar().showMessage('Recording Started')
        self.textshow.setText('Recording Started')
        app.processEvents()

        CHUNK = 1024
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        RATE = 44100
        SAMPLING_RATE = 8000
        RECORD_SECONDS = 5
        WAVE_OUTPUT_FILENAME = "input.wav"

        p = pyaudio.PyAudio()
        global audio_8k

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
            if i == 0:
                audio = np.fromstring(data, 'Float32')
            else:    
                audio = np.concatenate((audio, np.fromstring(data, 'Float32')), axis=0)

        stream.stop_stream()
        stream.close()
        p.terminate()        

        resampling_factor = RATE/SAMPLING_RATE
        audio_8k = resample(audio, int(len(audio)/resampling_factor))
        #audio_8k = audio
        self.statusBar().showMessage('Recording Stoped')
        self.textshow.setText('Recording Stoped')
        app.processEvents()

        sf.write(os.path.join(WAVE_OUTPUT_FILENAME), audio_8k, SAMPLING_RATE)
        self.statusBar().showMessage('File Saved')
        
        global filename

        filename = os.path.join('input.wav')
        self.textshow.setText("Recording completed and input file is saved")
 
    def fileinput(self): 
        global filename
        global audio_8k
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self, 'select audio file', '.', '*.wav')
        audio_8k, sample_rate = sf.read(os.path.join(filename))

    def fileEport(self):        
        global exportpath
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        exportpath = QFileDialog.getExistingDirectory(self,'Save File Directory')

    def playinput(self):       
        if 'filename' in globals():
            global player
            player = vlc.MediaPlayer(filename)
            player.play()
        else:
            pass

    def pauseinput(self):
        if 'player' in globals():           
            player.pause()   
        else:
            pass

    def stopinput(self):
        if 'player' in globals():
            player.stop()
        else:
            pass

    def playOutput1(self):        
        if 'exportpath' in globals():
            global player1
            player1 = vlc.MediaPlayer(os.path.join(exportpath,'SEN_1.wav'))
            player1.play()       
        else:
            pass

    def pauseOutput1(self):
        if 'player1' in globals():
            player1.pause()
        else:
            pass
   
    def stopOutput1(self):
        if 'player1' in globals():
            player1.stop()
        else:
            pass

    def playOutput2(self):        
        if 'exportpath' in globals():
            global player2
            player2 = vlc.MediaPlayer(os.path.join(exportpath,'SEN_2.wav'))
            player2.play()
        else:
            pass

    def pauseOutput2(self):
        if 'player2' in globals():
            player2.pause()
        else:
            pass
   
    def stopOutput2(self):
        if 'player2' in globals():
            player2.stop()
        else:
            pass

    def run(self):
        if 'exportpath' in globals() and os.path.exists(exportpath) and os.path.exists(filename):            
            self.statusBar().showMessage('Model Runing')
            self.textshow.setText('Wait!!! Model Runing')
            app.processEvents()
            out_put(400, os.path.join(exportpath), ratio_mask=False, silence_mask=False, CONFUSION_MATRIX=False, input_audio=True, frames=audio_8k)
            self.statusBar().showMessage('Model Successfully Enhanced the input audio file')
            self.textshow.setText('Model Successfully Enhanced the input audio file')
            app.processEvents()
        else: 
            if 'filename' in globals() and os.path.exists(filename):
                QMessageBox.about(self, 'Warning',
                "First Add an save directory for output files ")  
            else:
                QMessageBox.about(self, 'Warning',
                "First Add an input Audio file")       

    def input_plot(self): 
        if 'filename' in globals() and os.path.exists(filename):
            spectrogram(os.path.join(filename), 'Input Audio Sample', 'input.png')
            Pixmap = QPixmap('input.png')
            self.Spectrogram.setPixmap(Pixmap)                    
        else:
            QMessageBox.about(self, 'Warning',
            "First add external audio file or record audio file")

    def output1_plot(self): 
        if 'exportpath' in globals() and os.path.exists(exportpath):
            spectrogram(os.path.join(exportpath,'SEN_1.wav'), 'Output Audio Sample One', 'output01.png')
            Pixmap = QPixmap('output01.png')
            self.Spectrogram.setPixmap(Pixmap)                   
        else:
            QMessageBox.about(self, 'Warning',
            "First run Acoustic Separation Model")

    def output2_plot(self):
        if 'exportpath' in globals() and os.path.exists(exportpath):
            spectrogram(os.path.join(exportpath,'SEN_2.wav'), 'Output Audio Sample Two', 'output02.png')
            Pixmap = QPixmap('output02.png')
            self.Spectrogram.setPixmap(Pixmap)                    
        else:
            QMessageBox.about(self, 'Warning',
            "First run Acoustic Separation Model")

    def comboset(self):
        count = self.combo.currentIndex()
        if count == 1:           
            self.input_plot()
        elif count == 2:
            self.output1_plot()
        elif count == 3:
            self.output2_plot()
        else:
            pass

    def aboutus(self):
        QMessageBox.about(self, 'About',
        "Acoustic Source Separation\n\
        FYP Group 4\n\
        Usman Anwar 2015-EE-17\n\
        Muhammad Ismaeel 2015-EE-18\n\
        Mohsin Tanveer 2015-EE-23\n\
        Osama Rashid 2015-EE-24")

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def initUI(self):
        QToolTip.setFont(QFont('SansSerif', 10))

        exitAct = QAction(QIcon('exit.png'), 'Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.triggered.connect(self.close)
        aboutAct = QAction(QIcon('about.png'), 'About', self)
        aboutAct.setShortcut('Ctrl+A')
        aboutAct.triggered.connect(self.aboutus)
        self.toolbar = self.addToolBar('Exit')
        self.toolbar.addAction(exitAct)
        self.toolbar = self.addToolBar('About')
        self.toolbar.addAction(aboutAct)

        a = QLabel("Status", self)
        a.move(200,140)

        a = QLabel(self)
        a.setText("Duration:10s")
        a.move(200,80)

        b5 = QPushButton('', self)
        b5.clicked.connect(self.record_audio)
        b5.setIcon(QtGui.QIcon('record.png'))
        b5.setIconSize(QtCore.QSize(80,50))
        b5.setToolTip('Record Input Voice for Separation')
        b5.resize(100,70)
        b5.move(30,80)

        b6 = QPushButton('', self)
        b6.clicked.connect(self.run)
        b6.setIcon(QtGui.QIcon('run.png'))
        b6.setIconSize(QtCore.QSize(80,50))
        b6.setToolTip('Process the audio file using DL Model')
        b6.resize(100,70)
        b6.move(30,180)

        a = QLabel("Input Audio", self)
        a.move(30,270)

        b7 = QPushButton('', self)
        b7.clicked.connect(self.playinput)
        b7.setIcon(QtGui.QIcon('play.png'))
        b7.setToolTip('Play Input')
        b7.setIconSize(QtCore.QSize(25,25))
        b7.resize(40,40)
        b7.move(30,310)

        b8 = QPushButton('', self)
        b8.clicked.connect(self.pauseinput)
        b8.setIcon(QtGui.QIcon('pause.png'))
        b8.setToolTip('Pause Input')
        b8.setIconSize(QtCore.QSize(25,25))
        b8.resize(40,40)
        b8.move(80,310)

        b9 = QPushButton('', self)
        b9.clicked.connect(self.stopinput)
        b9.setIcon(QtGui.QIcon('stop.png'))
        b9.setToolTip('Stop Input')
        b9.setIconSize(QtCore.QSize(25,25))
        b9.resize(40,40)
        b9.move(130,310)

        a = QLabel("Output One", self)
        a.move(30,360)

        b11 = QPushButton('', self)
        b11.clicked.connect(self.playOutput1)
        b11.setIcon(QtGui.QIcon('play.png'))
        b11.setToolTip('Play Output One')
        b11.setIconSize(QtCore.QSize(25,25))
        b11.resize(40,40)
        b11.move(30,400)

        b12 = QPushButton('', self)
        b12.clicked.connect(self.pauseOutput1)
        b12.setIcon(QtGui.QIcon('pause.png'))
        b12.setToolTip('Pause Output One')
        b12.setIconSize(QtCore.QSize(25,25))
        b12.resize(40,40)
        b12.move(80,400)

        b13 = QPushButton('', self)
        b13.clicked.connect(self.stopOutput1)
        b13.setIcon(QtGui.QIcon('stop.png'))
        b13.setToolTip('Stop Output One')
        b13.setIconSize(QtCore.QSize(25,25))
        b13.resize(40,40)
        b13.move(130,400)

        a = QLabel("Output Two", self)
        a.move(30,450)

        b15 = QPushButton('', self)
        b15.clicked.connect(self.playOutput2)
        b15.setIcon(QtGui.QIcon('play.png'))
        b15.setToolTip('Play Output Two')
        b15.setIconSize(QtCore.QSize(25,25))
        b15.resize(40,40)
        b15.move(30,490)

        b16 = QPushButton('', self)
        b16.clicked.connect(self.pauseOutput2)
        b16.setIcon(QtGui.QIcon('pause.png'))
        b16.setToolTip('Pause Output Two')
        b16.setIconSize(QtCore.QSize(25,25))
        b16.resize(40,40)
        b16.move(80,490)

        b17 = QPushButton('', self)
        b17.clicked.connect(self.stopOutput2)
        b17.setIcon(QtGui.QIcon('stop.png'))
        b17.setToolTip('Stop Output Two')
        b17.setIconSize(QtCore.QSize(25,25))
        b17.resize(40,40)
        b17.move(130,490)

        self.statusBar()

        self.Spectrogram = QLabel(self)
        self.Spectrogram.resize(640,480)
        self.Spectrogram.move(420,80)

        menubar = self.menuBar()
        Menu = menubar.addMenu('File Input')
        SaveFile = QAction('Output Directory', self)
        SaveFile.setShortcut('Ctrl+S')
        SaveFile.triggered.connect(self.fileEport)
        Save = menubar.addMenu('Output Directory')
        Save.addAction(SaveFile)
        External = QAction('Open File', self)
        External.setShortcut('Ctrl+F')
        External.triggered.connect(self.fileinput)
        Menu.addAction(External)

        self.textshow = QTextEdit(self)
        self.textshow.resize(180,70)
        self.textshow.move(200,180)
        self.textshow.setText("Acoustic Source Separator")

        self.combo = QComboBox(self)
        self.combo.addItem("None")
        self.combo.addItem("Input")
        self.combo.addItem("Output One")
        self.combo.addItem("output Two")
        self.combo.move(200, 310)
        self.combo.resize(180, 40)
        self.combo.currentIndexChanged.connect(self.comboset)

        a = QLabel("Spectrogram", self)
        a.move(200,270)

        self.setWindowTitle('Acoustic Source Separator')
        self.setWindowIcon(QIcon('icon.png'))  
        self.resize(1100,600)
        self.center()
        self.show()  

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet('QMainWindow{background-color: DimGray;border: 20px solid Silver;}\
                        QPushButton{background-color: Gray;}\
                            QComboBox{background-color: Gray;}\
                                QTextEdit{background-color: Silver;}\
                                    QMenuBar{background-color: Gray;}\
                                        QToolBar{background-color: Silver}')
    ex = SpeechEnhancer()
    sys.exit(app.exec_())
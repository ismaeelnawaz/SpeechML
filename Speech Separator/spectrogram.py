import matplotlib.pyplot as plot
from scipy.io import wavfile

def spectrogram(path, description, FileName):

    plot.cla()
    plot.clf()
    samplingFrequency, signalData = wavfile.read(path)
    plot.subplot(211)
    plot.title(description)
    plot.plot(signalData)
    plot.subplot(212)
    plot.specgram(signalData,Fs=samplingFrequency)
    plot.savefig(FileName)
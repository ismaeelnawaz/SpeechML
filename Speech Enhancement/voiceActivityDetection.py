import numpy as np
from scipy.stats.mstats import gmean  
import librosa 
import os
import matplotlib.pyplot as plt
from GlobalConstont import *  

class VoiceActivityDetection(object):
    """
    Implements a voice activity detection module for continous stream of data. 
    References: 
    1. http://practicalcryptography.com/miscellaneous/machine-learning/voice-activity-detection-vad-tutorial/
    2. https://www.eurasip.org/Proceedings/Eusipco/Eusipco2009/contents/papers/1569192958.pdf
    3. https://medium.com/linagoralabs/voice-activity-detection-for-voice-user-interface-2d4bb5600ee3
    """
    def __init__(self, primary_energy_th=40, primary_max_freq_th=180, primary_SF_th=5):
        self.primSEthresh = primary_energy_th
        self.primMFthresh = primary_max_freq_th
        self.primSFthresh = primary_SF_th
        self.minSE = self.primSEthresh
        self.minMF = self.primMFthresh
        self.minSF = self.primSFthresh
        self.SEthresh = self.primSEthresh
        self.FramesProcessed = 0
        self.speechRun = 0
        self.silenceRun = 0

    def computeSE(self, frame):
        return np.sum(np.square(frame))

    def computeZC(self, frame):
        """
        Returns zero crossings for the signal 
        Ref: https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
        """
        return np.sum(np.diff(np.signbit(frame)))

    def computeSFMandMaxFreq(self, frame):
        """
        Computes spectral flatness measure: SFMdb = 10 log_10 (Geometric_Mean of spectrum/aritmetic mean of spectrum)
        and maximum frequency in the frame.
        """
        win = np.hanning(len(frame))
        frame *= win
        spec = np.fft.rfft(frame)
        G_m = np.abs(gmean(spec))
        A_m = np.abs(np.mean(spec))
        # print(A_m, G_m)
        # print(np.argmax(spec))
        return (10*np.log10(G_m/A_m), np.argmax(spec)*SAMPLING_RATE)
    
    def update_minSE(self, newSE):
        if newSE < self.minSE:
            self.minSE = newSE

    def update_minMF(self, newMF):
        if newMF < self.minMF:
            self.minMF = newMF
    
    def update_minSF(self, newSF):
        if newSF < self.minSF:
            self.minSF = newSF
    
    def update_thresh(self):
        self.SEthresh = self.primSEthresh*np.log10(self.minSE)

    def IsSpeech(self, frame):
        """
        Returns True if Speech else False.
        """
        if self.FramesProcessed <30:
            frame_SE = self.computeSE(frame)
            frame_SF, frame_MF = self.computeSFMandMaxFreq(frame)
            self.update_minMF(frame_MF)
            self.update_minSE(frame_SE)
            self.update_minSF(frame_SF)
            self.FramesProcessed = self.FramesProcessed + 1
            return False
        else:
            counter = 0
            frame_SE = self.computeSE(frame)
            frame_SF, frame_MF = self.computeSFMandMaxFreq(frame)
            if frame_SE - self.minSE >= self.SEthresh:
                counter = counter + 1
            if frame_SF - self.minSF >= self.primSFthresh:
                counter = counter + 1
            if frame_MF - self.minMF >= self.primMFthresh:
                counter = counter + 1
            if counter > 1:
                counter = 0
                self.speechRun = self.speechRun + 1
                self.silenceRun = 0
                return True  # Frame contains speech
            else:
                self.silenceRun = self.silenceRun + 1
                self.speechRun = 0
                self.minSE = ((self.silenceRun * self.minSE) + frame_SE)/(self.silenceRun + 1)
                self.SEthresh = self.primSEthresh*np.log(self.minSE)
                return False # Frame does not contain speech
    
if __name__ == "__main__":
    fig, ax = plt.subplots(6,1, figsize=(20,10))
    # loading file
    audio, sr = librosa.load(os.path.join('input_sample3', 'mix.wav'), sr=SAMPLING_RATE)
    # dividing into frames
    frames = []
    SE = np.zeros(audio.shape)
    ZC = np.zeros(audio.shape)
    SFM = np.zeros(audio.shape)
    MaxFreq = np.zeros(audio.shape)
    speechOrNot = np.zeros(audio.shape)
    
    z = 0
    VAD = VoiceActivityDetection()
    while(z + FRAME_SIZE < len(audio)):
        frame = (audio[z:z+FRAME_SIZE])
        # plt.plot(np.fft.rfft(frame * np.hanning(FRAME_SIZE)))
        # plt.show()
        SE[z:z+FRAME_SIZE] = [VAD.computeSE(frame)]*FRAME_SIZE
        ZC[z:z+FRAME_SIZE] = [VAD.computeZC(frame)]*FRAME_SIZE
        sfm, mf = VAD.computeSFMandMaxFreq(frame)
        # print(mf)
        SFM[z:z+FRAME_SIZE] = [sfm]*FRAME_SIZE
        MaxFreq[z:z+FRAME_SIZE] = [mf]*FRAME_SIZE
        speechOrNot[z:z+FRAME_SIZE] = [VAD.IsSpeech(frame)]*FRAME_SIZE
        z = z + FRAME_SIZE
    
    frame_cut = np.zeros(audio.shape)
    z = 0
    while(z + FRAME_SIZE*10 < len(audio)):
        frame_cut[z] = 1
        z = z + FRAME_SIZE*10

    ax[0].plot(audio)
    ax[0].plot(frame_cut)
    ax[1].plot((SE>40))
    ax[1].set_title('Speech Energy')
    print(1)
    # ax[1].plot([40]*len(audio))
    ax[2].plot((ZC >60))
    print(1)
    # ax[2].plot([50]*len(audio))
    ax[3].plot((SFM>15))
    print(1)
    # ax[3].plot([5]*len(audio))
    ax[4].plot((MaxFreq>400))
    print(1)
    # ax[4].plot([180]*len(audio))
    ax[5].plot(speechOrNot)
    plt.show()



    



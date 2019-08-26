'''
Class AudioSampleReader2:
    read in a single audio sample for test.
    Different from AudioSampleReader that it also takes in oracle1.wav and oracle2.wav
    and outputs ideal binary mask as well. 
'''

import numpy as np
import librosa
from numpy.lib import stride_tricks
# import ipdb
import os
# import matplotlib as mpl
# mpl.use('agg')
# from matplotlib import pyplot as plt
from GlobalConstont import *


def stft(sig, frameSize, overlapFac=0.75, window=np.hanning):
    """ short time fourier transform of audio signal """
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    # samples = np.append(np.zeros(np.floor(frameSize / 2.0)), sig)
    samples = np.array(sig, dtype='float64')
    # cols for windowing
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    frames = stride_tricks.as_strided(
        samples,
        shape=(int(cols), frameSize),
        strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win
    return np.fft.rfft(frames)


class AudioSampleReader2(object):
    '''
    Class AudioSampleReader:
        read in a single audio sample for test using trained model
    '''
    def __init__(self, data_dir, data_dir1, data_dir2):
        '''Load in the audio file and transform the signal into
        the formats required by the model'''

        # loading and transformation
        # if data_dir.split('.')[-1] == 'npy':
        #     speech_mix = np.load(data_dir)
        #     if speech_mix.shape[0] > 1:
        #         speech_mix = speech_mix[int(np.random.rand()),:]
        # else:
        
        speech_mix, _ = librosa.load(data_dir, SAMPLING_RATE)
        speech_1, _ = librosa.load(data_dir1, SAMPLING_RATE)
        speech_2, _ = librosa.load(data_dir2, SAMPLING_RATE)

        # Compute log spectrum for the mixed sample
        speech_mix_spec0 = stft(speech_mix, FRAME_SIZE)[:, :NEFF]
        speech_mix_spec = np.abs(speech_mix_spec0)
        speech_phase = speech_mix_spec0 / (speech_mix_spec + 0.000001)  # avoid dividing by zero
        speech_mix_spec = np.maximum(
            speech_mix_spec, np.max(speech_mix_spec) / MIN_AMP)
        speech_mix_spec = 20. * np.log10(speech_mix_spec * AMP_FAC)
        max_mag = np.max(speech_mix_spec)
        speech_VAD = (speech_mix_spec > (max_mag - THRESHOLD)).astype(int)
        speech_mix_spec = (speech_mix_spec - GLOBAL_MEAN) / GLOBAL_STD

        # Compute log spectrum for the first speaker
        speech_1_spec = np.abs(stft(speech_1, FRAME_SIZE)[:, :NEFF])
        speech_1_spec = np.maximum(speech_1_spec, np.max(speech_1_spec) / MIN_AMP)
        speech_1_spec = 20. * np.log10(speech_1_spec * AMP_FAC)
        # Compute log spectrum for the second speaker
        speech_2_spec = np.abs(stft(speech_2, FRAME_SIZE)[:, :NEFF])
        speech_2_spec = np.maximum(speech_2_spec, np.max(speech_2_spec) / MIN_AMP)
        speech_2_spec = 20. * np.log10(speech_2_spec * AMP_FAC)

        # Make Sure that speech_1_spec and speech_2_spec have same lengths as speech_mix_spec
        len_spec_mix = speech_mix_spec.shape[-1]
        len_spec_1 = speech_1_spec.shape[-1]
        len_spec_2 = speech_2_spec.shape[-1]

        assert(len_spec_mix == len_spec_1 == len_spec_2)

        len_spec = speech_mix_spec.shape[0]
        k = 0
        self.ind = 0
        self.samples = []

        # feed the transformed data into a sample list
        while(k + FRAMES_PER_SAMPLE < len_spec):
            phase = speech_phase[k: k + FRAMES_PER_SAMPLE, :]
            sample_mix = speech_mix_spec[k:k + FRAMES_PER_SAMPLE, :]
            VAD = speech_VAD[k:k + FRAMES_PER_SAMPLE, :]

            sample_1 = speech_1_spec[k:k + FRAMES_PER_SAMPLE, :]
            sample_2 = speech_2_spec[k:k + FRAMES_PER_SAMPLE, :]

            Y = np.array([sample_1 > sample_2, sample_1 < sample_2]).astype('bool')
            Y = np.transpose(Y, [1, 2, 0])


            sample_dict = {'Sample': sample_mix,
                           'VAD': VAD,
                           'Phase': phase,
                           'Labels': Y}
            self.samples.append(sample_dict)
            k = k + FRAMES_PER_SAMPLE

        # import ipdb; ipdb.set_trace()
        n_left = FRAMES_PER_SAMPLE - len_spec + k
        # store phase for waveform reconstruction
        phase = np.concatenate((speech_phase[k:, :], np.zeros([n_left, NEFF])))
        sample_mix = np.concatenate(
            (speech_mix_spec[k:, :], np.zeros([n_left, NEFF])))
        VAD = np.concatenate((speech_VAD[k:, :], np.zeros([n_left, NEFF])))

        sample_1 = np.concatenate((speech_1_spec[k:, :], np.zeros([n_left, NEFF])))
        sample_2 = np.concatenate((speech_2_spec[k:, :], np.zeros([n_left, NEFF])))

        Y = np.array([sample_1 > sample_2, sample_1 < sample_2]).astype('bool')
        Y = np.transpose(Y, [1, 2, 0])

         
        sample_dict = {'Sample': sample_mix,
                       'VAD': VAD,
                       'Phase': phase,
                       'Labels': Y}
        self.samples.append(sample_dict)
        self.tot_samp = len(self.samples)

    def gen_next(self):
        # ipdb.set_trace()
        begin = self.ind
        if begin >= self.tot_samp:
            return None
        self.ind += 1
        return [self.samples[begin]]

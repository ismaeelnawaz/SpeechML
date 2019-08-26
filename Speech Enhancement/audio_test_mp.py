'''
Script to test a audio sample using trained model
Args/Flags:
    model_dir = directory containing model
    data_dir = mixed_speech.wav file
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import pyaudio 
from scipy.signal import resample

import numpy as np
import librosa
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import tensorflow as tf
import itertools
from numpy.lib import stride_tricks

from AudioSampleReader2 import AudioSampleReader2
from AudioSampleReader import AudioSampleReader
from model import Model

from GlobalConstont import *
import soundfile as sf

# not useful during sample test
sum_dir = 'sum'
# dir to load model
#train_dir = os.path.join('log_12000', 'model.ckpt-10400')
train_dir = os.path.join('Model', 'model.ckpt-2000')
# train_dir = os.path.join('/home/osama/repos', 'Final Demo', 'log_12000', 'model.ckpt-12000')

lr = 0.00001  # not useful during test
n_hidden = 300  # hidden state size
batch_size = 1  # 1 for audio sample test
hop_size = 64
# oracle flag to decide if a frame need to be seperated
sep_flag = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * 30
# oracle permutation to concatenate the chuncks of output frames
oracal_p = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * 30
# import ipdb; ipdb.set_trace()
NUMBER_OF_SAMPLES = 5   # Samples for audio testing

def distance(x1, x2):
    """
    Params:
    x1, x2 are one numpy arrays of length n. 
    Returns:
    Function returns the euclidean distance between points.
    """
    return np.sqrt(np.sum(np.square(x1 - x2)))

def confusion_matrix(labels, predictions):
    """
    Returns a confusion matrix M of dim nxn, where n is the number of the
    speakers. M[0][0] contains the bins that belonged to speaker 0 and were
    classified as same by the model. M[0][n] contains the bins that belonged
    to speaker 0 but were classified by model as belonging to speaker n. 
    Params:
    labels: a ndarray of FRAME_SIZE x NEFF x n of 1 and 0. This is the true mask.
    predictions: an ndarray of FRAME_SIZE x NEFF x n of 1 and 0. This is the model 
    output mask.
    Returns:
    confusion_matrix: a nxn matrix which definces confusion matrix 
    """
    ## This code is primarily written for 2 speaker case and might not generalize
    ## to more speakers

    n = max(labels.shape[-1], predictions.shape[-1])
    confusion_matrix = np.zeros((n, n))
    for i in range(n):
        confusion_matrix[i][i] = np.sum(np.equal(labels[:,:,:,i], predictions[:,:,i]))
    
    confusion_matrix[0][1] = np.sum(np.greater(labels[:,:,:,0], predictions[:,:,0]))
    confusion_matrix[1][0] = np.sum(np.greater(labels[:,:,:,1], predictions[:,:,1]))
    
    return confusion_matrix


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
        shape=(cols, frameSize),
        strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win
    return np.fft.rfft(frames)


def out_put(N_frame, dir, batch_size=5, input_audio=False, ratio_mask=False, frames=None,
            CONFUSION_MATRIX=True, silence_mask=True, visualize_embedding=False, 
            visualize_tsne =False, spectrograms=False, time_domain_graphs=True):
    '''Use trained model to infer N _frame chuncks of
    frames of input audio'''
    if input_audio:
        assert(frames is not None)
    with tf.Graph().as_default():
        print('OUTPUT STARTED')
        # feed forward keep prob
        p_keep_ff = tf.placeholder(tf.float32, shape=None)
        # recurrent keep prob
        p_keep_rc = tf.placeholder(tf.float32, shape=None)

        # audio sample generator
        if input_audio:
                data_generator = AudioSampleReader(os.path.join(dir, 'mix.wav'), frames=frames, batch_size=batch_size)
        else:
            if CONFUSION_MATRIX:
                data_generator = AudioSampleReader2(os.path.join(dir, 'mix.wav'), 
                os.path.join(dir, 'oracle_fem.wav'), os.path.join(dir, 'oracle_male.wav'))
            else:
                data_generator = AudioSampleReader(os.path.join(dir, 'mix.wav'), batch_size=batch_size)

        # placeholder for model input
        in_data = tf.placeholder(
            tf.float32, shape=[batch_size, FRAMES_PER_SAMPLE, NEFF])
        # init the model
        BiModel = Model(n_hidden, batch_size, p_keep_ff, p_keep_rc)
        # make inference of embedding
        embedding = BiModel.inference(in_data)
        saver = tf.train.Saver(tf.all_variables())
        sess = tf.Session()
        # restore the model
        saver.restore(sess, train_dir)
        tot_frame = N_frame * FRAMES_PER_SAMPLE
        # arrays to store output waveform
        out_audio1 = np.zeros([(tot_frame - 1) * hop_size + FRAME_SIZE])
        out_audio2 = np.zeros([(tot_frame - 1) * hop_size + FRAME_SIZE])
        mix = np.zeros([(tot_frame - 1) * hop_size + FRAME_SIZE])
        N_assign = 0

        embeddings = []
        input_data = []
        input_phase = []
        VAD_data = []

        while(1):
            data_batch = data_generator.gen_next()
            if data_batch is None:
                break
            in_data_np = np.concatenate(
                [np.reshape(item['Sample'], [1, FRAMES_PER_SAMPLE, NEFF])
                for item in data_batch])
            # phase info.
            in_phase_np = np.concatenate(
                [np.reshape(item['Phase'], [1, FRAMES_PER_SAMPLE, NEFF])
                for item in data_batch])
            # VAD info.
            VAD_data_np = np.concatenate(
                [np.reshape(item['VAD'], [1, FRAMES_PER_SAMPLE, NEFF])
                for item in data_batch])
            input_data.append(in_data_np)
            input_phase.append(in_phase_np)
            VAD_data.append(VAD_data_np)
            print('Session running')
            embedding_np, = sess.run(
                [embedding],
                feed_dict={in_data: in_data_np,
                           p_keep_ff: 1,
                           p_keep_rc: 1})
            print('Session Run')
            embeddings.append(embedding_np)

    # embeddings have been obtained are in the form 
    # {embedding_batch1, ..., embedding_batchn} where batch_k = [batch_size*100, 129, EMBEDDING_D]
    # for every chunk of frames of data
    step = 0 # hack to make the code work, should be removed in refactoring
    for batch_no in range(len(embeddings)):
        embedding_batch = embeddings[batch_no]
        VAD_batch = VAD_data[batch_no]
        phase_batch = input_phase[batch_no]
        in_data_batch = input_data[batch_no]

        print("batch_no", batch_no)
        for frame in range(VAD_batch.shape[0]):
            print("Processing step", step)
            embedding_np = embedding_batch[frame*100:(frame+1)*100,:,:]
            VAD_data_np = VAD_batch[frame, :, :].reshape((-1, 100, 129))
            in_phase_np = phase_batch[frame, :, :].reshape(-1, 100, 129)
            in_data_np = in_data_batch[frame, :, :].reshape(-1, 100, 129)
            # get active TF-bin embedding according to VAD
            if silence_mask:
                embedding_ac = [embedding_np[i, j, :]
                                for i, j in itertools.product(
                                    range(FRAMES_PER_SAMPLE), range(NEFF))
                                if VAD_data_np[0, i, j] == 1]
            else:
                embedding_ac = embedding_np.reshape(-1, 40)
            
            print('Starting Clustering')
            if(sep_flag[step] == 1):
                # if the frame need to be seperated
                # cluster the embeddings
                if embedding_ac == []:
                    break
                kmean = KMeans(n_clusters=2, random_state=0).fit(embedding_ac)
                                       
            else:
                # if the frame don't need to be seperated
                # don't split the embeddings
                kmean = KMeans(n_clusters=1, random_state=0).fit(embedding_ac)

            mask = np.zeros([FRAMES_PER_SAMPLE, NEFF, 2])
            ind = 0
            if N_assign == 0:
                # print('N_assign is 0 in step', str(step))
                # if their is no existing speaker in previous frame
                center = kmean.cluster_centers_
                N_assign = center.shape[0]
            
            elif N_assign == 1:
                # print('N_assign is 1 in step', str(step))

                # if their is one speaker in previous frame
                center_new = kmean.cluster_centers_
                # assign the embedding for a speaker to the speaker with the
                # closest centroid in previous frames
                if center_new.shape[0] == 1:
                    # update and smooth the centroid for 1 speaker
                    center = 0.7 * center + 0.3 * center_new
                else:
                    # update and smooth the centroid for 2 speakers
                    N_assign = 2
                    # compute their relative affinity
                    cor = np.matmul(center_new, np.transpose(center))
                    # ipdb.set_trace()
                    if(cor[1] > cor[0]):
                        # rearrange their sequence if not consistant with
                        # previous frames
                        kmean.cluster_centers_ = np.array(
                            [kmean.cluster_centers_[1],
                             kmean.cluster_centers_[0]])
                        kmean.labels_ = (kmean.labels_ == 0).astype('int')
                    center = kmean.cluster_centers_

            else:
                # print('N_assign is 2 in step', str(step))
                # two speakers have appeared
                center_new = kmean.cluster_centers_
                # cor = np.matmul(center_new[0, :], np.transpose(center))
                cor = np.matmul(center_new, np.transpose(center))
                # print('Correlation btw new and old centers, ', cor)
                # rearrange their sequence if not consistant with previous
                # frames
                if(cor[0,1] > 0.85 or cor[1,0] > 0.85):
                # if(cor[1] > cor[0]):
                    # print('centers exhanged in step ', str(step))
                # if(cor[0] > cor[1]):
                    if(sep_flag[step] == 1):
                        kmean.cluster_centers_ = np.array(
                            [kmean.cluster_centers_[1],
                             kmean.cluster_centers_[0]])
                        kmean.labels_ = (kmean.labels_ == 0).astype('int')
                    else:
                        kmean.labels_ = (kmean.labels_ == 1).astype('int')
                        
                # need permutation of their order(Oracle)
                if(oracal_p[step]):
                    kmean.cluster_centers_ = np.array(
                        [kmean.cluster_centers_[1],
                         kmean.cluster_centers_[0]])
                    kmean.labels_ = (kmean.labels_ == 0).astype('int')
                else:
                    kmean.labels_ = (kmean.labels_ == 1).astype('int')
                
                center = center * 0.7 + 0.3 * kmean.cluster_centers_
            print('Clustering Done')

            if ratio_mask:
                ## Soft Clustering
                # 4.1 Get centroids by applying K-Means 
                # 4.3 Iterate over all embeddings in the given frame
                # 4.3.1 Determine the euclidean/frobneus distance between centroids = d_btw_centers
                # 4.3.2 For a given embedding calculate its distance d1 from center1 and distance d2 from center 2
                # 4.3.3 Weight1 = d2/d_btw_centers,  Weight2 = d1/d_btw_centers  ----- if point is between two centroids, 
                #                                                                       then greater weight will be given to 
                #                                                                       the centroid to which the point is nearer
                #                                                                       c1 <------------d1----------->p1<-d2-->c2
                #                                                                          <-----------d_btw_centers------------>
                #                                                                       As p1 is nearer to c2 hence w2 > w1
                #                                                                       in this case maximum weight will be one
                #                                                                       If point is on the side of one centre, then
                #                                                                       weight to the center to which point is nearer may be greater than one.

                d_btw_centers = distance(kmean.cluster_centers_[0], kmean.cluster_centers_[1])
                w1 = np.zeros((embedding_ac.shape[0]))
                w2 = np.zeros((embedding_ac.shape[0]))
                for i in range(embedding_ac.shape[0]):
                    tf_bin = embedding_ac[i,:]
                    d1 = distance(tf_bin, kmean.cluster_centers_[0])
                    d2 = distance(tf_bin, kmean.cluster_centers_[1])
                    if (d1 <= d_btw_centers and d2 <= d_btw_centers):
                        w1[i] = d2/d_btw_centers
                        w2[i] = d1/d_btw_centers
                    elif (d1 > d_btw_centers):
                        w2[i] = d1/d_btw_centers
                        w1[i] = -d2/d_btw_centers
                    elif (d2 > d_btw_centers):                    
                        w1[i] = d2/d_btw_centers
                        w2[i] = -d1/d_btw_centers
                # w1 = np.sqrt(np.abs(w1))
                # w2 = np.sqrt(np.abs(w2))
                        
                # transform the clustering result and VAD info. into masks
                for i in range(FRAMES_PER_SAMPLE):
                    for j in range(NEFF):
                        if silence_mask:
                            if VAD_data_np[0, i, j] == 1:
                                mask[i, j, 0] = w1[ind]
                                mask[i, j, 1] = w2[ind]
                                ind += 1     
                        else:
                            mask[i, j, 0] = w1[ind]
                            mask[i, j, 1] = w2[ind]
                            ind += 1
            else:
                # # transform the clustering result and VAD info. into masks
                for i in range(FRAMES_PER_SAMPLE):
                    for j in range(NEFF):
                        if silence_mask:
                            if VAD_data_np[0, i, j] == 1:
                                mask[i, j, kmean.labels_[ind]] = 1
                                ind += 1     
                        else:
                            mask[i, j, kmean.labels_[ind]] = 1
                            ind += 1  
  
            if CONFUSION_MATRIX:
                # if labels and masks order is not correct, 
                # swap the order of labels
                if np.sum(np.equal(Y_data_np[:,:,:,0], mask[:,:,0])) < np.sum(np.equal(Y_data_np[:,:,:,0], mask[:,:,1])):
                    Y_data_np = Y_data_np[:,:,:,[1,0]]

                z = confusion_matrix(Y_data_np, mask)
                print(VAD_data_np.shape)
                print("Silence Bins", np.sum(VAD_data_np[0,:,:]))
                print("Confusion Matrix \n",z)
                print("Accuracy: ", (z[0,0] - np.sum(VAD_data_np) + z[1,1])/(np.sum(z) - np.sum(VAD_data_np)))

            for i in range(FRAMES_PER_SAMPLE):
                # apply the mask and reconstruct the waveform
                tot_ind = step * FRAMES_PER_SAMPLE + i
                # ipdb.set_trace()
                # amp = (in_data_np[0, i, :] *
                #        data_batch[0]['Std']) + data_batch[0]['Mean']
                amp = in_data_np[0, i, :] * GLOBAL_STD + GLOBAL_MEAN
                
                out_data1 = (mask[i, :, 0] * amp *
                             VAD_data_np[0, i, :])
                out_data2 = (mask[i, :, 1] * amp *
                             VAD_data_np[0, i, :])
                out_mix = amp
                out_data1_l = 10 ** (out_data1 / 20) / AMP_FAC
                out_data2_l = 10 ** (out_data2 / 20) / AMP_FAC
                out_mix_l = 10 ** (out_mix / 20) / AMP_FAC

                out_stft1 = out_data1_l * in_phase_np[0, i, :]
                out_stft2 = out_data2_l * in_phase_np[0, i, :]
                out_stft_mix = out_mix_l * in_phase_np[0, i, :]

                con_data1 = out_stft1[-2:0:-1].conjugate()
                con_data2 = out_stft2[-2:0:-1].conjugate()
                con_mix = out_stft_mix[-2:0:-1].conjugate()

                out1 = np.concatenate((out_stft1, con_data1))
                out2 = np.concatenate((out_stft2, con_data2))
                out_mix = np.concatenate((out_stft_mix, con_mix))
                frame_out1 = np.fft.ifft(out1).astype(np.float64)
                frame_out2 = np.fft.ifft(out2).astype(np.float64)
                frame_mix = np.fft.ifft(out_mix).astype(np.float64)

                out_audio1[tot_ind * hop_size:tot_ind * hop_size + FRAME_SIZE] += frame_out1 * 0.5016
                out_audio2[tot_ind * hop_size:tot_ind * hop_size + FRAME_SIZE] += frame_out2 * 0.5016
                mix[tot_ind * hop_size:tot_ind * hop_size + FRAME_SIZE] += frame_mix * 0.5016
            step += 1
            # print(kmean.cluster_centers_)
            # print('points in first cluster', np.sum(kmean.labels_ == 1))
            # print(set(kmean.labels_))

    length = len(frames)
    print('Writing Files')

    #librosa.output.write_wav(os.path.join(dir,'mix_mp.wav'), mix * (1/np.max(mix)), SAMPLING_RATE)
    sf.write(os.path.join(dir,'SEN_1.wav'), out_audio1[:length] * (1/np.max(out_audio1)), SAMPLING_RATE)
    sf.write(os.path.join(dir,'SEN_2.wav'), out_audio2[:length] * (1/np.max(out_audio2)), SAMPLING_RATE)



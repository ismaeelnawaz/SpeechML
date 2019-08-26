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
import argparse

import numpy as np
import librosa
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from multiprocessing import Process

import tensorflow as tf
import pyaudio

import itertools
from numpy.lib import stride_tricks

from AudioSampleReader2 import AudioSampleReader2
from AudioSampleReader import AudioSampleReader
from model import Model

from GlobalConstont import *

import matplotlib.pyplot as plt
import soundfile as sf

# not useful during sample test
sum_dir = 'sum'
# dir to load model
# train_dir = os.path.join('/home/usman/repos/SEN', 'train_l5_500', 'model.ckpt-3800')
train_dir = os.path.join('Model', 'model.ckpt-2000')

lr = 0.00001  # not useful during test
n_hidden = 500  # hidden state size
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

def out_put(N_frame, dir, input_audio=False, frames=None, ratio_mask=False, CONFUSION_MATRIX=True, silence_mask=True, visualize_embedding=False, visualize_tsne =False, spectrograms=False, time_domain_graphs=True):
    '''Use trained model to infer N _frame chuncks of
    frames of input audio'''
    if input_audio:
        assert frames is not None
    with tf.Graph().as_default():
        print('OUTPUT STARTED')
        # feed forward keep prob
        p_keep_ff = tf.placeholder(tf.float32, shape=None)
        # recurrent keep prob
        p_keep_rc = tf.placeholder(tf.float32, shape=None)

        # audio sample generator
        if input_audio:
            data_generator = AudioSampleReader(dir, input_audio=True, frames=frames)
        else:
            if CONFUSION_MATRIX:
                data_generator = AudioSampleReader2(os.path.join(dir, 'mix.wav'), 
                os.path.join(dir, 'oracle_fem.wav'), os.path.join(dir, 'oracle_male.wav'))
            else:
                data_generator = AudioSampleReader(os.path.join(dir, 'mix.wav'))

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
        out_audio1 = np.zeros([(tot_frame - 1) * hop_size + FRAME_SIZE], dtype=np.float32)
        out_audio2 = np.zeros([(tot_frame - 1) * hop_size + FRAME_SIZE], dtype=np.float32)

        mix = np.zeros([(tot_frame - 1) * hop_size + FRAME_SIZE])
        N_assign = 0

        # for every chunk of frames of data
        for step in range(N_frame):
            # import ipdb; ipdb.set_trace()
            data_batch = data_generator.gen_next()
            if data_batch is None:
                break
            # log spectrum info.
            if CONFUSION_MATRIX:
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
                # Labels info.
                Y_data_np = np.concatenate(
                    [np.reshape(item['Labels'], [1, FRAMES_PER_SAMPLE, NEFF, 2])
                    for item in data_batch])
                Y_data_np = Y_data_np.astype('int')
            else:
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

            # get inferred embedding using trained model
            # with keep prob = 1
            print('Session running')
            embedding_np, = sess.run(
                [embedding],
                feed_dict={in_data: in_data_np,
                           p_keep_ff: 1,
                           p_keep_rc: 1})
            print('Session Run')
            # embeddings_v = np.reshape(embedding_np, [-1, FRAMES_PER_SAMPLE*NEFF, EMBBEDDING_D])
            # Y_v = np.reshape(Y_data_np, [-1, FRAMES_PER_SAMPLE*NEFF, 2]).astype(float)
            # vvT = np.matmul(embeddings_v, np.transpose(
            #         embeddings_v, [0, 2, 1]))
            # yyT = tf.matmul( Y_v, tf.transpose(
            #             Y_v, [0, 2, 1]))
            # # loss = vvT - yyT
            # print(embedding_np.shape)
            # print(vvT.shape)
            # print(yyT.shape)
            # print(loss)

            # ipdb.set_trace()
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
                # import ipdb; ipdb.set_trace()
                if embedding_ac == []:
                    break
                kmean = KMeans(n_clusters=2, random_state=0).fit(embedding_ac)

                if visualize_tsne:
                    tsne = TSNE(n_components=3, n_iter=250)
                    tsne_data = tsne.fit_transform(embedding_ac)
                    fig = plt.figure(1, figsize=(8, 6))
                    ax = Axes3D(fig, elev=-150, azim=110)
                    # ax.scatter(pca_Data[:, 0], pca_Data[:, 1], pca_Data[:, 2],
                    #            c=kmean.labels_, cmap=plt.cm.Paired)

                    ax.scatter(tsne_data[:, 0], tsne_data[:, 1], tsne_data[:, 2],
                            c=kmean.labels_.astype(np.float))
                    ax.set_title('Embedding visualization using TSNE')
                    ax.set_xlabel('1st pc')
                    ax.set_ylabel('2nd pc')
                    ax.set_zlabel('3rd pc')
                    fig.savefig(os.path.join(dir,'tsne_frame' +str(step) + '.png'))
                    plt.clf()
                                       
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
                    print(cor)
                    # ipdb.set_trace()
                    if(cor[0] > cor[1]):
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
                cor = np.matmul(center_new, np.transpose(center))
                # print('Correlation btw new and old centers, ', cor)
                # rearrange their sequence if not consistant with previous
                # frames
                print(cor)
                if(cor[0,1] > 0.85 or cor[1,0] > 0.85):
                    # print('centers exhanged in step ', str(step))
                # if(cor[0] > cor[1]):
                    if(sep_flag[step] == 1):
                        kmean.cluster_centers_ = np.array(
                            [kmean.cluster_centers_[1],
                             kmean.cluster_centers_[0]])
                        kmean.labels_ = (kmean.labels_ == 0).astype('int')
                    else:
                        kmean.labels_ = (kmean.labels_ == 1).astype('int')

                elif(cor[0,1] > cor[0,0] and cor[1,0] > cor[1,1]):
                    # print('centers exhanged in step ', str(step))
                # if(cor[0] > cor[1]):
                    if(sep_flag[step] == 1):
                        kmean.cluster_centers_ = np.array(
                            [kmean.cluster_centers_[1],
                             kmean.cluster_centers_[0]])
                        kmean.labels_ = (kmean.labels_ == 0).astype('int')
                    else:
                        kmean.labels_ = (kmean.labels_ == 1).astype('int')
                        
                # # need permutation of their order(Oracle)
                # if(oracal_p[step]):
                #     kmean.cluster_centers_ = np.array(
                #         [kmean.cluster_centers_[1],
                #          kmean.cluster_centers_[0]])
                #     kmean.labels_ = (kmean.labels_ == 0).astype('int')
                # else:
                #     kmean.labels_ = (kmean.labels_ == 1).astype('int')
                
                # center = center * 0.7 + 0.3 * kmean.cluster_centers_
            print('Clustering Done')

            # Plotting Clustered Embeddings
            # visualization of embeddings using PCA
            if visualize_embedding:
                pca_Data = PCA(n_components=3).fit_transform(embedding_ac)
                pca_Data_2 = PCA(n_components=2).fit_transform(embedding_ac)
                pca_centers = PCA(n_components=2).fit_transform(kmean.cluster_centers_)
                fig = plt.figure(1, figsize=(8, 6))
                ax = Axes3D(fig, elev=-150, azim=110)
                # ax.scatter(pca_Data[:, 0], pca_Data[:, 1], pca_Data[:, 2],
                #            c=kmean.labels_, cmap=plt.cm.Paired)

                ax.scatter(pca_Data[:, 0], pca_Data[:, 1], pca_Data[:, 2],
                        c=kmean.labels_.astype(np.float))
                ax.set_title('Embedding visualization using the first 3 PCs')
                ax.set_xlabel('1st pc')
                ax.set_ylabel('2nd pc')
                ax.set_zlabel('3rd pc')
                ax.set_xlim([-1.0,1.0])
                ax.set_ylim([-1.0,1.0])
                ax.set_zlim([-1.0,1.0])
                fig.savefig(os.path.join(dir,'pca_frame' +str(step) + '.png'))
                plt.clf()

                fig = plt.figure()
                plt.scatter(pca_Data_2[:, 0], pca_Data_2[:, 1],
                        c=kmean.labels_.astype(np.float))
                plt.scatter(pca_centers[:,0], pca_centers[:,1], c='r')
                plt.xlim([-1.0,1.0])
                plt.ylim([-1.0,1.0])
                fig.savefig(os.path.join(dir,'pca_2_frame' +str(step) + '.png'))
                plt.clf()

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


            if spectrograms:
                ## Plotting Masks
                fig = plt.figure()
                plt.imshow(mask[:, :, 0])
                plt.title('Speaker 1 mask.')
                plt.colorbar()
                plt.gca().invert_yaxis()
                fig.savefig(os.path.join(dir, 'Mask Speaker 1 Frame ' + str(step)+'.png'))
                plt.clf()

                fig = plt.figure()
                plt.imshow(mask[:, :, 1])
                plt.title('Speaker 2 mask.')
                plt.colorbar()
                plt.gca().invert_yaxis()
                fig.savefig(os.path.join(dir, 'Mask Speaker 2 Frame ' + str(step)+'.png'))
                plt.clf()

                fig = plt.figure()
                plt.imshow(in_data_np[0,:,:])
                plt.title('Input Spectrogram.')
                plt.colorbar()
                plt.gca().invert_yaxis()
                fig.savefig(os.path.join(dir, 'Input Spectrogram Frame ' + str(step)+'.png'))
                plt.clf()

                out_spec1 = in_data_np*mask[:,:,0]
                fig = plt.figure()
                plt.imshow(out_spec1[0,:,:])
                plt.title('Output 1 Spectrogram.')
                plt.colorbar()
                plt.gca().invert_yaxis()
                fig.savefig(os.path.join(dir, 'Output Spectrogram 1 Frame ' + str(step)+'.png'))
                plt.clf()


                out_spec2 = in_data_np*mask[:,:,1]
                fig = plt.figure()
                plt.imshow(out_spec2[0,:,:])
                plt.title('Output 2 Spectrogram.')
                plt.colorbar()
                plt.gca().invert_yaxis()
                fig.savefig(os.path.join(dir, 'Output Spectrogram 2 Frame ' + str(step)+'.png'))
                plt.clf()

                fig = plt.figure()
                plt.imshow(VAD_data_np[0, :, :])
                plt.title('Silence Mask')
                plt.colorbar()
                plt.gca().invert_yaxis()
                fig.savefig(os.path.join(dir, 'Silence Frame ' + str(step)+'.png'))
                plt.clf()

                # mask[0,0,0] = np.max(in_data_np)
                # mask[0,0,1] = np.max(in_data_np)
                # mask[99,0,0] = np.min(in_data_np)
                # mask[99,0,1] = np.min(in_data_np)
                # VAD_data_np[0,0,0] = np.max(in_data_np)
                # VAD_data_np[0,99,0] = np.min(in_data_np)

                fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(16,8))
                im2 = ax[0][0].imshow(VAD_data_np[0,:,:])
                ax[0][0].set_title('Silence Mask')
                ax[0][1].imshow(mask[:, :, 0])
                ax[0][1].set_title('Mask S1')
                ax[0][2].imshow(mask[:, :, 1])
                ax[0][2].set_title('Mask S2')
                im = ax[1][0].imshow(in_data_np[0,:,:])
                ax[1][0].set_title('Input Spectrogram')
                ax[1][1].imshow(out_spec1[0,:,:])
                ax[1][1].set_title('Output 1 Spectrogram')
                ax[1][2].imshow(out_spec2[0,:,:])
                ax[1][2].set_title('Output 2 Spectrogram')

                ax[0][0].invert_yaxis()
                fig.colorbar(im, ax=ax.ravel().tolist())
                # fig.colorbar(im2, ax=ax.ravel().tolist())

                fig.savefig(os.path.join(dir, 'Spectrograms Frame ' + str(step) + ' .png'))
                plt.clf()

                plt.close()

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
                plt.close('all')
            # print(kmean.cluster_centers_)
            # print('points in first cluster', np.sum(kmean.labels_ == 1))
            # print(set(kmean.labels_))


        print('Writing Files')
        # mix, _ = librosa.load(os.path.join(dir, 'mix.wav'), 8000)
        # length = len(mix)

        # fig, ax = plt.subplots(4,1, sharex=True, figsize=(6,12))
        # ax[0].plot(out_audio1[:length])
        # ax[0].set_title('Output Speaker 1')

        # ax[1].plot(out_audio2[:length])
        # ax[1].set_title('Output Speaker 2')

        # ax[2].plot(out_audio1[:length] + out_audio2[:length])
        # ax[2].set_title('Mixed Output')

        # ax[3].plot(mix*(1/np.max(mix)), color='r')
        # ax[3].set_title('Mixed Input')

        min_len = min(len(out_audio1), len(out_audio2))
        if frames is not None:
            min_len = min(len(out_audio1), len(out_audio2), len(frames))
            librosa.output.write_wav(os.path.join(dir, 'audio81.wav'), frames[:min_len], SAMPLING_RATE)

        # return out_audio1, out_audio2
        if np.sum(np.square(out_audio1)) > np.sum(np.square(out_audio2)):
            print('out_audio1')
            play_audio((out_audio1*(1/np.max(out_audio1)))[:min_len])
        else:
            print('out_audio2')
            play_audio((out_audio2*(1/np.max(out_audio2)))[:min_len])
        
        # out_audio1 = out_audio1[:length]*1/np.max(out_audio1[:length])
        # out_audio2 = out_audio2[:length]*1/np.max(out_audio2[:length])

        librosa.output.write_wav(os.path.join(dir,'SEN_audio8_1.wav'), out_audio1, SAMPLING_RATE)
        librosa.output.write_wav(os.path.join(dir,'SEN_audio8_2.wav'), out_audio2, SAMPLING_RATE)
        # librosa.output.write_wav(os.path.join(dir,'out_1.wav'), out_audio1[:length], SAMPLING_RATE)
        # librosa.output.write_wav(os.path.join(dir,'out_2.wav'), out_audio2[:length], SAMPLING_RATE)



# if __name__ == '__main__':
#     # for i in range(5, 5 + NUMBER_OF_SAMPLES):
#     #     out_put(10, os.path.join('Testing', 'Single Speaker', str(i)))
#     parser = argparse.ArgumentParser()
#     # parser.add_argument("record_seconds", type=int, default=10)
#     parser.add_argument("--recordFromMic", action='store_true')
#     args = parser.parse_args()
#     record_mic = args.recordFromMic
#     if record_mic:
#         audio = record_audio()
#         p1 = Process(target=play_audio, args=(audio))
#         p1.start()
#         # p2 = Process(librosa.core.resample)
#         p2 = Process(target=out_put, args=(100, os.path.join('/home/usman/repos/SEN', 'results'), silence_mask=False, 
#         visualize_embedding=False, CONFUSION_MATRIX=False, input_audio=True, frames=audio))
#         p2.start()
#         p1.join()
#         p2.join()
#         # play_audio(audio)
#         # start = time.time()
#         # out_put(100, os.path.join('/home/usman/repos/SEN', 'results'), silence_mask=False, 
#         # visualize_embedding=False, CONFUSION_MATRIX=False, input_audio=True, frames=audio)
#         # print('Processing Time ', time.time() - start)



#     else:
#         # samples = np.load('mix_train.npy')
#         # for i in range(12):
#         #     audio = samples[i,:]
#         try:
#             os.mkdir(os.path.join('/home/usman/repos/SEN', 'results', 'recorded_audio'))
#         except:
#             pass
#         audio, _ = librosa.load(os.path.join('/home/usman/repos/SEN', 'Audio_8.wav'), sr=SAMPLING_RATE)
#         # noise, _ = librosa.load(os.path.join('/media/usman/Dataset/Noise/files', 'PRESTO_16k', 'PRESTO', 'ch01.wav'), sr=SAMPLING_RATE)
#         # start = time.time()
#         # audio_start = 0*8000
#         # audio_end = 30*8000
#         # audio = np.array(audio[audio_start:audio_end], dtype=np.float32)
#         # audio = audio + 10*noise[:len(audio)]
#         play_audio(audio)
#         # librosa.output.write_wav(os.path.join('/home/usman/repos/SEN', 'results', 'recorded_audio', 'cafe.wav'), audio, sr=SAMPLING_RATE)
#         start = time.time()
#         out_put(100, os.path.join('/home/usman/repos/SEN/', 'results'), ratio_mask=False, silence_mask=False, 
#         visualize_embedding=True, CONFUSION_MATRIX=False, input_audio=True, frames=audio)
#         print('Processing Time ', time.time() - start)
#         # dir = os.path.join('Testing', 'Outputs1', str(i))
#         # mix, _ = librosa.load(os.path.join(dir, 'mix.wav'), 8000)
#         # s1, _ = librosa.load(os.path.join(dir, 'oracle_fem.wav'), 8000)
#         # s2, _ = librosa.load(os.path.join(dir, 'oracle_male.wav'), 8000)
#         # o1, _ = librosa.load(os.path.join(dir, 'out_1_mod.wav'),8000)
#         # o2, _ = librosa.load(os.path.join(dir, 'out_2_mod.wav'), 8000)
#         # # print((s1.size), len(s2), len(mix), len(o1), len(o2))
#         # length = len(mix)
#         # sf.write(os.path.join(dir,'mix.wav'), mix[:length], SAMPLING_RATE)
#         # sf.write(os.path.join(dir,'oracle_fem.wav'), s1[:length], SAMPLING_RATE)
#         # sf.write(os.path.join(dir,'oracle_male.wav'), s2[:length], SAMPLING_RATE)
#         # sf.write(os.path.join(dir,'out_1_mod.wav'), o1[:length], SAMPLING_RATE)
#         # sf.write(os.path.join(dir,'out_2_mod.wav'), o2[:length], SAMPLING_RATE)

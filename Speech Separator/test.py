import soundfile as sf
import pyaudio
import numpy as np
from scipy.signal import resample

def record_audio(self):
        CHUNK = 1024
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        RATE = 44100
        SAMPLING_RATE = 8000
        RECORD_SECONDS = 3
        WAVE_OUTPUT_FILENAME = "input.wav"

        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        frames = []
        #global audio_8k

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

        # resampling_factor = RATE/SAMPLING_RATE
        # audio_8k = resample(audio, int(len(audio)/resampling_factor))


        sf.write(os.path.join(WAVE_OUTPUT_FILENAME), audio_8k, SAMPLING_RATE)
        

import os

import numpy as np
from matplotlib import pyplot as plt
import librosa
import librosa.display
import moviepy.editor as mp
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from scipy.signal import butter, filtfilt, find_peaks

def create_OnsetDetector(videofile):

    detector = OnsetDetector()
    detector.load(videofile)

    return detector


class OnsetDetector(object):

    def _loadFromFile(self, filename):
        self.videofile = filename
        self.clip      = mp.VideoFileClip(filename)
        self.waveform  = self.clip.audio.to_soundarray()
        self.fs        = int(self.waveform.shape[0] // self.clip.duration)

    def __init__(self):

        self.clip     = None  # Video clip
        self.waveform = None  # Audio waveform
        self.fs       = None  # Sampling frequency

    def _extract(self, folder, onsets):

        # TODO: Fix so that last onsets is properly extracted
        for i in range(len(onsets) - 1):
            output = os.path.join(folder, f'onset_%d.mov' % (i))
            ffmpeg_extract_subclip(self.videofile, onsets[i], onsets[i + 1], output)

    def _filter(self, waveform, Wn, btype='lowpass', order=5):
        nyq = self.fs / 2
        Wn = np.array(Wn) / nyq
        b, a = butter(order, Wn, btype, analog=False)
        output = filtfilt(b, a, waveform.T).T

        return output

    def load(self, filename):
        self._loadFromFile(filename)

    def detect(self, bandpass=False):
        mono = librosa.to_mono(self.waveform.T)
        
        if bandpass:
            mono = self._filter(mono, (20, 10000), btype='bandpass', order=5)
        
        onset_frames = librosa.onset.onset_detect(y=mono, sr=self.fs, 
            units='frames', pre_max=10, post_max=10, pre_avg=5,
            post_avg=50, delta=0.5, backtrack=True)

        times = librosa.frames_to_time(onset_frames, sr=self.fs)

        return times

    def extract(self, folder, onsets):
        self._extract(folder, onsets)

    def displayWaveform(self):
        plt.figure()
        mono = librosa.to_mono(self.waveform.T)
        librosa.display.waveplot(mono, self.fs)
        plt.show()

    def displayOnsets(self, onsets):
        plt.figure()
        mono = librosa.to_mono(self.waveform.T)
        librosa.display.waveplot(mono, self.fs)
        plt.plot(onsets, np.zeros(onsets.shape), 'ro')
        plt.show()

if __name__ == '__main__':
    
    videofile = './data/training/training_centre_synced.mov'
    detector = create_OnsetDetector(videofile)
    onset_times = detector.detect(bandpass=True)
    detector.displayOnsets(onset_times)
    detector.extract('./data/training/segmented/', onset_times)



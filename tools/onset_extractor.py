# This script multiple video using audio output
import argparse
from typing import Iterable

import pygame
import librosa
import librosa.display
from moviepy.audio.AudioClip import AudioClip
import moviepy.editor as mp
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from matplotlib import pyplot as plt
import numpy as np from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq

parser = argparse.ArgumentParser()
parser.add_argument('videofile', type=str, 
    help='Video file')
parser.add_argument('--output', type=str, default=None,
    help='Output file name')
parser.add_argument('--visualise', action='store_true', default=False,
    help='Show visualisations')


def parse_input(prompt):
    n = input(prompt)
    while not n.isdigit():
        n = input('Please enter a digit: ')
    
    n = int(n)


class AudioProcessing(object):


    def __init__(self, waveform, fs, fps):
        self.waveform = waveform
        self.fs = int(fs)
        self.fps = fps

    def get_times(self):
        return librosa.times_like(self.waveform.T, sr=self.fs)
    
    def filter(self, Wn, btype='lowpass', order=5):
        nyq = self.fs / 2
        Wn = np.array(Wn) / nyq
        b, a = butter(order, Wn, btype, analog=False)
        output = filtfilt(b, a, self.waveform.T).T

        return AudioProcessing(output, self.fs, self.fps)

    def detect_onsets(self):
        mono = librosa.to_mono(self.waveform.T)
        onset_frames = librosa.onset.onset_detect(y=mono, sr=self.fs, 
            units='frames', pre_max=500, post_max=500, pre_avg=100,
            post_avg=100, delta=0.01)
        times = librosa.frames_to_time(onset_frames, sr=self.fs)
        return times

    def plot_onsets(self, peaks):
        mono = librosa.to_mono(self.waveform.T)
        librosa.display.waveplot(mono, self.fs)
        plt.plot(peaks, np.zeros(peaks.shape), 'ro')

    def plot_timedomain(self):
        plt.figure()
        mono = librosa.to_mono(self.waveform.T)
        librosa.display.waveplot(mono, self.fs)

    def plot_fft(self):
        plt.figure()
        mono = librosa.to_mono(self.waveform.T)
        N = self.waveform.shape[0]  # number of samping points
        T = 1 / self.fs
        yf = fft(mono)
        xf = fftfreq(N, T)[:N//2]
        plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))

    def plot_spectrogram(self):
        plt.figure()
        mono = librosa.to_mono(self.waveform.T)
        D = librosa.stft(mono)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(S_db, y_axis='log', x_axis='time')
        plt.colorbar(img, format="%+2.f dB")


class VideoSyncGUI(tk.Frame):

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.mainfig = Figure(figsize=(5, 4), dpi=100)

        self.canvas = FigureCanvasTkAgg(self.mainfig, master=self)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def update(self):
        self.canvas.update()


def sync_videos(videofile, visualise=False, output='output.mov'):

    clip = mp.VideoFileClip(videofile)
    waveform = clip.audio.to_soundarray()  
    fs = waveform.shape[0] // clip.duration
    audio = AudioProcessing(waveform, fs, clip.fps)
    bandpass = audio.filter((3700, 4000), btype='bandpass', order=7)

    peaks = bandpass.detect_onsets()
    n1 = parse_input('Select start peak: ')
    n2 = parse_input('Select end peak (-1 for EOF): ')
    
    if n1 >= len(peaks):
        print(f'error - peak {n1} does not exists')
        err = True

    if n2 >= len(peaks) or (n2 < -1):
        print(f'error - peak {n2} does not exists or invalid')
        err = True

    if n2 != -1 and n1 >= n2:
        print(f'error - peak2 must be greater than peak1')
        err = True
    
    if err:
        print('fatal error has occured')
        exit(0)


    if output:
        ffmpeg_extract_subclip(videofile, peaks[n], clip.duration, output)
            

if __name__ == '__main__':
    args = parser.parse_args()
    sync_videos(args.videofile, visualise=args.visualise, output=args.output)

    # GUI
    root = tk.Tk()
    VideoSyncGUI(root).pack(side="top", fill="both", expand=True)
    root.mainloop()


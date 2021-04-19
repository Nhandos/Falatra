import argparse
import tkinter as tk

from matplotlib.backends.backend_tkagg import(
        FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

import numpy as np
import librosa
import librosa.display
import moviepy.editor as mp
from scipy.signal import butter, filtfilt


parser = argparse.ArgumentParser()
parser.add_argument('videofile', type=str, 
    help='Video file')
parser.add_argument('--output', type=str, default=None,
    help='Output file name')
CLI_ARGS=parser.parse_args()


def butterworth_filter(waveform, fs, Wn, btype='lowpass', order=5):
    nyq = fs / 2
    Wn = np.array(Wn) / nyq
    b, a = butter(order, Wn, btype, analog=False)
    output = filtfilt(b, a, waveform)

    return output 


def detect_onsets(waveform, fs):
    onset_frames = librosa.onset.onset_detect(y=waveform, sr=fs, 
        units='frames', pre_max=500, post_max=500, pre_avg=100,
        post_avg=100, delta=0.01)
    times = librosa.frames_to_time(onset_frames, sr=fs)

    return times


class VideoSyncGUI(tk.Frame):

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.initUI()

        # load video clip
        self.clip = mp.VideoFileClip(CLI_ARGS.videofile)
        self.waveform = librosa.to_mono(self.clip.audio.to_soundarray().T)
        self.fs = int(self.waveform.shape[0] // self.clip.duration)

        # detect and plot onsets
        times = librosa.times_like(self.waveform, self.fs)
        bandpass = butterworth_filter(self.waveform, self.fs, (3700, 4000), btype='bandpass', order=7)
        onsets = detect_onsets(bandpass, self.fs)
        librosa.display.waveplot(bandpass, sr=self.fs, ax=self.mainplot)
        self.mainplot.plot(onsets, np.zeros(onsets.shape), 'ro')

        self.update()

    def initUI(self):

        # UI elements
        self.mainfig = Figure(figsize=(5, 4), dpi=100)
        self.mainplot = self.mainfig.add_subplot()
        self.canvas = FigureCanvasTkAgg(self.mainfig, master=self)  # A tk.DrawingArea.
        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)

        # UI layout (packing order is important)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.toolbar.pack(side=tk.BOTTOM,fill=tk.X)

    def update(self):
        self.canvas.draw()
        self.toolbar.update()


if __name__ == '__main__':
    # GUI
    root = tk.Tk()
    VideoSyncGUI(root).pack(side="top", fill="both", expand=True)
    root.mainloop()


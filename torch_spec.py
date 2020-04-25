import os
import torch
import numpy as np
import soundfile as sf
import torch.nn.functional as F
import librosa
from librosa.filters import mel
from librosa.util import pad_center
from scipy.signal import get_window
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import LogarithmicFilteredSpectrogramProcessor


# same with librosa
class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft and
    Jongwook Kim's https://github.com/jongwook/onsets-and-frames"""
    def __init__(self, filter_length, hop_length, win_length=None, window='hann'):
        super(STFT, self).__init__()
        if win_length is None:
            win_length = filter_length

        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])

        if window is not None:
            assert(filter_length >= win_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.forward_basis.require_grad = False

    def forward(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            self.forward_basis,
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.atan2(imag_part.data, real_part.data)

        return magnitude, phase
    

class MelSpectrogram(torch.nn.Module):
    """adapted from Jongwook Kim's https://github.com/jongwook/onsets-and-frames"""
    def __init__(self, n_mels, sample_rate, filter_length, hop_length, win_length=None, mel_fmin=0.0, mel_fmax=None):
        super(MelSpectrogram, self).__init__()
        self.stft = STFT(filter_length, hop_length, win_length)

        mel_basis = mel(sample_rate, filter_length, n_mels, mel_fmin, mel_fmax, htk=True)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def forward(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: torch.FloatTensor with shape (B, T) in range [-1, 1]
        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, T, n_mels)
        """

        self.stft.to(y.device)
        self.mel_basis = self.mel_basis.to(y.device)

        magnitudes, phases = self.stft(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = torch.log(torch.clamp(mel_output, min=1e-5))
        return mel_output


class LMLFSpectrogram(torch.nn.Module):
    """adapted from madmom package: https://github.com/CPJKU/madmom => madmom.audio.spectrogram.LogarithmicFilteredSpectrogram"""
    def __init__(self, sample_rate=44100, filter_length=8192, hop_length=8820, win_length=None, num_bands=24, fmin=65, fmax=2100, unique_filters=True):
        super(LMLFSpectrogram, self).__init__()
        self.stft = STFT(filter_length, hop_length, win_length)
        # filterbank from madmom
        fname = 'lmlf.wav'
        sf.write(fname, np.random.uniform(-1, 1, 100000), sample_rate)
        _sig = SignalProcessor(num_channels=1, sample_rate=sample_rate)
        _frames = FramedSignalProcessor(frame_size=filter_length, fps=sample_rate / hop_length)
        _stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        _spec = LogarithmicFilteredSpectrogramProcessor(num_bands=num_bands, fmin=fmin, fmax=fmax, unique_filters=unique_filters)
        _spec(_stft(_frames(_sig(fname))))
        os.remove(fname)
        self.filterbank = torch.FloatTensor(np.asarray(_spec.filterbank))

    def forward(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: torch.FloatTensor with shape (B, T) in range [-1, 1]
        RETURNS
        -------
        lmlf_output: torch.FloatTensor of shape (B, T, n_bins)
        """
        
        self.stft.to(y.device)
        self.filterbank = self.filterbank.to(y.device)

        magnitudes, phases = self.stft(y)
        magnitudes = magnitudes.data
        lmlf_output = torch.matmul(magnitudes.permute(0, 2, 1)[:, :, :-1], self.filterbank)
        lmlf_output = torch.log10(1 + lmlf_output)
        return lmlf_output

        
if __name__ == '__main__':
    sr = 44100
    filter_length = 8192
    hop_length = 8820 
    real_wave, _ = librosa.load(librosa.util.example_audio_file(), sr=sr, mono=True)
    real_wave = torch.from_numpy(real_wave)

    # ! Log-magnitude log-frequency spectrogram
    num_bands = 24
    fmin = 65
    fmax = 2100

    # torch
    torch_lmlf = LMLFSpectrogram(sample_rate=sr, filter_length=filter_length, hop_length=hop_length, num_bands=num_bands, fmin=65, fmax=2100)
    lmlf = torch_lmlf(real_wave.unsqueeze(0))

    # madmom
    _sig = SignalProcessor(num_channels=1, sample_rate=sr)
    _frames = FramedSignalProcessor(frame_size=filter_length, fps=sr / hop_length)
    _stft = ShortTimeFourierTransformProcessor()  # caching FFT window
    _spec = LogarithmicFilteredSpectrogramProcessor(num_bands=num_bands, fmin=fmin, fmax=fmax)    
    sig = _sig(librosa.util.example_audio_file())
    frames = _frames(sig)
    stft = _stft(frames)
    spec = _spec(stft)

    diff = np.mean(np.abs(lmlf.squeeze(0).numpy() - spec))
    print('===== log-magnitude log-frequency spectrogram =====')
    print('mean difference between outputs from torch and madmom : ', diff)
    print('shape : ', lmlf.shape)

    # ! Mel-spectrogram
    n_mels = 128

    # torch
    torch_mel = MelSpectrogram(n_mels=n_mels, sample_rate=sr, filter_length=filter_length, hop_length=hop_length)
    t_mel = torch_mel(real_wave.unsqueeze(0))

    # librosa
    mel = librosa.feature.melspectrogram(real_wave.numpy(), n_mels=n_mels, sr=sr, n_fft=filter_length, hop_length=hop_length, htk=True, power=1.0)

    diff = np.mean(np.abs(t_mel.squeeze(0).numpy() - np.log(np.clip(mel, 1e-5, None))))
    print('===== mel-spectrogram =====')
    print('mean difference between outputs from torch and librosa: ', diff)
    print('shape : ', t_mel.shape)

    # ! STFT

    # torch
    torch_stft = STFT(filter_length=filter_length, hop_length=hop_length)
    t_stft, phases = torch_stft(real_wave.unsqueeze(0))

    # librosa
    lib_stft = librosa.stft(real_wave.numpy(), n_fft=filter_length, hop_length=hop_length)

    diff = np.mean(np.abs(t_stft.squeeze(0).numpy() - np.abs(lib_stft)))
    print('===== shot time fourier transform =====')
    print('mean difference between outputs from torch and librosa : ', diff)
    print('shape : ', t_stft.shape)
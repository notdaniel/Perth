import torch
import numpy as np
from librosa import resample

from .model.perth_net import PerthNet
from .. import PREPACKAGED_MODELS_DIR
from perth.watermarker import WatermarkerBase


def _to_tensor(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.copy())
    return x.to(dtype=torch.float, device=device)


class PerthImplicitWatermarker(WatermarkerBase):
    def __init__(self, run_name:str="implicit", models_dir=PREPACKAGED_MODELS_DIR,
                 device="cpu", perth_net=None):
        assert (run_name is None) or (perth_net is None)
        if perth_net is None:
            self.perth_net = PerthNet.load(run_name, models_dir).to(device)
        else:
            self.perth_net = perth_net.to(device)

    def apply_watermark(self, signal,  sample_rate, **_):
        # Handle stereo/multi-channel audio
        if signal.ndim == 2:
            # Process each channel separately
            watermarked_channels = []
            for channel_idx in range(signal.shape[0]):
                channel_signal = signal[channel_idx]
                watermarked_channel = self._apply_watermark_mono(channel_signal, sample_rate)
                watermarked_channels.append(watermarked_channel)
            return np.stack(watermarked_channels, axis=0)
        else:
            # Mono audio
            return self._apply_watermark_mono(signal, sample_rate)
    
    def _apply_watermark_mono(self, signal, sample_rate):
        change_rate = sample_rate != self.perth_net.hp.sample_rate
        original_signal = signal.copy()  # Keep original for analysis
        signal = resample(signal, orig_sr=sample_rate, target_sr=self.perth_net.hp.sample_rate) if change_rate \
            else signal

        # split signal into magnitude and phase
        signal = _to_tensor(signal, self.perth_net.device)
        magspec, phase = self.perth_net.ap.signal_to_magphase(signal)

        # encode the watermark
        magspec = magspec[None].to(self.perth_net.device)
        wm_magspec, _mask = self.perth_net.encoder(magspec)
        wm_magspec = wm_magspec[0]

        # assemble back into watermarked signal
        wm_signal = self.perth_net.ap.magphase_to_signal(wm_magspec, phase)
        wm_signal = wm_signal.detach().cpu().numpy()
        wm_signal = resample(wm_signal, orig_sr=self.perth_net.hp.sample_rate, target_sr=sample_rate) if change_rate \
            else wm_signal
        
        # Somewhat smart gain compensation
        original_peak = np.max(np.abs(original_signal))
        watermarked_peak = np.max(np.abs(wm_signal))
        
        if watermarked_peak > 0:
            headroom = 1.0 - original_peak
            if original_peak > 0.95:
                scale_factor = original_peak / watermarked_peak
            elif watermarked_peak > original_peak:
                max_allowed_peak = min(1.0, original_peak + headroom * 0.5)
                scale_factor = max_allowed_peak / watermarked_peak
            else:
                scale_factor = 1.0
            
            wm_signal = wm_signal * scale_factor
        
        return wm_signal

    def get_watermark(self, wm_signal, sample_rate, round=True, **_):
        # Handle stereo/multi-channel audio
        if wm_signal.ndim == 2:
            # Extract watermark from each channel and average the results
            watermarks = []
            for channel_idx in range(wm_signal.shape[0]):
                channel_signal = wm_signal[channel_idx]
                channel_watermark = self._get_watermark_mono(channel_signal, sample_rate, round)
                watermarks.append(channel_watermark)
            return np.mean(watermarks, axis=0)
        else:
            # Mono audio
            return self._get_watermark_mono(wm_signal, sample_rate, round)
    
    def _get_watermark_mono(self, wm_signal, sample_rate, round=True):
        change_rate = sample_rate != self.perth_net.hp.sample_rate
        if change_rate:
            wm_signal = resample(wm_signal, orig_sr=sample_rate, target_sr=self.perth_net.hp.sample_rate,
                                 res_type="polyphase")
        wm_signal = _to_tensor(wm_signal, self.perth_net.device)
        wm_magspec, _phase = self.perth_net.ap.signal_to_magphase(wm_signal)
        wm_magspec = wm_magspec.to(self.perth_net.device)
        wmark_pred = self.perth_net.decoder(wm_magspec[None])[0]
        wmark_pred = wmark_pred.clip(0., 1.)
        wmark_pred = wmark_pred.round() if round else wmark_pred
        return wmark_pred.detach().cpu().numpy()

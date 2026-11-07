"""Cache identity covers frozen preprocessing only; never cache projector outputs."""
from .task import TaskError, digest

def feature_cache_key(audio_sha256, preprocessing_version, encoder_sha256, pooling, dtype):
    if not all((audio_sha256, preprocessing_version, encoder_sha256, pooling, dtype)):
        raise TaskError('complete frozen feature identity required')
    return digest({'audio': audio_sha256, 'preprocessing': preprocessing_version, 'encoder': encoder_sha256,
                   'pooling': pooling, 'dtype': dtype, 'stage': 'before_trainable_projector'})

def add_noise(samples, snr_db, seed):
    """White Gaussian noise scaled to measured waveform RMS. No clipping or renormalization."""
    import numpy as np
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all(): raise TaskError('invalid waveform')
    power = np.mean(x*x)
    if power <= 0: raise TaskError('cannot define SNR for silence')
    noise = np.random.default_rng(seed).standard_normal(x.shape)
    noise *= np.sqrt(power / (10 ** (snr_db / 10) * np.mean(noise*noise)))
    return x + noise

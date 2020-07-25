""" TODO """

from numpy import sin, cos, exp, pi
import pandas as pd
import scipy.stats as stats


def normal(x, mu=0, std=1):
    """ TODO """
    return stats.norm.pdf(x, mu, std)


def sine_wave(t, freq=1.0, amp=1.0):
    """ TODO """
    return amp * sin(t * freq * 2 * pi)


def cosine_wave(t, freq=1.0, amp=1.0):
    """ TODO """
    return amp * cos(t * freq * 2 * pi)


def damped_sine_wave(t, freq=1.0, amp=1.0, damp=1.0):
    """ TODO """
    return exp(-t * damp) * amp * cos(t * freq * 2 * pi)


def dataframe(**kw):
    """ TODO """
    return pd.DataFrame(kw)

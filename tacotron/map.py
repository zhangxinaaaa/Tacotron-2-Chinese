import os
import fnmatch
from scipy.special import erfinv, erf
import numpy as np
import random


class Map(object):
    """
    Base map class.
    """

    def __init__(self):
        """
        Argument/s:
        """
        self.mu = []
        self.sigma = []

    def stats(self, x):
        """
        The base stats() function is used when no statistics are requied for
        the map function.
        Argument/s:
            x - a set of samples.
        """
        pass


class NormalCDF(Map):
    """
    Normal cumulative distribution function (CDF) map.
    """

    def forward(self, x):
        """
        Normal (Gaussian) cumulative distribution function (CDF).
        Argument/s:
            x - random variable realisations.
        Returns:
            CDF
        """
        bar = []
        for i in range(len(x)):
            v_1 = x[i] - self.mu[i]
            v_2 = self.sigma[i] * np.sqrt(2.0)
            v_3 = erf(v_1 / v_2)
            x_bar = (v_3 + 1.0) * 0.5
            bar.append(x_bar)
        return bar

    def backward(self, bar):
        """
        Inverse of normal (Gaussian) cumulative distribution function (CDF).
        Argument/s:
            y - cumulative distribution function value.
        Returns:
            Inverse of CDF.
        """
        x = []
        for i in range(len(bar)):
            v_1 = self.sigma[i] * np.sqrt(2.0)
            v_2 = 2.0 * bar[i]
            v_3 = erfinv(v_2 - 1.0)
            v_4 = v_1 * v_3
            x.append(v_4 + self.mu[i])
        return x

    def map(self, y):
        pitch, corr = y[:, -2], y[:, -1]
        return self.forward([pitch, corr])

    def inverse(self, y):
        pitch_bar, corr_bar = y[:, 0], y[:, 1]
        return self.backward([pitch_bar, corr_bar])

    def stats(self, x):
        """
        Compute stats for each frequency bin.
        Argument/s:
            x - sample.
        """
        pitch = np.array([])
        corr = np.array([])
        for sample in x:
            s = np.fromfile(sample, dtype='float32')
            s = np.resize(s, (-1, 20))  # pitch
            pitch = np.concatenate((pitch, s[:, -2]))
            corr = np.concatenate((corr, s[:, -1]))
        self.mu.append(np.mean(pitch))
        self.sigma.append(np.std(pitch))
        self.mu.append(np.mean(corr))
        self.sigma.append(np.std(corr))


def find_files(dictionary, pattern, name_style='root'):
    """
        Find files with specified patterns in given dictionary and its sub dictionaries
    Arguments:
        dictionary {[string]} -- [dictionary to find files in]

    Keyword Arguments:
        pattern {list} -- [file extensions] (default: {['*.wav', '*.WAV']})

    Returns:
        [list] -- [contains files' absolute path]
    """
    files = []
    for root, dirnames, filenames in os.walk(dictionary):
        if name_style == 'root':
            for filename in fnmatch.filter(filenames, pattern):
                files.append(os.path.join(root, filename))
        elif name_style == 'local':
            for filename in fnmatch.filter(filenames, pattern):
                files.append(filename)
        else:
            raise ValueError('No this type')
    return files





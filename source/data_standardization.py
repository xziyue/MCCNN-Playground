import numpy as np
from scipy.stats import describe

class DataStandardizer:

    def fit(self, xs):
        self.channelStats = [describe(np.take(xs, ind, axis=4).flat) for ind in range(xs.shape[4])]

    def transform(self, _xs):
        xs = _xs.copy()
        for ind in range(xs.shape[4]):
            theIndex = tuple([slice(None)] * 4 + [ind])
            # special treatment for the e channel
            if ind == 2:
                stat = self.channelStats[ind]
                xs[theIndex] -= stat.minmax[0]
                xs[theIndex] /= (stat.minmax[1] - stat.minmax[0])
            else:
                xs[theIndex] = self._get_standardized_channel(xs[theIndex], ind)
        return xs

    def _get_standardized_channel(self, _channel, ind):
        channel = _channel.copy()
        channel -= self.channelStats[ind].minmax[0]
        channel += 1.0
        channel = np.log(channel)
        channel /= channel.max()
        return channel


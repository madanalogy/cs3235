import os
import datetime
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

raw = ['time', 'size', 'direction']
features = ['in_ratio', 'out_ratio', 'avg_sz', 'avg_sz_out', 'avg_sz_in', 'freq', 'freq_in', 'freq_out']
cols = features.copy()
cols.append('label')


def delta(d):
    return d.microseconds + d.seconds * pow(10, 6)


def divide(top, bot):
    return top / bot if bot else 0


def load(directory='traces/profile', n=35, k=8):
    data = []
    for i in range(k):
        for j in range(n):
            with open(directory + str(i + 1) + '/' + str(j + 1)) as fp:
                for line in fp:
                    time, size, dir = line.split()
                    hrs, mins, secs = time.split(':')
                    secs, micro = secs.split('.')
                    size = int(size)
                    d = datetime.timedelta(
                        hours=int(hrs), minutes=int(mins), seconds=int(secs), microseconds=int(micro)
                    )
                    data.append([d, size, dir])
    return pd.DataFrame(np.array(data), columns=raw)


def extract(directory='traces/profile', n=35, k=8):
    data = []
    for i in range(k):
        for j in range(n):
            num_out = 0
            num_in = 0
            sz_total = 0
            sz_in = 0
            sz_out = 0
            d_in = 0
            d_out = 0
            d_total = 0
            with open(directory + str(i + 1) + '/' + str(j + 1)) as fp:
                lines = list(fp.readlines())
                length = len(lines)
                last_in = None
                last_out = None
                deltas = []
                for idx, line in enumerate(lines):
                    time, size, dir = line.split()
                    hrs, mins, secs = time.split(':')
                    secs, micro = secs.split('.')
                    size = int(size)
                    sz_total += size
                    deltas.append(datetime.timedelta(
                        hours=int(hrs), minutes=int(mins), seconds=int(secs), microseconds=int(micro)))
                    if dir == 'in':
                        num_in += 1
                        sz_in += size
                        if last_in is not None:
                            d_in += delta(deltas[idx] - last_in)
                        last_in = deltas[idx]
                    elif dir == 'out':
                        num_out += 1
                        sz_out += size
                        if last_out is not None:
                            d_out += delta(deltas[idx] - last_out)
                        last_out = deltas[idx]
                if length != 0:
                    d_total = delta(deltas[length - 1])
            data.append([
                divide(num_in, length),
                divide(num_out, length),
                divide(sz_total, length),
                divide(sz_in, num_in),
                divide(sz_out, num_out),
                divide(d_total, length),
                divide(d_in, num_in),
                divide(d_out, num_out),
                j + 1
            ])
    return pd.DataFrame(np.array(data), columns=cols)


def evaluate(split=0.1, metric='distance', k=5):
    data = extract()
    X = preprocessing.normalize(data[features])
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)

    neigh = KNeighborsClassifier(n_neighbors=k, weights=metric)
    neigh.fit(X_train, y_train)
    return neigh.score(X_test, y_test)


if __name__ == '__main__':
    pass

import sys
import datetime
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

features = ['left_in', 'mid_in', 'right_in', 'left_out', 'mid_out', 'right_out']
label = 'label'
cols = features.copy()
cols.append(label)


def delta(d):
    return d.microseconds + d.seconds * pow(10, 6)


def divide(top, bot):
    return top / bot if bot else 0


def load(prefix='traces/profile', n=35, k=8):
    data = []
    for i in range(k):
        for j in range(n):
            with open(prefix + str(i + 1) + '/' + str(j + 1)) as fp:
                in_data, out_data = cycle(fp)
            data.append(extract(in_data, out_data, j + 1))
    return pd.DataFrame(np.array(data), columns=cols)


def cycle(fp):
    in_data = []
    out_data = []
    for line in fp:
        time, size, dir = line.split()
        hrs, mins, secs = time.split(':')
        secs, micro = secs.split('.')
        size = int(size)
        if size != 0:
            d = datetime.timedelta(
                hours=int(hrs), minutes=int(mins), seconds=int(secs), microseconds=int(micro)
            )
            if dir == 'in':
                in_data.append((delta(d), size))
            else:
                out_data.append((delta(d), size))
    return in_data, out_data


def extract(in_data, out_data, target=0):
    left_in, mid_in, right_in, in_total = gradients(in_data)
    left_out, mid_out, right_out, out_total = gradients(out_data)
    return [left_in, mid_in, right_in, left_out, mid_out, right_out, target]


def gradients(data):
    left = 0
    mid = 0
    right = 0
    total_s = 0
    total_d = 0
    total = 0
    for i, (d, s) in enumerate(data):
        total += s
        total_s += s
        total_d += d
        if i == int((len(data) - 1) / 3):
            left = divide(total_s, total_d)
            total_s = 0
            total_d = 0
        elif i == int((len(data) - 1) * 2 / 3):
            mid = divide(total_s, total_d)
            total_s = 0
            total_d = 0
        elif i == len(data) - 1:
            right = divide(total_s, total_d)
            total_s = 0
            total_d = 0
    return left, mid, right, total


def evaluate(data, split=35, metric='distance', k=7):
    X = preprocessing.normalize(data[features])
    y = data[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, shuffle=False)

    neigh = KNeighborsClassifier(n_neighbors=k, weights=metric)
    neigh.fit(X_train, y_train)
    return neigh.score(X_test, y_test)


def prepare(data, metric='distance', k=8):
    X = preprocessing.normalize(data[features])
    y = data[label]

    neigh = KNeighborsClassifier(n_neighbors=k, weights=metric)
    neigh.fit(X, y)
    return neigh


def observe(obs, prefix='../', suffix='-anon'):
    data = []
    for i in range(35):
        with open(prefix + obs + '/' + str(i + 1) + suffix) as fp:
            in_data, out_data = cycle(fp)
        data.append(extract(in_data, out_data, i + 1))
    return pd.DataFrame(np.array(data), columns=cols)


def execute(obs1, obs2):
    model = prepare(load())
    test1 = observe(obs1)
    test2 = observe(obs2)
    result1 = model.predict(preprocessing.normalize(test1[features]))
    result2 = model.predict(preprocessing.normalize(test2[features]))
    with open('../result.txt', 'w') as fp:
        for i in range(35):
            fp.write(result1[i] + ' ' + result2[i] + '\n')


if __name__ == '__main__':
    execute(sys.argv[1], sys.argv[2])

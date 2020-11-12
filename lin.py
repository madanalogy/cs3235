import datetime
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def delta(d):
    return d.microseconds + d.seconds * pow(10, 6)


def divide(top, bot):
    return top / bot if bot else 0


def load(directory='traces/profile', n=35, k=8):
    data = []
    for i in range(k):
        for j in range(n):
            in_data = []
            out_data = []
            cum = 0
            with open(directory + str(i + 1) + '/' + str(j + 1)) as fp:
                for line in fp:
                    time, size, dir = line.split()
                    hrs, mins, secs = time.split(':')
                    secs, micro = secs.split('.')
                    size = int(size)
                    if size != 0:
                        cum += size
                        d = datetime.timedelta(
                            hours=int(hrs), minutes=int(mins), seconds=int(secs), microseconds=int(micro)
                        )
                        in_data.append([delta(d), cum])
            data.append((
                pd.DataFrame(np.array(in_data), columns=['time', 'size']) if len(in_data) > 0 else None,
                pd.DataFrame(np.array(out_data), columns=['time', 'size']) if len(out_data) > 0 else None,
                j + 1
            ))
    return data


def extract(sample, target):
    data = []
    for in_data, out_data, label in sample:
        model = linear_model.LinearRegression()
        if in_data is not None and label == target:
            x = np.array(in_data['time']).reshape(-1, 1)
            model.fit(x, in_data['size'])
            print(model.coef_[0], "in")
        if out_data is not None and label == target:
            x = np.array(out_data['time']).reshape(-1, 1)
            model.fit(x, out_data['size'])
            print(model.coef_[0], "out")


def evaluate(data, split=35, metric='distance', k=7):
    X = preprocessing.normalize(data[features])
    y = data[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, shuffle=False)

    neigh = KNeighborsClassifier(n_neighbors=k, weights=metric)
    neigh.fit(X_train, y_train)
    return neigh.score(X_test, y_test)


if __name__ == '__main__':
    print(evaluate(load()))

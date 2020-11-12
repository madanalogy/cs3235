import datetime
import pandas as pd


def delta(d):
    return d.microseconds + d.seconds * pow(10, 6)


def divide(top, bot):
    return top / bot if bot else 0


def load(prefix='traces/profile', n=35, k=8):
    data = []
    for i in range(k):
        for j in range(n):
            sequence = []
            with open(prefix + str(i + 1) + '/' + str(j + 1)) as fp:
                for line in fp:
                    time, size, direction = line.split()
                    hrs, minutes, secs = time.split(':')
                    secs, micro = secs.split('.')
                    size = int(size)
                    d = datetime.timedelta(
                        hours=int(hrs), minutes=int(minutes), seconds=int(secs), microseconds=int(micro)
                    )
                    sequence.append([delta(d), size, 0 if direction == 'in' else 1])
            data.append([pd.DataFrame(sequence, columns=['time', 'size', 'dir']), j + 1])
    return pd.DataFrame(data, columns=['sequence', 'url'])

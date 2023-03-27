import csv
import numpy as np
from scipy.cluster import hierarchy
from matplotlib import pyplot as plt

# Author: Cinthya Nguyen


def load_data(filepath):
    with open(filepath, 'r') as file:
        d = list(csv.DictReader(file))

    return d


def calc_features(row):
    features_row = np.zeros(6, dtype='int64')
    features_row[0] = row['Attack']
    features_row[1] = row['Sp. Atk']
    features_row[2] = row['Speed']
    features_row[3] = row['Defense']
    features_row[4] = row['Sp. Def']
    features_row[5] = row['HP']

    return features_row


def hac(features):
    n = len(features)
    features = np.array(features)
    A = np.zeros((n, 1))  # 800 x 1
    B = np.ones((n, 1))  # 800 x 1

    for i in range(len(features)):
        A[i] = np.matmul(features[i].T, features[i])

    # Compute distance matrix
    D = np.sqrt(np.matmul(A, B.T) + np.matmul(B, A.T) - (2 * (np.matmul(features, features.T))))

    Z = np.empty([n - 1, 4])  # Where clusters will be stored

    numPKM = dict()  # Create a dictionary to keep track of the # of Pokemon in each cluster
    for i in range(n):
        numPKM[i] = 1

    visited = list()  # Keep track of visited clusters

    D = np.triu(D)
    D[D == 0] = np.NaN
    print(D)

    # Create list of clusters
    for k in range(n - 1):
        minimum = float('inf')  # Minimum value
        ci, cj = 0, 0  # Indices of minimum value
        lenD = len(D)

        for i in range(lenD):
            if visited.count(i) > 0:
                continue

            tempMin = np.nanmin(D[i])

            tempIndex = np.where(D[i] == tempMin)[0]

            if (visited.count(tempIndex)) > 0 or (D[i][tempIndex] is np.NaN) or (i == tempIndex):
                continue

            if tempMin < minimum:
                ci = i
                cj = tempIndex
                minimum = tempMin
            elif tempMin == min and tempIndex < cj:
                cj = tempMin

        newRowCol = list()
        D[ci][cj] = np.NaN
        visited.append(ci)
        visited.append(cj)

        for co in range(lenD):
            if (visited.count(co) > 0) or (D[co][ci] is np.NaN) or (D[co][cj] is np.NaN):
                newRowCol.append(np.NaN)
            else:
                newRowCol.append(max(D[co][ci], D[co][cj]))

        print(len(newRowCol))

        D = np.row_stack([D, (newRowCol, object)])
        newRowCol.append(0)
        D = np.column_stack([D, (newRowCol, object)])

        numPKM[n + k] = numPKM[ci] + numPKM[cj]

        zInfo = [ci, cj, minimum, numPKM[n + k]]
        Z[k] = zInfo

    return Z


def imshow_hac(Z, names):
    plt.figure()
    plt.title('N = ' + str(len(names)))  # Create figure
    hierarchy.dendrogram(Z, labels=names, leaf_font_size=8, leaf_rotation=90.0)  # Plot dendrogram
    plt.tight_layout()
    plt.show()  # Show plot

    pass


if __name__ == '__main__':
    feat_and_names = [(calc_features(row), row['Name']) for row in load_data('Pokemon.csv')[:50]]
    Z = hac([row[0] for row in feat_and_names])
    names = [row[1] for row in feat_and_names]
    imshow_hac(Z, names)

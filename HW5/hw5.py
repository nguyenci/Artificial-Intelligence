import sys
import matplotlib.pyplot as plt
import numpy as np

# Author: Cinthya Nguyen
# Class: CS540 SP23


def lin_regress(d):
    n = len(d)

    X = np.ones((n, 2), dtype=np.int64)
    Y = np.empty(n, dtype=np.int64)

    for i in range(n):
        X[i][1] = d[i][0]
        Y[i] = d[i][1]

    print('Q3a:')
    print(X)

    print('Q3b:')
    print(Y)

    Z = np.matmul(X.T, X)
    print('Q3c:')
    print(Z)

    inverse = np.linalg.inv(Z)
    print('Q3d:')
    print(inverse)

    PI = np.matmul(inverse, X.T)
    print('Q3e:')
    print(PI)

    hat_beta = np.matmul(PI, Y)  # coefficient of linear regression equation
    print('Q3f:')
    print(hat_beta)

    beta_0 = hat_beta[0]
    beta_1 = hat_beta[1]

    y_test = beta_0 + (beta_1 * 2022)
    print('Q4: ' + str(y_test))  # TODO: output different from specification

    if beta_1 < 0:
        print('Q5a: <')
        print('Q5b: Because hat beta_1 is negative, as the year increases the number of ice days '
              'decreases for Lake Mendota.')
    elif beta_1 > 0:
        print('Q5a: >')
        print('Q5b: Because hat beta_1 is positive, as the year increases, the number of ice days'
              'increases on Lake Mendota.')
    else:
        print('Q5a: =')
        print('Q5b: Because hat beta_1 is 0, this means that the year as no effect on the number '
              'of ice days on Lake Mendota.')

    prediction = (-beta_0) / beta_1
    print('Q6a: ' + str(prediction))

    print('Q6b: This is a compelling prediction because looking at the graph generated from the '
          'csv file, one can see that the number of ice days is slowly decreasing each year. '
          'Looking at the csv, you can also see the number of ice days decrease from 3 digits to 2 '
          'digits from 1850 to now. This means that in 400 years it could be possible to have 0 '
          'ice days given the current trends. Additionally, hat beta_1 is negative, which holds.')


def visualize(d):
    T = d.T
    x = T[0]
    y = T[1]

    for i in range(len(d)):
        x[i] = str(x[i])

    plt.plot(x, y)
    plt.xlabel('Year')
    plt.ylabel('Number of Frozen Days')
    plt.legend()
    plt.savefig("plot.jpg")


def load_data(filename):
    result = np.loadtxt(open(filename, 'r'), delimiter=",", skiprows=1)

    return result


if __name__ == '__main__':
    data = load_data(sys.argv[1])
    visualize(data)
    lin_regress(data)

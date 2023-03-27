from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

"""
Author: Cinthya Nguyen
Class: CS540, SP23
Notes: I discussed questions and concepts about the project and PCA with Patrick Wilcox.
"""


def load_and_center_dataset(filename):
    """
    Load dataset and center around origin.
    :param filename: dataset to load
    :return: array of floats
    """
    dataset = np.load(filename)
    dataset = dataset - np.mean(dataset, axis=0)

    return dataset


def get_covariance(dataset):
    """
    Calculate covariance matrix.
    :param dataset: use data set to form matrix
    :return: covariance matrix
    """
    d = np.array(dataset)
    transpose1 = np.transpose(d)
    transpose2 = np.dot(transpose1, d)

    print(transpose2)

    cov = [(i / (len(dataset) - 1)) for i in transpose2]

    return cov


def get_eig(S, m):
    """
    Get the m largest eigenvalues and corresponding eigenvectors.
    :param S: covariance matrix
    :param m: how many eigenvalues to get
    :return: eigenvalues and corresponding eigenvectors
    """
    evalues, evectors = eigh(S, subset_by_index=[len(S) - m, len(S) - 1])

    ex = evalues.argsort()[::-1]  # sort in decreasing order
    evalues = evalues[ex]
    evectors = evectors[:, ex]  # sort corresponding eigenvectors
    evalues = np.diag(evalues)

    return evalues, evectors


def get_eig_prop(S, prop):
    """
    Get eigenvalues whose variance is equal to or greater than the proportion.
    :param S: covariance matrix
    :param prop: compare eigenvalue's variance with
    :return: eigenvalues and corresponding eigenvectors
    """
    evalues, evectors = eigh(S)
    evectors = np.transpose(evectors)

    total = sum(evalues)  # sum of eigenvalues
    variance = [(i / total) for i in evalues]

    thislist = []
    otherlist = []
    for i in range(len(variance)):  # find eigenvalues whose variance is greater than/equal to prop
        if variance[i] >= prop:
            thislist.append(evalues[i])
            otherlist.append(evectors[i])

    evalues = np.array(thislist)
    evectors = np.transpose(np.array(otherlist))

    ex = evalues.argsort()[::-1]  # sort in decreasing order
    evalues = np.diag(evalues[ex])
    evectors = evectors[:, ex]  # sort corresponding eigenvectors

    return evalues, evectors


def project_image(image, U):
    """
    Project image using PCA.
    :param image: which image to use
    :param U: array of eigenvectors
    :return: PCA projection of the image
    """
    m = len(U[0])
    uT = np.transpose(U)
    alpha = np.dot(uT, image)

    pca = np.dot(alpha[0], uT[0])

    for i in range(1, m):  # find pca of image
        pca += np.dot(alpha[i], uT[i])

    return pca


def display_image(orig, proj):
    """
    Display the original image and projection.
    :param orig: original image
    :param proj: projected image
    :return: two subplots and colorbars
    """

    origImg = np.transpose(orig.reshape(32, 32))
    progImg = np.transpose(proj.reshape(32, 32))

    ax1 = plt.subplot(121)  # original subplot
    ax1.set_title('Original')
    color1 = plt.imshow(origImg, aspect='equal')
    plt.colorbar(color1)

    ax2 = plt.subplot(122)  # projection subplot
    ax2.set_title('Projection')
    color2 = plt.imshow(progImg, aspect='equal')
    plt.colorbar(color2)

    plt.show()

    pass

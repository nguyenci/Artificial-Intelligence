import sys
import math

"""
File: hw2.py
Author: Cinthya Nguyen
Class: CS540, SP23
"""


def get_parameter_vectors():
    """
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    described in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    """
    # Implementing vectors e,s as lists (arrays) of length 26
    # with p[0] being the probability of 'A' and so on
    e = [0] * 26
    s = [0] * 26

    with open('../../Downloads/assignment/e.txt',
              encoding='utf-8') as f:  # '../../Downloads/assignment/e.txt'
        for line in f:
            # strip: removes the newline character
            # split: split the string on space character
            char, prob = line.strip().split(" ")
            # ord('E') gives the ASCII (integer) value of character 'E'
            # we then subtract it from 'A' to give array index
            # This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char) - ord('A')] = float(prob)
    f.close()

    with open('../../Downloads/assignment/s.txt',
              encoding='utf-8') as f:  # '../../Downloads/assignment/s.txt'
        for line in f:
            char, prob = line.strip().split(" ")
            s[ord(char) - ord('A')] = float(prob)
    f.close()

    return (e, s)


def shred(filename):
    # Using a dictionary here. You may change this to any data structure of
    # your choice such as lists (X=[]) etc. for the assignment
    X = dict()
    char = 65
    for i in range(26):
        X[chr(char)] = 0
        char += 1

    with open(filename, encoding='utf-8') as f:
        while True:
            c = f.read(1).upper()
            if not c:
                break
            for key in X:
                if c == key:
                    X[key] += 1
                    break
    f.close()

    print('Q1')
    for letter, num in X.items():
        print(letter, num)

    return X


def calc(filename):
    # Q2
    e, s = get_parameter_vectors()
    d = shred(filename)
    print('Q2\n' + '{:.4f}'.format(round(int(d['A']) * math.log(e[0]), 4)) + '\n' + '{:.4f}'.format(
        round(int(d['A']) * math.log(s[0]), 4)))

    # Q3
    char = 65
    sum_e = 0
    sum_s = 0

    for i in range(26):
        sum_e += int(d[chr(char)]) * math.log(e[i])
        sum_s += int(d[chr(char)]) * math.log(s[i])
        char += 1

    f_english = round(math.log(0.6) + sum_e, 4)
    f_spanish = round(math.log(0.4) + sum_s, 4)
    print('Q3\n' + '{:.4f}'.format(f_english) + '\n' + '{:.4f}'.format(f_spanish))

    # Q4
    p_english = 1 / (1 + math.e ** (f_spanish - f_english))
    print('Q4\n' + '{:.4f}'.format(round(p_english, 4)))


if __name__ == '__main__':
    calc('letter.txt')

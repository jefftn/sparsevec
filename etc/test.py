import math
import csv
import numpy as np
import random as rand
import spectral
import matplotlib.pyplot as plt

print 'Test'


def test():
  # example data
  x = np.arange(0.1, 4, 0.5)
  y = np.exp(-x)
  # example error bar values that vary with x-position
  error = 0.1 + 0.2 * x
  # error bar values w/ different -/+ errors
  lower_error = 0.4 * error
  upper_error = error
  asymmetric_error = [lower_error, upper_error]

  # fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
  plt.errorbar(x, y, yerr=error, fmt='o')
  # plt.set_title('variable, symmetric error')

  # ax1.errorbar(x, y, xerr=asymmetric_error, fmt='o')
  # ax1.set_title('variable, asymmetric error')
  # ax1.set_yscale('log')
  plt.show()
  x = input()


def test2():
  plt.axis([0, 10, 0, 10])

  # example data
  x = np.array([1, 2, 3])
  y = np.array([4, 5, 6])

  lower_error = np.array([1, 1, 1])
  upper_error = np.array([1, 1, 2])

  print lower_error
  asymmetric_error = [lower_error, upper_error]

  plt.errorbar(x, y, yerr=asymmetric_error, fmt='o')

  plt.show()
  x = input()


test2()












































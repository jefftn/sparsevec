import math
import csv
import numpy as np
import random as rand
import spectral
import matplotlib.pyplot as plt

print 'Sparse Vector Recovery'


def test_file():
  Y = np.loadtxt(open("Y.csv","rb"),delimiter=",",skiprows=0)
  q = np.loadtxt(open("q.csv","rb"),delimiter=",",skiprows=0)
  sparse_rec(Y, q, 2, 10)

def S_lam(v, lam):
  w = []
  # print 'printing vi'
  for vi in v:
    # print vi
    wi = np.sign(vi) * max(abs(vi) - lam, 0)
    w.append(wi)
  return np.array(w)

# Y: p x n, q: n x 1
def sparse_rec(Y, q, lam, num_it):
  # print Y
  # print Y.shape
  # print Y.T
  # print q
  # print q.shape
  # print q.reshape(-1,)
  # print q.tolist()
  # print type(np.array([1,2,3]))
  # print Y.dot(q)
  # print 'fcn called ', q
  x = None
  YT = Y.T
  if num_it > 0:
    # print 'if case'
    for k in range(0, num_it):
      # print 'k: ', k
      Y_q_prod = Y.dot(q)
      old_x = x
      x = S_lam(Y.dot(q), lam)
      if np.linalg.norm(x, 2) == 0:
        # print 'x is 0!'
        # print 'k: ', k
        # print 'Yq: ', Y_q_prod
        # print 'lam: ', lam
        # print 'old x: ', old_x
        return Y_q_prod
      y_x_prod = YT.dot(x)
      y_x_prod_norm = np.linalg.norm(y_x_prod, 2)
      q = y_x_prod / y_x_prod_norm
      # if math.isnan(y_x_prod_norm):
      #   print '\nk: ', k
      #   print 'q: ', q
      #   print 'YT: ', YT
      #   print 'Yq: ', Y_q_prod
      #   print 'lam: ', lam
      #   print 'old x: ', old_x
      #   print 'x: ', x
      #   print 'Yx ', y_x_prod
      #   print '|Yx| ', y_x_prod_norm
      # print 'iteration: ', k
      # print 'x: ', x
      # print ' q: ', q
  else:
    # print 'else case'

    while True:
      Y_q_prod = Y.dot(q)
      old_x = x
      x = S_lam(Y.dot(q), lam)
      if np.linalg.norm(x, 2) == 0:
        # print 'x is 0!'
        # print 'k: ', k
        # print 'Yq: ', Y_q_prod
        # print 'lam: ', lam
        # print 'old x: ', old_x
        return Y_q_prod
      YT = Y.T
      y_x_prod = YT.dot(x)
      y_x_prod_norm = np.linalg.norm(y_x_prod, 2)
      q = y_x_prod / y_x_prod_norm
      if np.linalg.norm(Y_q_prod - Y.dot(q), 2) < 1e-7:
        break

  return Y.dot(q)

def deriv_p(n):
  return int(5 * n * math.log(n))

def deriv_lam(p):
  return 1 / math.sqrt(p)

def gen_k_sparse(k, p):
  # print 'gen sparse, ', k, p
  if k >= p:
    return [1] * p
  indicies = range(0, p)
  rand.shuffle(indicies)
  x = [0] * p
  for i in range(0, k):
    # if indicies[i] >= len(x):
      # print 'out of range: ', indicies[i], len(x)
    x[indicies[i]] = 1
  return x

# rows of return matrix are basis vectors
def gen_subspace(p, n, k):
  stdev = 1/float(p)
  G = np.random.normal(0, scale = stdev, size = (n - 1, p))
  x0 = gen_k_sparse(k, p)
  S = np.vstack([G, x0])
  return S, x0

def GS(S):
  Y = spectral.orthogonalize(S)
  return Y

def error(x0, Yq):
  x0_normalized = x0 / np.linalg.norm(x0, 2)
  return np.linalg.norm(x0_normalized - Yq, 2)

def gen_instance(n, k):
  p = deriv_p(n)
  lam = deriv_lam(p)
  S, x0 = gen_subspace(p, n, k) # n x p
  # print 'S, x0: ', S, x0
  Y = GS(S) # n x p
  Y = Y.T # p x n
  return Y, x0, lam


def run_random(n, k, num_it_alg, num_it_test):
  Y, x0, lam = gen_instance(n, k)
  best = float("inf")

  for j in range(0, num_it_test):
    i = rand.randint(0, len(Y) - 1)
    # print 'i: ', i
    q = Y[i]
    Yq = sparse_rec(Y, q, lam, num_it_alg)
    err = error(x0, Yq)
    if err < best:
      # print 'x0: ', x0
      # print 'Yq: ', Yq
      best = err
  return best


def test_random(n_range, k_range, num_it_alg, ep, num_it_test, num_pts):
  succ = ([], [])
  fail = ([], [])
  errs = []
  for t in range(0, num_pts):
    print 'point number: ', t
    n = rand.randint(n_range[0], n_range[-1])
    p = int(deriv_p(n))
    k = rand.randint(k_range[0], k_range[-1])
    err = run_random(n, k, num_it_alg, num_it_test)
    errs.append(err)
    if err <= ep:
      succ[0].append(p)
      succ[1].append(k)
    else:
      fail[0].append(p)
      fail[1].append(k)
  return succ, fail, errs


def plot(succ, fail):
  plt.axis([0, 2200, 0, 550])
  plt.xlabel('Ambient Dimension p')
  plt.ylabel('Sparsity k')
  plt.plot(succ[0], succ[1], 'bs')
  plt.plot(fail[0], fail[1], 'rs')
  plt.show()
  x = input()


def test_single(n, k, num_it_alg, num_it_test):
  p = int(deriv_p(n))
  err = run_random(n, k, num_it_alg, num_it_test)
  return err


succ, fail, errs = test_random(range(2,90), range(2,400), 0, 0.001, 20, 200)
print succ, fail, errs
plot(succ, fail)

# print test_single(70, 100, 0, 10)



def run_all(n, k, num_it):
  Y, x0, lam = gen_instance(n, k)
  best = float("inf")
  # for q in Y:
  print 'len Y: ', len(Y)
  for i in range(0, len(Y)):
    print 'i: ', i
    q = Y[i]
    Yq = sparse_rec(Y, q, lam, num_it)
    err = error(x0, Yq)
    if err < best:
      best = err
  return best

def test_all(n_range, k_range, num_it, ep):
  succ = ([], [])
  fail = ([], [])
  for n in n_range:
    print 'n: ', n
    p = int(deriv_p(n))
    for k in k_range:
      print 'k: ', k
      err = run(n, k, num_it)
      if err <= ep:
        succ[0].append(p)
        succ[1].append(k)
      else:
        fail[0].append(p)
        fail[1].append(k)
  return succ, fail


# >>> plt.axis([0, 6, 0, 20])
# [0, 6, 0, 20]
# >>> plt.show()
# >>> plt.plot([1,2,3,4], [1,4,9,16], 'bs')
# [<matplotlib.lines.Line2D object at 0x10777d5d0>]
# >>> plt.show()
# >>> plt.plot([1,2,3,4], [1,4,9,16], 'bs')
# [<matplotlib.lines.Line2D object at 0x104715a50>]
# >>> plt.plot([1,2,3,4], [5, 6 ,7 ,8], 'rs')
# [<matplotlib.lines.Line2D object at 0x104715d50>]
# >>> plt.show()


# succ, fail = test(range(2,90), range(2,600), 5000, 0.001)
# succ, fail = test(range(17,18), range(2,100), 5000, 0.001)
# succ, fail = test(range(2,10), range(2,10), 10, 0.001)
# print succ, fail

# test_file()


















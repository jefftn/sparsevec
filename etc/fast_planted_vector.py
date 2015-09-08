import math
import csv
import numpy as np
import random as rand
import spectral
import matplotlib.pyplot as plt

print 'Fast Planted Vector'

def test_file():
  Y = np.loadtxt(open("Y.csv","rb"),delimiter=",",skiprows=0)
  q = np.loadtxt(open("q.csv","rb"),delimiter=",",skiprows=0)
  YSY = get_YSY(Y)
  sparse_rec(Y, YSY, q, 0)

def test_mult():
  Y = np.loadtxt(open("Y.csv","rb"),delimiter=",",skiprows=0)
  s = np.array([1,2,3])
  print s * Y

# Y: p x n, q: n x 1
# def mult_diag(d, mtx, left=True):
#     """Multiply a full matrix by a diagonal matrix.
#     This function should always be faster than dot.

#     Input:
#       d -- 1D (N,) array (contains the diagonal elements)
#       mtx -- 2D (N,N) array

#     Output:
#       mult_diag(d, mts, left=True) == dot(diag(d), mtx)
#       mult_diag(d, mts, left=False) == dot(mtx, diag(d))
#     """
#     if left:
#         return (d*mtx.T).T
#     else:
#         return d*mtx

def get_s(Y):
  Y_shape = Y.shape
  p = Y_shape[0]
  s = []
  acc = 0
  for row in Y:
    acc += (np.linalg.norm(row, 2)) ** 2
  acc = acc / p
  for row in Y:
    yi_sq = (np.linalg.norm(row, 2)) ** 2
    s.append(yi_sq - acc)
  return np.array(s)

# (n x p)(p x p)(p x n)
# return: n x n
def get_YSY(Y):
  s = get_s(Y)  
  YT = Y.T
  YS = s * YT
  YSY = YS.dot(Y)
  return YSY

def sparse_rec(Y, YSY, q, num_it):
  if num_it > 0:
    for k in range(0, num_it):
      num = YSY.dot(q)
      denom = np.linalg.norm(num, 2)
      q = num / denom
  else:
    q_old = q
    while True:
      # print 'q: ', q
      num = YSY.dot(q)
      denom = np.linalg.norm(num, 2)
      q = num / denom
      if np.linalg.norm(q - q_old, 2) < 1e-8:
        break
      q_old = q
  return Y.dot(q)

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

def error(x0, x):
  x0_normalized = x0 / np.linalg.norm(x0, 2)
  return np.linalg.norm(x0_normalized - x, 2)

def error_2(x0, x):
  return np.linalg.norm(x0 - x, 2)

def deriv_p(n):
  return int(5 * n * math.log(n))

def gen_instance(n, k):
  p = deriv_p(n)
  S, x0 = gen_subspace(p, n, k) # n x p
  # print 'S, x0: ', S, x0
  Y = GS(S) # n x p
  Y = Y.T # p x n
  return Y, x0

def run_random(n, k, num_it_alg, num_it_test, Y_rows = True):
  Y, x0 = gen_instance(n, k)
  YSY = get_YSY(Y)
  best = float("inf")
  recs = []

  for j in range(0, num_it_test):
    i = rand.randint(0, len(Y) - 1)
    # print 'i: ', i
    q = None
    if Y_rows:
      q = Y[i]
    else:
      q = np.random.normal(0, scale = 1, size = (1, len(Y[i])))[0]
    x = sparse_rec(Y, YSY, q, num_it_alg)
    err = error(x0, x)
    if err < best:
      # print 'x0: ', x0
      # print 'Yq: ', Yq
      best = err
      # if len(recs) <= j:
      #   recs.append((x0, x))
      # else:
      #   recs[j - 1] = (x0, x)
  return best, recs

# def run(n, k, num_it_alg):
#   Y, x0 = gen_instance(n, k)
#   YSY = get_YSY(Y)
#   best = float("inf")
#   recs = []

#   for j in range(0, len(Y)):
#     q = Y[j]
#     x = sparse_rec(Y, YSY, q, num_it_alg)
#     err = error(x0, x)
#     if err < best:
#       # print 'x0: ', x0
#       # print 'Yq: ', Yq
#       best = err
#       # if len(recs) <= j:
#       #   recs.append((x0, x))
#       # else:
#       #   recs[j - 1] = (x0, x)
#   return best, recs

# def test_random(n_range, k_range, num_it_alg, ep, num_it_test, num_pts):
#   succ = ([], [])
#   fail = ([], [])
#   errs = []
#   for t in range(0, num_pts):
#     print 'point number: ', t
#     n = rand.randint(n_range[0], n_range[-1])
#     p = int(deriv_p(n))
#     k = rand.randint(k_range[0], k_range[-1])
#     err, recs = run_random(n, k, num_it_alg, num_it_test)
#     errs.append(err)
#     if err <= ep:
#       succ[0].append(p)
#       succ[1].append(k)
#     else:
#       fail[0].append(p)
#       fail[1].append(k)
#   return succ, fail, errs

def test_rand2(n_range, k_range, num_it_alg, num_it_test, num_pts, Y_rows = True):
  errs = ([], [], [])
  for t in range(0, num_pts):
    print 'point number: ', t
    n = rand.randint(n_range[0], n_range[-1])
    p = int(deriv_p(n))
    k = rand.randint(k_range[0], k_range[-1])
    err, recs = run_random(n, k, num_it_alg, num_it_test, Y_rows)
    errs[0].append(p)
    errs[1].append(k)
    errs[2].append(err)
  return errs

def test_single_rand(n, k, num_it_alg, num_it_test):
  p = int(deriv_p(n))
  err, recs = run_random(n, k, num_it_alg, num_it_test)
  return err

def test_single(n, k, num_it_alg):
  p = int(deriv_p(n))
  err, recs = run(n, k, num_it_alg)
  return err, recs

# def plot(succ, fail):
#   plt.axis([0, 2200, 0, 550])
#   plt.xlabel('Ambient Dimension p')
#   plt.ylabel('Sparsity k')
#   plt.plot(succ[0], succ[1], 'bs')
#   plt.plot(fail[0], fail[1], 'rs')
#   plt.show()
#   x = input()

def plot2(errs):
  plt.axis([0, 2200, 0, 550])
  plt.xlabel('Ambient Dimension p')
  plt.ylabel('Sparsity k')
  plt.scatter(errs[0], errs[1], c = errs[2], s = 100, marker = 's')
  plt.gray()
  plt.show()
  x = input()

def normalize_errs(errs):
  for i in range(0, len(errs[0])):
    if errs[2][i] >= 1:
      errs[2][i] = 0
    else:
      errs[2][i] = 1 - errs[2][i]
  errs[0].append(0)
  errs[1].append(0)
  errs[2].append(1)

# succ, fail, errs = test_random(range(2,90), range(2,400), 0, 0.001, 20, 200)
# print succ, fail, errs
# plot(succ, fail)


# print test_single(5, 3, 0, 10)
# n = 60
# k = 30
# err, recs = test_single(n, k, 0)
# print recs
# print err
# print deriv_p(n)

# test_file()

errs = test_rand2(range(2,90), range(2,300), 0, 50, 50, Y_rows = True)
print errs
normalize_errs(errs)
# print errs
plot2(errs)

# test_mult()




















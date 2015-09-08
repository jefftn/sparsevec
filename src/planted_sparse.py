import math
import csv
import numpy as np
import random as rand
import spectral
import matplotlib.pyplot as plt

print 'Planted Sparse Vector Recovery Comparison Tests'

NONLIN_RANDVEC = 1
NONLIN_RANDROW = 2
NONLIN_SPECROW = 3
LINEAR_OPERATR = 4
LIN_AND_NONLIN = 5

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

def fast_planted(Y, q, num_it, ret_q = False):
  YSY = get_YSY(Y)
  for k in range(0, num_it):
    num = YSY.dot(q)
    denom = np.linalg.norm(num, 2)
    q = num / denom
  if ret_q:
    return q
  return Y.dot(q)

def S_lam(v, lam):
  for i in range(0, len(v)):
    vi = v[i]
    v[i] = np.sign(vi) * max(abs(vi) - lam, 0)
  return v

# Y: p x n, q: n x 1
def sparse_rec(Y, q, lam, num_it):
  x = None
  YT = Y.T
  for k in range(0, num_it):
    Y_q_prod = Y.dot(q)
    old_x = x
    x = S_lam(Y.dot(q), lam)
    if np.linalg.norm(x, 2) == 0:
      return Y_q_prod
    y_x_prod = YT.dot(x)
    y_x_prod_norm = np.linalg.norm(y_x_prod, 2)
    q = y_x_prod / y_x_prod_norm
  return Y.dot(q)

def deriv_p(n):
  return int(5 * n * math.log(n))

def deriv_p2(n):
  return int(n * n)

def deriv_k(p):
  return int(0.1 * p)

def deriv_k2(p):
  return int(0.05 * p)

def deriv_lam(p):
  return 1 / math.sqrt(p)

def gen_k_sparse(k, p):
  if k >= p:
    return [1] * p
  indices = range(0, p)
  rand.shuffle(indices)
  x = [0] * p
  for i in range(0, k):
    x[indices[i]] = 1
  return x

def gen_k_sparse2(k, p):
  if k >= p:
    return [1] * p
  x = [0] * p
  counter = 0
  while counter < k:
    i = rand.randint(0, p - 1)
    if x[i] == 0:
      x[i] = 1
      counter += 1
  return x

# rows of return matrix are basis vectors
def gen_subspace(p, n, k):
  stdev = 1/float(p)
  G = np.random.normal(0, scale = stdev, size = (n - 1, p))
  x0 = gen_k_sparse2(k, p)
  S = np.vstack([G, x0])
  return S, x0

def GS(S):
  Y = spectral.orthogonalize(S)
  return Y

def gen_instance(n, k, p_from_n):
  p = p_from_n(n)
  S, x0 = gen_subspace(p, n, k) # n x p
  Y = GS(S) # n x p
  Y = Y.T # p x n
  return Y, x0

def error(x0, x1):
  x0_unit = x0 / np.linalg.norm(x0, 2)
  x1_unit = x1 / np.linalg.norm(x1, 2)
  return abs(np.inner(x0_unit, x1_unit))

def run_instance(Y, x0, num_it, num_it_alg, lam, alg_type):
  accs = []
  for k in range(0, num_it):
    # print 'it: ', k
    x1 = None
    if alg_type == LINEAR_OPERATR:
      q = np.random.normal(0, scale = 1, size = (1, len(Y[0])))[0]
      x1 = fast_planted(Y, q, num_it_alg)
    elif alg_type == LIN_AND_NONLIN:
      q = np.random.normal(0, scale = 1, size = (1, len(Y[0])))[0]
      # print q.shape
      x_intermediate = fast_planted(Y, q, num_it_alg, ret_q = True)
      # print x_intermediate.shape
      x1 = sparse_rec(Y, x_intermediate, lam, num_it_alg)
    else:
      q = None
      if alg_type == NONLIN_RANDVEC:
        q = np.random.normal(0, scale = 1, size = (1, len(Y[0])))[0]
      elif alg_type == NONLIN_RANDROW:
        i = rand.randint(0, len(Y) - 1)
        q = Y[i]
      elif alg_type == NONLIN_SPECROW:
        indices = [i for i, x in enumerate(x0) if x == 1]
        rand.shuffle(indices)
        q = Y[indices[0]]
      else:
        print "invalid!"
      x1 = sparse_rec(Y, q, lam, num_it_alg)
    accs.append(error(x0, x1))
  return accs

def run(n_range, deriv_p, deriv_k, num_it, num_it_alg, alg_type):
  results = []
  for n in n_range:
    # print "n: ", n
    p = deriv_p(n)
    k = deriv_k(p)
    lam = deriv_lam(p)
    Y, x0 = gen_instance(n, k, deriv_p)
    accs = run_instance(Y, x0, num_it, num_it_alg, lam, alg_type)
    res = (n, accs)
    results.append(res)
  # plot(resuts)
  return results

def plot(results):
  x = []
  y = []
  lower_error = []
  upper_error = []
  for res in results:
    n = res[0]
    accs = res[1]
    x.append(n)
    mean = np.mean(accs)
    stdev = np.std(accs)
    y.append(mean)
    # lower_error.append(mean - min(accs))
    # upper_error.append(max(accs) - mean)
    plt.plot([n], [max(accs)], color='r', marker='_', markersize=8)
    upper = stdev if mean + stdev < 1 else max(accs) - mean
    lower = stdev if mean - stdev > 0 else mean - min(accs)
    lower_error.append(lower)
    upper_error.append(upper)
  # print 'x: ', x
  # print 'y: ', y
  # print 'lower: ', lower_error
  # print 'upper: ', upper_error
  plt.xlabel('Subspace Dimension n')
  plt.ylabel('Accuracy')
  plt.axis([x[0] - 2, x[-1] + 2, -0.5, 1.5])
  asymmetric_error = [lower_error, upper_error]
  plt.errorbar(x, y, yerr=asymmetric_error, fmt='o')
  plt.show()
  # x = input()


# res = run(range(20, 90, 2), deriv_p, deriv_k, num_it=12, 
#           num_it_alg=130, alg_type = NONLIN_RANDVEC)
# res = run(range(20, 90, 2), deriv_p, deriv_k, num_it=12, 
#           num_it_alg=130, alg_type = NONLIN_RANDROW)
# res = run(range(20, 90, 2), deriv_p, deriv_k, num_it=12, 
#           num_it_alg=130, alg_type = NONLIN_SPECROW)
# res = run(range(20, 90, 2), deriv_p, deriv_k, num_it=12, 
#           num_it_alg=130, alg_type = LINEAR_OPERATR)
# res = run(range(20, 90, 2), deriv_p, deriv_k, num_it=12, 
#           num_it_alg=130, alg_type = LIN_AND_NONLIN)
# print res
# plot(res)

# res = run(range(70, 75, 5), deriv_p2, deriv_k, num_it=20, 
#           num_it_alg=130, alg_type = NONLIN_RANDVEC)
# plot(res)

# NONLIN_RANDVEC = 1
# NONLIN_RANDROW = 2
# NONLIN_SPECROW = 3
# LINEAR_OPERATR = 4
# LIN_AND_NONLIN = 5








































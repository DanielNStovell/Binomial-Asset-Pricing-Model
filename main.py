import numpy as np

# S_{i, j} = S_0u^jd^{i-j}

# Initial parameters
S_0 = 100     # initial stock price
K = 100       # strike price
T = 1         # time to maturity in years
r = 0.06      # annual risk-free rate
N = 3         # number of time steps
sigma = 0.2   # volatility
opttype = 'C' # Option Type 'C' or 'P'


def binomial_tree_slow(K, T, S_0, r, N, sigma, opttype='C'):
  # precompute constants
  dt = T / N
  u = np.exp(sigma * np.sqrt(dt))
  d = 1/u
  q = (np.exp(r * dt) - d) / (u - d)
  disc = np.exp(-r * dt)

  # initialise asset prices at maturity - Time step N
  S = np.zeros(N + 1)
  S[0] = S_0 * d ** N
  for j in range(1, N + 1):
    S[j] = S[j - 1] * u / d

  # initialise option values at maturity
  C = np.zeros(N + 1)
  for j in range(0, N + 1):
    C[j] = max(0, S[j] - K)

  # step backwards through tree
  for i in np.arange(N, 0, -1):
    for j in range(0, i):
      C[j] = disc * (q * C[j + 1] + (1 - q) * C[j])

  return C[0]


def binomial_tree_fast(K, T, S_0, r, N, sigma, opttype='C'):
  # precompute constants
  dt = T / N
  u = np.exp(sigma * np.sqrt(dt))
  d = 1/u
  q = (np.exp(r * dt) - d) / (u - d)
  disc = np.exp(-r * dt)

  # initialise asset prices at maturity - Time step N
  C = S_0 * d ** (np.arange(N, -1, -1)) * u ** (np.arange(0, N + 1, 1))

  # initialise option values at maturity
  C = np.maximum(C - K, np.zeros(N + 1))

  # step backwards through tree
  for i in np.arange(N, 0, -1):
    C = disc * (q * C[1:i + 1] + (1 - q) * C[0:i])

  return C[0]

for N in [3, 50, 100, 1000, 5000]:
  print(binomial_tree_slow(K, T, S_0, r, N, sigma, opttype='C'))
  print(binomial_tree_fast(K, T, S_0, r, N, sigma, opttype='C'))
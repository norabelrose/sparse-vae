#  tsne_torch.py
#
# Implementation of t-SNE in pytorch. The implementation was tested on pytorch
# > 1.0, and it requires Numpy to read files. In order to plot the results,
# a working installation of matplotlib is required.
#
#
# The example can be run by executing: `python tsne_torch.py`
#
#
#  Created by Xiao Li on 23-03-2020.
#  Copyright (c) 2020. All rights reserved.
from math import log
import torch


def Hbeta_torch(D, beta=1.0):
    P = torch.exp(-D * beta)
    sumP = P.sum()

    H = sumP.log() + beta * torch.sum(D * P) / sumP
    return H, P / sumP


def pairwise_dist_sq(x):
    norms = x.pow(2.0).sum(dim=-1)
    return (norms - 2.0 * x @ x.T).T + norms

def newton1d(f, f_prime, x, tol=1e-5):
    while True:
        delta = f(x) / f_prime(x)
        x -= delta
        if abs(delta) < tol:
            return x

def find_beta_quasinewton(x, k=30.0, tol=1e-5, max_iter=5, knn=250):
    n, d = x.shape

    # First quickly find the global parameter p1 using Newton's method on the CPU
    const = log(min((2 * n) ** 0.5, k))
    p1 = newton1d(
        f=lambda p: 2 * (1 - p) * log(n / (2 * (1 - p))) - const,
        f_prime=lambda p: -2 * log(n) + 2 * log(2 - 2 * p) + 2,
        x = 0.875   # Solution will always be in interval [0.75, 1.0]
    )
    odds1 = p1 / (1 - p1)

    # Use a sparse distance matrix with only the top K closest points
    dists = pairwise_dist_sq(x).topk(k=knn + 1, largest=False)[:, 1:]

    # Compute lower and upper bounds on beta for each point- See Theorem 1.1
    delta_n = dists[:, -1] - dists[:, 0]
    delta_2 = dists - dists[:, 0]
    upper = log(odds1 * (n - 1)) / (delta_2 + 1e-7)
    lower = torch.maximum(
        n * log(n / k) / ((n - 1) * delta_n),
        (log(n / k) / (dists[:, -1] ** 2 - dists[:, 0] ** 2)) ** 0.5
    )

    # First guess: try the average of the upper and lower bounds
    betas = (upper + lower) / 2.0
    for i in range(max_iter):
        z = dists.neg().mul(betas[:, None]).exp().sum(dim=-1)
        m1 = -z.reciprocal() * betas

def x2p_torch(X, tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """
    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape

    D = pairwise_dist_sq(X)
    P = torch.zeros(n, n)
    beta = torch.ones(n, 1)
    logU = torch.log(torch.tensor([perplexity]))
    n_list = list(range(n))

    # Loop over all datapoints
    for i in range(n):
        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        # there may be something wrong with this setting None
        betamin = None
        betamax = None
        Di = D[i, n_list[:i] + n_list[i + 1:n]]

        (H, thisP) = Hbeta_torch(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while torch.abs(Hdiff) > tol and tries < 50:
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].clone()
                beta[i] = beta[i] * 2. if betamax is None else (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].clone()
                beta[i] = beta[i] / 2. if betamin is None else (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta_torch(Di, beta[i])

            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, n_list[0:i] + n_list[i + 1:n]] = thisP

    # Return final P-matrix
    return P


def pca_torch(x, k=50):
    print("Initializing using PCA...")
    x = x - x.mean(dim=0)  # De-mean
    _, M = torch.symeig(x.T @ x, eigenvectors=True)
    return x @ M[:, :k]


def tsne(x, target_dims=2, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntax of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """
    assert x.ndim == 2
    assert isinstance(target_dims, int), "Number of dimensions should be an integer."

    # Initialize variables
    n, d = x.shape
    x = pca_torch(x, round(d ** 0.5))  # Use PCA to reduce down to sqrt(N) dimensions
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = torch.randn(n, target_dims)
    dY = torch.zeros(n, target_dims)
    iY = torch.zeros(n, target_dims)
    gains = torch.ones(n, target_dims)

    # Compute P-values
    P = x2p_torch(x, 1e-5, perplexity)
    P = (P + P.T) / P.sum()
    P = P * 4.  # early exaggeration
    print("Got P shape:", P.shape)
    P = torch.max(P, torch.tensor([1e-21]))

    # Run iterations
    for iter in range(max_iter):
        # Compute pairwise affinities
        affinities = 1. / (1. + pairwise_dist_sq(Y))
        affinities.fill_diagonal_(0.0)
        Q = affinities / affinities.sum()
        Q = torch.max(Q, torch.tensor([1e-12]))

        # Compute gradient
        diff = P - Q
        prod = diff * affinities
        y_diffs = Y[:, None] - Y
        for i in range(n):
            dY[i, :] = torch.sum(prod[None, :, i].T * y_diffs[i], 0)

        # Perform the update
        momentum = initial_momentum if iter < 20 else final_momentum

        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)).double() + (gains * 0.8) * ((dY > 0.) == (iY > 0.)).double()
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - Y.mean(dim=0)   # Re-center at every iteration

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = torch.sum(P * torch.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y

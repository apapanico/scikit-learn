"""Low Rank Plus Sparse Matrix Decomposition with Robust PCA"""
# Author: Alex Papanicolaou
# License: BSD 3 clause

import numpy as np

from numpy.linalg import svd

# Use PROPACK partial SVD if available, otherwise SciPy
try:
    from pypropack import svdp as svds
except:
    from scipy.sparse.linalg import svds


from ..utils import check_array
from ..utils.validation import check_is_fitted
from ..base import BaseEstimator, TransformerMixin


class RobustPCA(BaseEstimator, TransformerMixin):

    """Robust Principal Components Analysis (RobustPCA)

    Finds the Principal Component Pursuit solution.

    Solves the optimization problem::

        (L^*,S^*) = argmin || L ||_* + gamma * || S ||_1
                    (L,S)
                    subject to    L + S = X

    where || . ||_* is the nuclear norm.  Uses an augmented Lagrangian approach

    Read more in the :ref:`User Guide <RobustPCA>`.

    Parameters
    ----------

    maxiter: int, 500 by default
        Maximum number of iterations to perform.

    tol: float, 1e-6 by default

    Attributes
    ----------

    L_: array of shape, [n_components, n_features]
        The low rank component

    S_: array of shape, [n_components, n_features]
        The sparse component

    n_iter_ : int
        Number of iterations run.

    Notes
    -----

    **References:**

        Candes, Li, Ma, and Wright
        Robust Principal Component Analysis?
        Submitted for publication, December 2009.
        (http://arxiv.org/abs/0912.3599)

    """

    def __init__(self, maxiter=500, tol=1e-6):
        self.maxiter = maxiter
        self.tol = tol

    def fit(self, X, y=None, gamma=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        n_samples, n_features = check_array(X).shape

        if gamma is None:
            gamma = 1 / np.sqrt(np.max([n_samples, n_features]))
        self.gamma = gamma

        self.L_, self.S_, self.n_iter_ = rpca_alm(
            X,
            gamma=gamma,
            maxiter=self.maxiter,
            tol=self.tol
        )
        return self


def rpca_alm(X, gamma=None, maxiter=500, tol=1e-6, check_input=True):
    """Principal Component Pursuit

    Finds the Principal Component Pursuit solution.

    Solves the optimization problem::

        (L^*,S^*) = argmin || L ||_* + gamma * || S ||_1
                    (L,S)
                    subject to    L + S = X

    where || . ||_* is the nuclear norm.  Uses an augmented Lagrangian approach

    Parameters
    ----------

    X: array of shape (n_samples, n_features)
        Data matrix.

    gamma: float, 1/sqrt(max(n_samples, n_features)) by default
        The l_1 regularization parameter to be used.

    maxiter: int, 500 by default
        Maximum number of iterations to perform.

    tol: float, 1e-6 by default

    check_input: boolean, optional
        If False, the input array X will not be checked.

    Returns
    -------

    L: array of shape (n_components, n_features)
        The low rank component

    S: array of shape (n_components, n_features)
        The sparse component

    n_iter: int
        Number of iterations

    Reference
    ---------

       Candes, Li, Ma, and Wright
       Robust Principal Component Analysis?
       Submitted for publication, December 2009.

    """

    if check_input:
        X = check_array(X, ensure_min_samples=2)

    n = X.shape
    Frob_norm = np.linalg.norm(X, 'fro')
    two_norm = np.linalg.norm(X, 2)
    one_norm = np.sum(np.abs(X))
    inf_norm = np.max(np.abs(X))

    mu_inv = 4 * one_norm / np.prod(n)

    # Kicking
    k = np.min([
        np.floor(mu_inv / two_norm),
        np.floor(gamma * mu_inv / inf_norm)
    ])
    Y = k * X
    sv = 10

    # Variable init
    zero_mat = np.zeros(n)
    S = zero_mat.copy()
    L = zero_mat.copy()
    R = X.copy()
    T1 = zero_mat.copy()
    T2 = zero_mat.copy()

    np.multiply(Y, mu_inv, out=T1)
    np.add(T1, X, out=T1)

    for i in range(maxiter):
        # Shrink entries
        np.subtract(T1, L, out=T2)
        S = _vector_shrink(T2, gamma * mu_inv, out=S)

        # Shrink singular values
        np.subtract(T1, S, out=T2)
        L, r = _matrix_shrink(T2, mu_inv, sv, out=L)

        if r < sv:
            sv = np.min([r + 1, np.min(n)])
        else:
            sv = np.min([r + np.round(0.05 * np.min(n)), np.min(n)])

        np.subtract(X, L, out=R)
        np.subtract(R, S, out=R)
        stopCriterion = np.linalg.norm(R, 'fro') / Frob_norm

        # Check convergence
        if stopCriterion < tol:
            break

        # Update dual variable
        np.multiply(R, 1. / mu_inv, out=T2)
        np.add(T2, Y, out=Y)
        Y += R / mu_inv

        np.add(T1, R, out=T1)

    return L, S, i + 1


def lad(X, b, rho, alpha, maxiter=1000, is_orth=False):
    """Least absolute deviations fitting via ADMM

    Solves the optimization problem via ADMM:

        (w^*,s^*) = argmin || s ||_1
                      s
                    subject to   X w + s = b

    The solution is returned in the vector w and the residual is in s.

    Parameters
    ----------

    X: array of shape (m, n)
        Data matrix.

    b: array of shape (n)
        Target values

    rho: float
        The augmented Lagrangian parameter

    alpha: float
        The over-relaxation parameter (typical values for alpha
        are between 1.0 and 1.8).

    Returns
    -------

    w: array of shape (n)
        Solution

    s: array of shape (n)
        Residual

    Reference
    ---------

       S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein.  Distributed
       optimization and statistical learning via the alternating direction
       method ofmultipliers.  Foundations and Trends in Machine Learning,
       3(1):1–124, 2011.

       http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html

    """

    # Global constants and defaults

    _ABSTOL = 1e-4
    _RELTOL = 1e-2

    m, n = X.shape

    # ADMM solver
    w = np.zeros(n, 1)
    s = np.zeros(m, 1)
    u = np.zeros(m, 1)

    if is_orth:
        R = sparse.eye(n, n)
    else:
        R = np.chol(X.T.dot(X))

    for k in range(maxiter):

        w_tmp = X.T.dot(b + s - u)
        w = np.linalg.solve(
            R,
            np.linalg.solve(
                R.T,
                x_tmp
            )
        )

        s_old = s
        Ax_hat = alpha * X.dot(w) + (1 - alpha) * (s_old + b)
        _vector_shrink(Ax_hat - b + u, 1 / rho, out=s)

        u = u + (Ax_hat - s - b)

        history.objval(k) = objective(s)

        r_norm = np.linalg.norm(A * w - s - b)
        s_norm = np.linalg.norm(-rho * A.T.dot(s - s_old))

        eps_pri = np.sqrt(m) * _ABSTOL + \
            _RELTOL * np.max([
                np.linalg.norm(A.dot(w)),
                np.linalg.norm(-s),
                np.linalg.norm(b)
            ])
        eps_dual = np.sqrt(n) * _ABSTOL + _RELTOL * norm(rho * A.T.dot(u))

        if (r_norm < eps_pri) and (s_norm < eps_dual):
            break

    return w, s


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Auxilliary functions
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _matrix_shrink(X, tau, sv, out=None):
    m = np.min(X.shape)

    if _choosvd(m, sv):
        U, sig, V = svds(X, int(sv))
    else:
        U, sig, V = svd(X, full_matrices=0)

    r = np.sum(sig > tau)
    if r > 0:
        np.multiply(U[:, :r], (sig[:r] - tau), out=X[:, :r])
        Z = np.dot(X[:, :r], V[:r, :], out=out)
    else:
        out[:] = 0
        Z = out
    return (Z, r)


def _vector_shrink(X, tau, out=None):
    np.absolute(X, out=out)
    np.subtract(out, tau, out=out)
    np.maximum(out, 0.0, out=out)
    return np.multiply(np.sign(X), out, out=out)


def _choosvd(n_int, d_int):
    n = float(n_int)
    d = float(d_int)
    if n <= 100:
        if d / n <= 0.02:
            return True
        else:
            return False
    elif n <= 200:
        if d / n <= 0.06:
            return True
        else:
            return False
    elif n <= 300:
        if d / n <= 0.26:
            return True
        else:
            return False
    elif n <= 400:
        if d / n <= 0.28:
            return True
        else:
            return False
    elif n <= 500:
        if d / n <= 0.34:
            return True
        else:
            return False
    else:
        if d / n <= 0.38:
            return True
        else:
            return False

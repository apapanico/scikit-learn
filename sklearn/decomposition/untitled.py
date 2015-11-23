import numpy as np
from scipy import sparse


def lad(X, b, rho, alpha, is_orth=False):

    # % lad  Least absolute deviations fitting via ADMM
    # %
    # % [x, history] = lad(X, b, rho, alpha)
    # %
    # % Solves the following problem via ADMM:
    # %
    # %   minimize     ||Ax - b||_1
    # %
    # % The solution is returned in the vector x.
    # %
    # % history is a structure that contains the objective
    # value, the primal and
    # % dual residual norms, and the tolerances for the primal
    # and dual residual
    # % norms at each iteration.
    # %
    # % rho is the augmented Lagrangian parameter.
    # %
    # % alpha is the over-relaxation parameter (typical values for alpha are
    # % between 1.0 and 1.8).
    # %
    # %
    # % More information can be found in the paper linked at:
    # % http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
    # %

    # Global constants and defaults

    QUIET = 0
    MAX_ITER = 1000
    ABSTOL = 1e-4
    RELTOL = 1e-2

    m, n = X.shape

    # ADMM solver
    x = np.zeros(n, 1)
    z = np.zeros(m, 1)
    u = np.zeros(m, 1)

    if is_orth:
        R = sparse.eye(n, n)
    else:
        R = np.chol(X.T.dot(X))

    for k in range(MAX_ITER):

        x_tmp = X.T.dot(b + z - u)
        x = np.linalg.solve(
            R,
            np.linalg.solve(
                R.T,
                x_tmp
            )
        )

        zold = z
        Ax_hat = alpha * X.dot(x) + (1 - alpha) * (zold + b)
        z = shrinkage(Ax_hat - b + u, 1 / rho)

        u = u + (Ax_hat - z - b)

        history.objval(k) = objective(z)

        r_norm = np.linalg.norm(A * x - z - b)
        s_norm = np.linalg.norm(-rho * A.T.dot(z - zold))

        eps_pri = np.sqrt(m) * ABSTOL + \
            RELTOL * np.max([
                np.linalg.norm(A.dot(x)),
                np.linalg.norm(-z),
                np.linalg.norm(b)
            ])
        eps_dual = np.sqrt(n) * ABSTOL + RELTOL * norm(rho * A.T.dot(u))

        if (r_norm < eps_pri) and (s_norm < eps_dual):
            break

    return x, z


def shrinkage(a, kappa):
    return np.max(0, a - kappa) - np.max(0, -a - kappa)

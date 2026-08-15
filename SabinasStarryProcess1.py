import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as jlx

from jaxoplanet.starry.ylm import Ylm
from functools import partial

from utils import hyp2f1_sequence, gamma_sequence1, gamma_sequence2


def log_factorial(x):
    return jax.lax.lgamma(x + 1.0)

def kron(m, n):
    """Kronecker delta. Can we do better?"""
    return m == n

def Umn(m, n):
    """Compute the (m, n) term of the transformation
    matrix from complex to real Ylms. This part is from starry"""
    if n < 0:
        term1 = 1j
    elif n == 0:
        term1 = jnp.sqrt(2) / 2
    else:
        term1 = 1
    if (m > 0) and (n < 0) and (n % 2 == 0):
        term2 = -1
    elif (m > 0) and (n > 0) and (n % 2 != 0):
        term2 = -1
    else:
        term2 = 1
    return term1 * term2 * 1 / jnp.sqrt(2) * (kron(m, n) + kron(m, -n) + 0j)

def U(l):
    """Compute the complete U transformation matrix.. This part is from starry"""
    res = jnp.zeros((2 * l + 1, 2 * l + 1)) + 0j
    for m in range(-l, l + 1):
        for n in range(-l, l + 1):
            res = res.at[m + l, n + l].set(Umn(m, n))
    return res


def clmmpi_sequence(l, m, mp, imax=None):
    """Returns the sequence of Wigner-d coefficients c^l_{m, m', i}.

    Same structure as the *_sequence functions in utils.py: a seed plus a
    jax.lax.scan recurrence. Stepping i -> i+2 multiplies the coefficient by a
    closed-form rational factor,

        c(i+2) = c(i) * -(cc*dd) / ((a+1)*(b+1))

    with a=(i-m-mp)/2, b=(i+m+mp)/2, cc=(2l-i-m+mp)/2, dd=(2l-i+m-mp)/2, so the
    whole sequence follows from one evaluation of the closed form. The old scalar
    clmmpi recomputed eight lgammas for every (m, mp, i).
    """
    if imax is None:
        imax = 2 * l

    # (m - mp - i) even
    i0 = max(0, m + mp, -m - mp)
    if (m - mp - i0) % 2 != 0:
        i0 += 1

    hi = min(2 * l, 2 * l + m - mp, 2 * l - m + mp)
    if i0 > min(hi, imax):
        return jnp.zeros(imax + 1)

    # closed form, evaluated once at i = i0
    a0 = (i0 - m - mp) / 2
    b0 = (i0 + m + mp) / 2
    c0 = (2 * l - i0 - m + mp) / 2
    d0 = (2 * l - i0 + m - mp) / 2
    log_num = 0.5 * (log_factorial(l - m) + log_factorial(l + m)
                     + log_factorial(l - mp) + log_factorial(l + mp))
    log_denom = (l * jnp.log(2.0) + log_factorial(a0) + log_factorial(b0)
                 + log_factorial(c0) + log_factorial(d0))
    sign0 = 1.0 if ((2 * l - m + mp - i0) / 2 % 2 == 0) else -1.0
    init = sign0 * jnp.exp(log_num - log_denom)

    def scan_f(state, i):
        a = (i - m - mp) / 2
        b = (i + m + mp) / 2
        cc = (2 * l - i - m + mp) / 2
        dd = (2 * l - i + m - mp) / 2
        next_state = state * (-(cc * dd) / ((a + 1) * (b + 1)))
        return next_state, state

    idx = jnp.arange(i0, imax + 1, 2)
    _, result_vals = jlx.scan(scan_f, init, idx)

    result = jnp.zeros(imax + 1)
    return result.at[idx].set(jnp.where(idx <= hi, result_vals, 0.0))


# def clmmpi_tensor(l):
#     """C[m+l, mp+l, i] = c^l_{m, m', i}: from clmmpi_sequence."""
#     return jnp.stack([
#         jnp.stack([clmmpi_sequence(l, m, mp) for mp in range(-l, l + 1)])
#         for m in range(-l, l + 1)
#     ])

def clmmpi_tensor(l):
    """C[m+l, mp+l, i] = c^l_{m, m', i}: the whole Wigner-d coefficient tensor.
 
    Every factorial argument in c^l_{m,m',i} is an integer in 0..2l, so all of them
    are lookups into a single log(n!) table. That table is built by the recurrence
    log(n!) = log((n-1)!) + log(n), i.e. one cumsum -- the same "compute once, index
    many times" idea as the *_sequence functions in utils.py.
 
    This replaces building (2l+1)^2 separate lax.scans (one per (m, mp) pair), which
    dominated compile time: the whole tensor now comes from one table plus broadcast
    indexing.
 
    Note the clamp-then-mask pattern: the four half-integer arguments a, b, c, d go
    negative for invalid (m, mp, i), and we cannot branch per element inside
    vectorised code. So we clip the indices to keep the lookup in bounds, then zero
    the invalid entries with `valid` -- reproducing the three early `return 0.0`
    guards of the original scalar clmmpi.
    """
    N = 4 * l + 2
    n = jnp.arange(N + 1)
    LF = jnp.cumsum(jnp.where(n > 0, jnp.log(jnp.maximum(n, 1)), 0.0))
 
    m = jnp.arange(-l, l + 1)[:, None, None]
    mp = jnp.arange(-l, l + 1)[None, :, None]
    i = jnp.arange(0, 2 * l + 1)[None, None, :]
 
    a = (i - m - mp) // 2
    b = (i + m + mp) // 2
    c = (2 * l - i - m + mp) // 2
    d = (2 * l - i + m - mp) // 2
 
    log_num = 0.5 * (LF[l - m] + LF[l + m] + LF[l - mp] + LF[l + mp])
    log_denom = (l * jnp.log(2.0) + LF[jnp.clip(a, 0, N)] + LF[jnp.clip(b, 0, N)]
                 + LF[jnp.clip(c, 0, N)] + LF[jnp.clip(d, 0, N)])
 
    sign = jnp.where(((2 * l - m + mp - i) // 2) % 2 == 0, 1.0, -1.0)
    val = sign * jnp.exp(log_num - log_denom)
 
    valid = (((m - mp - i) % 2) == 0) & (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    return jnp.where(valid, val, 0.0)

def qlphi(alpha, beta, l):
    return gamma_sequence1(l, alpha, beta, 2*l) * hyp2f1_sequence(l, alpha, beta, 2*l)


def Qllpphi(alpha, beta, l, lp):
    L = l + lp
    return gamma_sequence1(L, alpha, beta, 2*L) * hyp2f1_sequence(L, alpha, beta, 2*L)


def qllambda(l):
    return gamma_sequence2(l, 1.0, 1.0, 2*l)


def Qllambda(l, lp):
    L = l + lp
    return gamma_sequence2(L, 1.0, 1.0, 2*L)


def ebar_func(er, l):
    er_array = jnp.array([er[l, m] for m in range(-l, l + 1)])
    ebar = U(l) @ er_array

    return ebar


def plphim(l, ebar, alpha, beta, q):
    """The whole (p^l_phi) vector, shape (2l+1,)."""
    C0 = clmmpi_tensor(l)[:, l, :]                  # (m, i): the mp = 0 slice
    term1 = jnp.exp(jax.lax.lgamma(alpha + beta)
                    - (jax.lax.lgamma(alpha) + jax.lax.lgamma(beta)))
    m = jnp.arange(-l, l + 1)
    return term1 * ebar[l] * jnp.exp(1j * jnp.pi * m / 2) * (C0 @ q)


def pllambdam(l, ebarphi, q):
    """The whole (p^l_lambda) vector."""
    return jnp.einsum('ami,i,m->a', clmmpi_tensor(l), q, ebarphi)


def Pllpphimmp(l, lp, alpha, beta, ebar, ebarp, Q, C_l, C_lp):
    """The whole (2l+1, 2lp+1) P^{l,l'}_phi block.
    """
    c1 = C_l[:, l, :]                  # (m, i)
    c2 = C_lp[:, lp, :]                # (mp, ip)
    i = jnp.arange(2*l + 1)[:, None]
    ip = jnp.arange(2*lp + 1)[None, :]
    sumterm = jnp.einsum('ai,bj,ij->ab', c1, c2, Q[i + ip])

    term1 = jnp.exp(jax.lax.lgamma(alpha + beta)
                    - (jax.lax.lgamma(alpha) + jax.lax.lgamma(beta)))
    mm = jnp.arange(-l, l + 1)[:, None]
    mmp = jnp.arange(-lp, lp + 1)[None, :]
    return term1 * ebar[l] * ebarp[lp] * jnp.exp(1j * jnp.pi / 2 * (mm + mmp)) * sumterm


def Pllplambdammp(l, lp, Ebarphi_block, Q, C_l, C_lp):
    """The whole (2l+1, 2lp+1) P^{l,l'}_lambda block.
    """
    i = jnp.arange(2*l + 1)[:, None]
    ip = jnp.arange(2*lp + 1)[None, :]
    return jnp.einsum('aui,bvj,ij,uv->ab', C_l, C_lp, Q[i + ip], Ebarphi_block)


def _legendre_basis(lmax, x):
    """Stacked Legendre polynomials P_0(x)..P_lmax(x), pure jax.

    Uses the Bonnet recurrence (l+1) P_{l+1} = (2l+1) x P_l - l P_{l-1}.
    `x` is an array; returns shape (lmax+1, len(x)).
    """
    P0 = jnp.ones_like(x)
    if lmax == 0:
        return jnp.stack([P0])
    P = [P0, x]
    for l in range(1, lmax):
        P.append(((2 * l + 1) * x * P[l] - l * P[l - 1]) / (l + 1))
    return jnp.stack(P)


def _spot_basis(lmax, spts, eps, smoothing):

    theta = jnp.linspace(0.0, jnp.pi, spts)
    cost = jnp.cos(theta)
    l = jnp.arange(lmax + 1)

    P = _legendre_basis(lmax, cost)
    B = (jnp.sqrt(2.0 * l + 1.0)[:, None] * P).T
    Bp = jnp.linalg.solve(B.T @ B + eps * jnp.eye(lmax + 1), B.T)
    idx = l * (l + 1)
    Sm = jnp.exp(-0.5 * idx * smoothing ** 2)
    Bp = Sm[:, None] * Bp
    return theta, Bp, idx


def compute_radius_moment(lmax, r, c, smoothing=0.075, spts=1000, eps=1e-9, sfac=300):

    theta, Bp, idx = _spot_basis(lmax, spts, eps, smoothing)
    z = sfac * (theta - r)
    b = 1.0 / (1.0 + jnp.exp(-z)) - 1.0
    yl = Bp @ b
    N = (lmax + 1) ** 2
    y = jnp.zeros(N).at[idx].set(yl)
    return Ylm.from_dense(y, normalize=False)

@partial(jax.jit, static_argnames="lmax")
def compute_first_moment(alpha, beta, r, c, lmax):
    """
    Compute E[y]
    """
    er = compute_radius_moment(lmax, r, c)

    ephi = []
    elambda = []

    for l in range(0, lmax + 1):
        u = U(l)

        q_phi_l = qlphi(alpha, beta, l)
        q_lambda_l = qllambda(l)

        erl = jnp.array([er[l, m] for m in range(-l, l + 1)])
        erlbar = u @ erl
        p = plphim(l, erlbar, alpha, beta, q_phi_l)

        ephi_ = jnp.real(jnp.conj(u.T) @ p)
        ephi.append(ephi_)
        ebarphi = u @ ephi_

        plambda = pllambdam(l, ebarphi, q_lambda_l)
        elambda.append(jnp.real(jnp.conj(u.T) @ plambda))

    elambda = jnp.concatenate(elambda)
    ephi = jnp.concatenate(ephi)

    return elambda

@partial(jax.jit, static_argnames="lmax")
def compute_second_moment(alpha, beta, r, c, lmax):
    N = (lmax + 1) ** 2
    er = compute_radius_moment(lmax, r, c)

    Us = [U(l) for l in range(lmax + 1)]
    Cs = [clmmpi_tensor(l) for l in range(lmax + 1)]

    Elambda = jnp.zeros((N, N), dtype=jnp.complex128)

    for l in range(lmax + 1):

        Ul = Us[l]
        erbarl = ebar_func(er, l)

        for lp in range(lmax + 1):

            Ulp = Us[lp]
            erbarlp = ebar_func(er, lp)

            Q_phi_llp = Qllpphi(alpha, beta, l, lp)
            Q_lambda_llp = Qllambda(l, lp)

            Pphi = Pllpphimmp(l, lp, alpha, beta, erbarl, erbarlp, Q_phi_llp, Cs[l], Cs[lp])

            Ebarphi_block = Pphi

            pllp = Pllplambdammp(l, lp, Ebarphi_block, Q_lambda_llp, Cs[l], Cs[lp])

            block = jnp.conj(Ul).T @ pllp @ jnp.conj(Ulp)
            Elambda = Elambda.at[l**2:l**2 + 2*l + 1, lp**2:lp**2 + 2*lp + 1].set(block)

    return jnp.real(Elambda)


def compute_covariance_one(alpha, beta, r, c, lmax):

    first = compute_first_moment(alpha, beta, r, c, lmax)
    second = compute_second_moment(alpha, beta, r, c, lmax)
    covariance = jnp.real(second - jnp.outer(first, first))

    return covariance * (jnp.pi * c) ** 2

@partial(jax.jit, static_argnames="lmax")
def compute_mean(alpha, beta, r, c, n, lmax):
    return n * compute_first_moment(alpha, beta, r, c, lmax) * jnp.pi * c

@partial(jax.jit, static_argnames="lmax")
def compute_covariance(alpha, beta, r, c, n, lmax):
    return n * compute_covariance_one(alpha, beta, r, c, lmax)
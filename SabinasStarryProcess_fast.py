import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as jlx

from jaxoplanet.starry.ylm import Ylm
from functools import partial

import functools

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
        term1 = np.sqrt(2) / 2
    else:
        term1 = 1
    if (m > 0) and (n < 0) and (n % 2 == 0):
        term2 = -1
    elif (m > 0) and (n > 0) and (n % 2 != 0):
        term2 = -1
    else:
        term2 = 1
    return term1 * term2 * 1 / np.sqrt(2) * (kron(m, n) + kron(m, -n) + 0j)

# Manual cache of the full complex-to-real Ylm transformation, indexed by lmax.
# The full matrix is block diagonal.
_Uylm_cache = {}


def set_Uylm(lmax):
    """Build/cache the full U transformation matrix through lmax using numpy.
    Degree-l coefficients occupy rows/columns l**2:(l+1)**2, so each U_l block
    can be retrieved with a simple slice. 
    """
    lmax = int(lmax)
    if lmax not in _Uylm_cache:
        N = (lmax + 1) ** 2
        U = np.zeros((N, N), dtype=np.complex128)
        for ell in range(lmax + 1):
            start, stop = ell**2, (ell + 1)**2
            for m in range(-ell, ell + 1):
                for n in range(-ell, ell + 1):
                    U[start + m + ell, start + n + ell] = Umn(m, n)
        _Uylm_cache[lmax] = U
    return _Uylm_cache[lmax]

def _U_block(U_all, l):
    """Return the degree-l diagonal block from a full set_Uylm matrix."""
    start, stop = l**2, (l + 1)**2
    return jnp.asarray(U_all[start:stop, start:stop])

def Uylm(l):
    """Full degree-l transformation matrix."""
    return _U_block(set_Uylm(l), l)

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


# Manual cache of all Wigner-d coefficient tensors through a requested lmax.
# The same as with Uylm, each degree occupies the square index block l**2:(l+1)**2
# along the m and m' axes. The final axis stores i=0,...,2*lmax; each degree
# uses only its first 2*l+1 entries.
_clmmpi_cache = {}


def _build_clmmpi_tensor_numpy(l):
    """Build the degree-l Wigner-d coefficient tensor with numpy."""
    N = 4 * l + 2
    n = np.arange(N + 1)
    LF = np.cumsum(np.where(n > 0, np.log(np.maximum(n, 1)), 0.0))

    m = np.arange(-l, l + 1)[:, None, None]
    mp = np.arange(-l, l + 1)[None, :, None]
    i = np.arange(0, 2 * l + 1)[None, None, :]

    a = (i - m - mp) // 2
    b = (i + m + mp) // 2
    c = (2 * l - i - m + mp) // 2
    d = (2 * l - i + m - mp) // 2

    log_num = 0.5 * (LF[l - m] + LF[l + m] + LF[l - mp] + LF[l + mp])
    log_denom = (l * np.log(2.0) + LF[np.clip(a, 0, N)] + LF[np.clip(b, 0, N)]
                 + LF[np.clip(c, 0, N)] + LF[np.clip(d, 0, N)])
    sign = np.where(((2 * l - m + mp - i) // 2) % 2 == 0, 1.0, -1.0)
    val = sign * np.exp(log_num - log_denom)
    valid = (((m - mp - i) % 2) == 0) & (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    return np.where(valid, val, 0.0)


def set_clmmpi_tensor(lmax):
    """Build/cache all C tensors up to lmax in one numpy array.
    For degree l, retrieve C_l with _C_block(C_all, l). The cache is keyed
    by lmax, so repeated calls at the same maximum degree do no rebuilding.
    """
    lmax = int(lmax)
    if lmax not in _clmmpi_cache:
        N = (lmax + 1) ** 2
        C_all = np.zeros((N, N, 2 * lmax + 1), dtype=np.float64)
        for ell in range(lmax + 1):
            start, stop = ell**2, (ell + 1)**2
            C_l = _build_clmmpi_tensor_numpy(ell)
            C_all[start:stop, start:stop, :2 * ell + 1] = C_l
        _clmmpi_cache[lmax] = C_all
    return _clmmpi_cache[lmax]


def _C_block(C_all, l):
    """Return C_l from the tensor produced by set_clmmpi_tensor."""
    start, stop = l**2, (l + 1)**2
    return jnp.asarray(C_all[start:stop, start:stop, :2 * l + 1])


def clmmpi_tensor(l):
    """Full access function for the degree-l coefficient tensor."""
    return _C_block(set_clmmpi_tensor(l), l)

def qlphi(alpha, beta, l):
    return gamma_sequence1(l, alpha, beta, 2*l) * hyp2f1_sequence(l, alpha, beta, 2*l)


def Qllpphi(alpha, beta, l, lp):
    L = l + lp
    return gamma_sequence1(L, alpha, beta, 2*L) * hyp2f1_sequence(L, alpha, beta, 2*L)


# These sequences only depend on L. Evaluate them
# and cache them like Uylm, instead of staging scans.
_qlambda_cache = {}


def Qllambda(l, lp):
    L = l + lp
    if L not in _qlambda_cache:
        with jax.ensure_compile_time_eval():
            _qlambda_cache[L] = np.asarray(gamma_sequence2(L, 1.0, 1.0, 2*L))
    return jnp.asarray(_qlambda_cache[L])


def qllambda(l):
    return Qllambda(l, 0)


def ebar_func(er, l):
    er_array = jnp.array([er[l, m] for m in range(-l, l + 1)])
    ebar = Uylm(l) @ er_array

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


def beta_norm(alpha, beta):
    """Gamma(alpha + beta) / (Gamma(alpha) Gamma(beta))."""
    return jnp.exp(jax.lax.lgamma(alpha + beta)
                   - (jax.lax.lgamma(alpha) + jax.lax.lgamma(beta)))


def Pllpphimmp(l, lp, alpha, beta, ebar, ebarp, Q, C_l, C_lp, term1=None):
    """The whole (2l+1, 2lp+1) P^{l,l'}_phi block.

    Pass term1 = beta_norm(alpha, beta) when calling in a loop so it is
    computed once rather than per block. Index arrays and phases are NumPy, so
    they are compile-time constants instead of traced ops.
    """
    c1 = C_l[:, l, :]                  # (m, i)
    c2 = C_lp[:, lp, :]                # (mp, ip)
    i = np.arange(2*l + 1)[:, None]
    ip = np.arange(2*lp + 1)[None, :]
    sumterm = jnp.einsum('ai,bj,ij->ab', c1, c2, Q[i + ip])

    if term1 is None:
        term1 = beta_norm(alpha, beta)
    mm = np.arange(-l, l + 1)[:, None]
    mmp = np.arange(-lp, lp + 1)[None, :]
    phase = np.exp(1j * np.pi / 2 * (mm + mmp))
    return term1 * ebar[l] * ebarp[lp] * phase * sumterm


# The lambda step plus the U back-transform is a constant linear map on each
# (l, lp) block: block = real(conj(U_l).T @ Pllplambdammp(Pphi) @ conj(U_lp)).
# I contract it once per (l, lp) and cache it.
_lambda_kernel_cache = {}


def lambda_kernel(l, lp, lmax):
    key = (l, lp, lmax)
    if key not in _lambda_kernel_cache:
        with jax.ensure_compile_time_eval():
            U_all, C_all = set_Uylm(lmax), set_clmmpi_tensor(lmax)
            Ul, Ulp = np.asarray(_U_block(U_all, l)), np.asarray(_U_block(U_all, lp))
            C_l, C_lp = np.asarray(_C_block(C_all, l)), np.asarray(_C_block(C_all, lp))
            Q = np.asarray(Qllambda(l, lp))
            i = np.arange(2*l + 1)[:, None]
            ip = np.arange(2*lp + 1)[None, :]
            T = np.einsum('aui,bvj,ij->abuv', C_l, C_lp, Q[i + ip])
            K = np.einsum('xa,abuv,by->xyuv', np.conj(Ul).T, T, np.conj(Ulp))
        _lambda_kernel_cache[key] = K
    return jnp.asarray(_lambda_kernel_cache[key])


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


# def compute_radius_moment(lmax, r, c, smoothing=0.075, spts=1000, eps=1e-9, sfac=300):

#     theta, Bp, idx = _spot_basis(lmax, spts, eps, smoothing)
#     z = sfac * (theta - r)
#     b = 1.0 / (1.0 + jnp.exp(-z)) - 1.0
#     yl = Bp @ b
#     N = (lmax + 1) ** 2
#     y = jnp.zeros(N).at[idx].set(yl)
#     return Ylm.from_dense(y, normalize=False)


def compute_radius_moment(lmax, r, c, smoothing=0.075, spts=1000, eps=1e-9, sfac=300):

    theta, Bp, idx = _spot_basis(lmax, spts, eps, smoothing)
    z = sfac * (theta - r)
    b = 1.0 / (1.0 + jnp.exp(-z)) - 1.0
    yl = Bp @ b
    N = (lmax + 1) ** 2
    return jnp.zeros(N).at[idx].set(yl)

@partial(jax.jit, static_argnames="lmax")
def compute_first_moment(alpha, beta, r, c, lmax):
    """
    Compute E[y]
    """
    er = compute_radius_moment(lmax, r, c)

    ephi = []
    elambda = []
    U_all = set_Uylm(lmax)

    for l in range(0, lmax + 1):
        u = _U_block(U_all, l)

        q_phi_l = qlphi(alpha, beta, l)
        q_lambda_l = qllambda(l)

        erl = er[l**2:(l + 1)**2]
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

    U_all = set_Uylm(lmax)
    Us = [_U_block(U_all, ell) for ell in range(lmax + 1)]
    C_all = set_clmmpi_tensor(lmax)
    Cs = [_C_block(C_all, ell) for ell in range(lmax + 1)]

    # Precompute each degree's transformed radius coefficients once.
    erbar = []
    for ell in range(lmax + 1):
        er_l = er[ell**2:(ell + 1)**2]
        erbar.append(Us[ell] @ er_l)

    # Q_phi depends only on L = l + lp: build each one once, not once per block!!!
    Q_phi = [Qllpphi(alpha, beta, L, 0) for L in range(2 * lmax + 1)]
    term1 = beta_norm(alpha, beta)

    blocks = [[None] * (lmax + 1) for _ in range(lmax + 1)]

    for l in range(lmax + 1):

        Ul = Us[l]
        erbarl = erbar[l]

        # The second moment is symmetric, so only compute lp >= l.
        for lp in range(l, lmax + 1):

            Ulp = Us[lp]
            erbarlp = erbar[lp]

            Q_phi_llp = Q_phi[l + lp]

            Pphi = Pllpphimmp(l, lp, alpha, beta, erbarl, erbarlp, Q_phi_llp, Cs[l], Cs[lp], term1=term1)

            Ebarphi_block = Pphi

            # = real(conj(Ul).T @ Pllplambdammp(...) @ conj(Ulp)), with the
            # constant part precontracted.
            block = jnp.real(jnp.einsum('xyuv,uv->xy', lambda_kernel(l, lp, lmax), Ebarphi_block))
            blocks[l][lp] = block
            blocks[lp][l] = block.T

    # One assembly at the end instead of (lmax+1)**2 full-matrix .at[].set updates.
    return jnp.block(blocks)


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


@partial(jax.jit, static_argnames="lmax")
def compute_mean_and_covariance(alpha, beta, r, c, n, lmax):
    """Same as (compute_mean, compute_covariance), sharing one first moment."""
    first = compute_first_moment(alpha, beta, r, c, lmax)
    second = compute_second_moment(alpha, beta, r, c, lmax)
    mean = n * first * jnp.pi * c
    cov = n * jnp.real(second - jnp.outer(first, first)) * (jnp.pi * c) ** 2
    return mean, cov

import jax.scipy.special as jss
from jaxoplanet.starry.core.rotation import dot_rotation_matrix

def qlI(l, imax=None):
    if imax is None:
        imax = 2 * l
    i = jnp.arange(imax + 1)
    F = jss.hyp2f1(1.0, -i / 2.0, 2.0 + (2 * l - i) / 2.0, -1.0)
    return (-1.0) ** i / ((2 * l - i) / 2.0 + 1.0) * F


def QllI(l, lp):
    L = l + lp
    return qlI(L, 2 * L)


def Rx(lmax, angle, y):
    return dot_rotation_matrix(lmax, 1.0, 0.0, 0.0, angle)(y)


def plIm(l, ebar, q):
    C = clmmpi_tensor(l)
    m = jnp.arange(-l, l + 1)[:, None]
    mu = jnp.arange(-l, l + 1)[None, :]
    S = jnp.einsum('aui,i->au', C, q)
    return jnp.einsum('u,au,au->a', ebar, jnp.exp(1j * jnp.pi / 2 * (m - mu)), S)


def PllImmp(l, lp, Ebar_block, Q, C_l, C_lp):
    i = jnp.arange(2 * l + 1)[:, None]
    ip = jnp.arange(2 * lp + 1)[None, :]
    T = jnp.einsum('aui,bvj,ij->aubv', C_l, C_lp, Q[i + ip])
    m = jnp.arange(-l, l + 1)[:, None, None, None]
    mu = jnp.arange(-l, l + 1)[None, :, None, None]
    mp = jnp.arange(-lp, lp + 1)[None, None, :, None]
    mup = jnp.arange(-lp, lp + 1)[None, None, None, :]
    phase = jnp.exp(1j * jnp.pi / 2 * (m - mu + mp - mup))
    return jnp.einsum('uv,aubv,aubv->ab', Ebar_block, phase, T)

@partial(jax.jit, static_argnames="lmax")
def compute_first_moment_marginalized(ey, lmax):
    N = (lmax + 1) ** 2
    ey0 = jnp.real(Rx(lmax, jnp.pi / 2, ey))

    ey00 = []
    U_all = set_Uylm(lmax)
    for l in range(lmax + 1):
        u = _U_block(U_all, l)
        ebar = u @ ey0[l**2:l**2 + 2*l + 1]
        pI = plIm(l, ebar, qlI(l))
        ey00.append(jnp.real(jnp.conj(u.T) @ pI))

    return jnp.concatenate(ey00)

@partial(jax.jit, static_argnames="lmax")
def compute_second_moment_marginalized(Ey, lmax):
    N = (lmax + 1) ** 2
    Rxhalf = Rx(lmax, jnp.pi / 2, jnp.eye(N))
    Ey0 = Rxhalf @ Ey @ Rxhalf.T

    U_all = set_Uylm(lmax)
    Us = [_U_block(U_all, ell) for ell in range(lmax + 1)]
    C_all = set_clmmpi_tensor(lmax)
    Cs = [_C_block(C_all, ell) for ell in range(lmax + 1)]

    Ey00 = jnp.zeros((N, N), dtype=jnp.complex128)
    for l in range(lmax + 1):
        for lp in range(lmax + 1):
            blk = Ey0[l**2:l**2 + 2*l + 1, lp**2:lp**2 + 2*lp + 1]
            Ebar = Us[l] @ blk @ jnp.conj(Us[lp]).T
            PllI = PllImmp(l, lp, Ebar, QllI(l, lp), Cs[l], Cs[lp])
            block = jnp.conj(Us[l]).T @ PllI @ jnp.conj(Us[lp])
            Ey00 = Ey00.at[l**2:l**2 + 2*l + 1, lp**2:lp**2 + 2*lp + 1].set(block)

    return jnp.real(Ey00)
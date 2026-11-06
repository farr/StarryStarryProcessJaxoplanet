import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as jlx

from jaxoplanet.starry.ylm import Ylm
from functools import partial

import functools

from utils import hyp2f1_sequence, gamma_sequence1, gamma_sequence2, hyp2f1_1_b_c_m1, Fam, Fcm


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
    For degree l, retrieve C_l with _C_block(C_all, l). 
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
    computed once rather than per block. 
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


# The lambda step plus the U is a constant linear map on each
# (l, lp) block: block = real(conj(U_l).T @ Pllplambdammp(Pphi) @ conj(U_lp)).
# I do it once per (l, lp) and cache it.
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


# Manual cache of the spot basis, keyed by (lmax, spts, eps, smoothing).  It does
# not depend on (r, c), and XLA does *not* constant-fold it inside jit (the
# linalg.solve is a LAPACK custom call), so build it once in numpy.
_spot_basis_cache = {}


def _spot_basis(lmax, spts, eps, smoothing):
    key = (int(lmax), int(spts), float(eps), float(smoothing))
    if key not in _spot_basis_cache:
        theta = np.linspace(0.0, np.pi, spts)
        cost = np.cos(theta)
        l = np.arange(lmax + 1)

        # Legendre P_0..P_lmax via the Bonnet recurrence (numpy twin of _legendre_basis).
        P = np.empty((lmax + 1, spts))
        P[0] = 1.0
        if lmax > 0:
            P[1] = cost
        for ell in range(1, lmax):
            P[ell + 1] = ((2 * ell + 1) * cost * P[ell] - ell * P[ell - 1]) / (ell + 1)

        B = (np.sqrt(2.0 * l + 1.0)[:, None] * P).T
        Bp = np.linalg.solve(B.T @ B + eps * np.eye(lmax + 1), B.T)
        idx = l * (l + 1)
        Sm = np.exp(-0.5 * idx * smoothing ** 2)
        _spot_basis_cache[key] = (theta, Sm[:, None] * Bp, idx)
    return _spot_basis_cache[key]


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

def qlphi_padded(alpha, beta, l, lpad):
    """q^l_phi = gamma_sequence1 * hyp2f1_sequence, zero-padded to length 2*lpad+1.

    Unlike `qlphi`, `l` may be traced, so this can be vmapped over all degrees
    at once (one pair of scans instead of one pair per degree).  Recurrence
    steps past i = 2l are clamped to i = 2l, so the padding never evaluates
    the recurrence at a pole (keeps values *and* gradients finite); those
    entries are then masked to zero.  Entries i <= 2l equal `qlphi(alpha, beta, l)`.
    """
    imax = 2 * lpad
    i_all = jnp.arange(0, imax + 1, 2)
    i_eff = jnp.minimum(i_all, 2 * l)

    # gamma_sequence1
    g0 = jnp.exp(jax.lax.lgamma(alpha) + jax.lax.lgamma(l + beta + 1.0)
                 - jax.lax.lgamma(l + alpha + beta + 1.0))

    def gstep(state, i):
        nxt = state * (l + alpha + beta - i / 2) / (l + beta - i / 2)
        return nxt, nxt

    _, gam = jlx.scan(gstep, g0, i_eff)

    # hyp2f1_sequence
    F0 = (hyp2f1_1_b_c_m1(alpha, l + alpha + beta),
          hyp2f1_1_b_c_m1(alpha, l + alpha + beta + 1.0))

    def fstep(state, i):
        F_, Fcp_ = state
        Fam_ = Fam(F_, Fcp_, l, alpha, beta, i)
        Fcm_ = Fcm(Fam_, F_, l, alpha, beta, i)
        return (Fcm_, Fam_), Fam_

    _, hyp = jlx.scan(fstep, F0, i_eff)

    evens = jnp.where(i_all <= 2 * l, gam * hyp, 0.0)
    return jnp.zeros(imax + 1).at[0::2].set(evens)


def qlphi_all(alpha, beta, Lmax):
    """Stack of qlphi_padded for L = 0..Lmax, shape (Lmax+1, 2*Lmax+1)."""
    return jax.vmap(lambda L: qlphi_padded(alpha, beta, L, Lmax))(jnp.arange(Lmax + 1))


# Everything in E[y] that does not depend on (alpha, beta, r, c), cached per lmax.
_first_moment_cache = {}


def _first_moment_constants(lmax):
    """Precompute the constant pieces of E[y] = beta_norm * M @ g, where

        g_row = erbar_{l,0} * sum_i C_l[m, 0, i] q^phi_l[i]       (real)
        M     = Re(U^H Lambda U) @ Re(U^H diag(i^m))               (N x N, real)

    and Lambda is the block-diagonal sum_i C_l[a, m, i] q^lambda_l[i].  The
    Re(.) factorization is exact because g and E_phi are real.
    """
    lmax = int(lmax)
    if lmax not in _first_moment_cache:
        N = (lmax + 1) ** 2
        K = 2 * lmax + 1
        U = set_Uylm(lmax)
        C_all = set_clmmpi_tensor(lmax)

        ell_row = np.zeros(N, dtype=np.int32)   # degree of each flat row
        m_row = np.zeros(N)                     # order m of each flat row
        C0 = np.zeros((N, K))                   # C_l[m, mp=0, :] for each flat row
        Lam = np.zeros((N, N))
        for l in range(lmax + 1):
            s, e = l**2, (l + 1) ** 2
            ell_row[s:e] = l
            m_row[s:e] = np.arange(-l, l + 1)
            C0[s:e] = C_all[s:e, s + l, :]
            with jax.ensure_compile_time_eval():
                ql = np.asarray(qllambda(l))
            Lam[s:e, s:e] = np.einsum("ami,i->am", C_all[s:e, s:e, :2 * l + 1], ql)

        UH = U.conj().T
        B = np.real(UH @ np.diag(np.exp(1j * np.pi * m_row / 2)))
        A = np.real(UH @ Lam @ U)

        # m = 0 rows of U pick erbar_{l,0} out of er; they are real.
        U0 = U[[l * l + l for l in range(lmax + 1)], :]
        assert np.allclose(U0.imag, 0.0)

        _first_moment_cache[lmax] = (A @ B, C0, ell_row, np.real(U0))
    return _first_moment_cache[lmax]


@partial(jax.jit, static_argnames="lmax")
def compute_first_moment(alpha, beta, r, c, lmax):
    """
    Compute E[y].

    Vectorized over l: the per-degree phi and lambda steps are folded into one
    constant matrix M (see _first_moment_constants), and q^phi_l is computed
    for all l at once.  Equivalent to, for each l,
        ephi_l    = Re(U_l^H plphim(l, U_l er_l, alpha, beta, qlphi(alpha, beta, l)))
        elambda_l = Re(U_l^H pllambdam(l, U_l ephi_l, qllambda(l)))
    """
    M, C0, ell_row, U0 = _first_moment_constants(lmax)
    er = compute_radius_moment(lmax, r, c)

    Q = qlphi_all(alpha, beta, lmax)                         # (lmax+1, 2lmax+1)
    erbar0 = U0 @ er                                         # (lmax+1,)
    g = erbar0[ell_row] * jnp.sum(C0 * Q[ell_row], axis=1)   # (N,)
    return beta_norm(alpha, beta) * (M @ g)

# Everything in E[y y^T] that does not depend on (alpha, beta, r, c), cached per lmax.
_second_moment_cache = {}


# Eigenbasis of the Wigner d-matrices, shared by the lambda step and the
# inclination marginalization, cached per lmax.
_wigner_eigenbasis_cache = {}


def wigner_d(l, theta, C_all):
    """D^l(theta) = 2^l sum_i C_l[:, :, i] cos^i(theta/2) sin^(2l-i)(theta/2), in numpy.

    This is the (real, orthogonal) Wigner d-matrix in the complex-Ylm basis used
    throughout; C_all is set_clmmpi_tensor(lmax) for some lmax >= l.
    """
    s, e = l**2, (l + 1) ** 2
    i = np.arange(2 * l + 1)
    return 2.0**l * np.einsum("abi,i->ab", C_all[s:e, s:e, :2 * l + 1],
                              np.cos(theta / 2) ** i * np.sin(theta / 2) ** (2 * l - i))


def _wigner_eigenbasis(lmax):
    """Block-diagonal unitary V and integer orders mu with D^l(theta) = V diag(e^{i mu theta}) V^H.

    Because D(theta) is a one-parameter rotation group, any average of
    D X D^T over a distribution of theta is, in this basis, a Hadamard product:
        E[D X D^T] = V [F * (V^H X V)] V^H,   F_pq = E[exp(i (mu_p - mu_q) theta)].
    """
    lmax = int(lmax)
    if lmax not in _wigner_eigenbasis_cache:
        N = (lmax + 1) ** 2
        C_all = set_clmmpi_tensor(lmax)
        V = np.zeros((N, N), dtype=np.complex128)
        mu_row = np.zeros(N)
        theta1 = 0.1   # any angle with |mu * theta1| < pi for all |mu| <= lmax
        for l in range(lmax + 1):
            s, e = l**2, (l + 1) ** 2
            D = wigner_d(l, theta1, C_all)
            w, v = np.linalg.eig(D)
            mu = np.rint(np.angle(w) / theta1)
            assert np.allclose(np.abs(w), 1.0) and len(set(mu)) == 2 * l + 1
            v, _ = np.linalg.qr(v)   # distinct eigenvalues of a normal matrix: unitary up to phases
            assert np.allclose(v @ np.diag(w) @ v.conj().T, D)
            V[s:e, s:e] = v
            mu_row[s:e] = mu
        _wigner_eigenbasis_cache[lmax] = (V, mu_row)
    return _wigner_eigenbasis_cache[lmax]


def _second_moment_constants(lmax):
    """Precompute the constant pieces of E[y y^T].

    phi step:     S[p, q] = sum_ij C_lp[m_p, 0, i] C_lq[m_q, 0, j] Qphi_{lp+lq}[i + j]
                  Pphi    = beta_norm * (e e^T) * diag(i^m) S diag(i^m),   e_p = erbar_{lp,0}
    lambda step:  T = E_theta[D(theta) Pphi D(theta)^T],   E[y y^T] = Re(U^H T conj(U))

    D^l(theta) = 2^l sum_i C_l[:, :, i] cos^i(theta/2) sin^(2l-i)(theta/2) is the Wigner
    d-matrix, and Qlambda is exactly the average of 2^L cos^k sin^(2L-k) over a uniform
    angle on the full circle.  So the lambda step averages a one-parameter rotation group
    over a full period: in the eigenbasis D = V diag(e^{i mu theta}) V^H it just keeps
    the entries with mu_p == mu_q,
        T = V [mask * (V^H Pphi V)] V^H.
    This replaces the per-(l, lp) lambda_kernel tensors (O(sum (2l+1)^2)^2 storage).
    """
    lmax = int(lmax)
    if lmax not in _second_moment_cache:
        N = (lmax + 1) ** 2
        K = 2 * lmax + 1
        U = set_Uylm(lmax)
        C_all = set_clmmpi_tensor(lmax)

        ell_row = np.zeros(N, dtype=np.int32)       # degree of each flat row
        m_row = np.zeros(N)                         # order m of each flat row
        pad_idx = np.zeros(N, dtype=np.int32)       # flat row -> index in the (lmax+1)*K padded layout
        C0pad = np.zeros((lmax + 1, K, K))          # C_l[m, mp=0, :], with m padded to K
        V, mu_row = _wigner_eigenbasis(lmax)

        for l in range(lmax + 1):
            s, e = l**2, (l + 1) ** 2
            n = 2 * l + 1
            ell_row[s:e] = l
            m_row[s:e] = np.arange(-l, l + 1)
            pad_idx[s:e] = l * K + np.arange(n)
            C0pad[l, :n, :] = C_all[s:e, s + l, :]

        phase = np.exp(1j * np.pi * m_row / 2)
        VH = V.conj().T

        # Q-index for the Hankel-like gather H[P, Q, i, j] = Qphi_{P+Q}[i + j].
        P = np.arange(lmax + 1)
        ii = np.arange(K)
        H_idx = ((P[:, None, None, None] + P[None, :, None, None]) * (4 * lmax + 1)
                 + ii[None, None, :, None] + ii[None, None, None, :])

        # m = 0 rows of U pick erbar_{l,0} out of er; they are real.
        U0 = U[[l * l + l for l in range(lmax + 1)], :]
        assert np.allclose(U0.imag, 0.0)

        _second_moment_cache[lmax] = dict(
            ell_row=ell_row, pad_idx=pad_idx, C0pad=C0pad, H_idx=H_idx, U0=np.real(U0),
            Gl=VH * phase[None, :],                  # V^H diag(i^m)
            Gr=phase[:, None] * V,                   # diag(i^m) V
            A=U.conj().T @ V,                        # U^H V
            B=VH @ U.conj(),                         # V^H conj(U)
            mask=(mu_row[:, None] == mu_row[None, :]).astype(float),
        )
    return _second_moment_cache[lmax]


@partial(jax.jit, static_argnames="lmax")
def compute_second_moment(alpha, beta, r, c, lmax):
    """
    Compute E[y y^T].

    Fully vectorized: no loop over (l, lp).  See _second_moment_constants for the
    derivation; equivalent to the per-block
        block_{l,lp} = Re(einsum('xyuv,uv->xy', lambda_kernel(l, lp, lmax),
                                 Pllpphimmp(l, lp, ...)))
    """
    k = _second_moment_constants(lmax)
    er = compute_radius_moment(lmax, r, c)
    erbar0 = k["U0"] @ er                                       # (lmax+1,)

    # phi step.
    Q = qlphi_all(alpha, beta, 2 * lmax)                        # (2lmax+1, 4lmax+1)
    H = Q.reshape(-1)[k["H_idx"]]                               # (lmax+1, lmax+1, K, K)
    Spad = jnp.einsum("Pai,PQij,Qbj->PaQb", k["C0pad"], H, k["C0pad"], optimize="optimal")
    n = Spad.shape[0] * Spad.shape[1]
    S = Spad.reshape(n, n)[k["pad_idx"]][:, k["pad_idx"]]       # (N, N), real
    e = erbar0[k["ell_row"]]
    S = beta_norm(alpha, beta) * e[:, None] * e[None, :] * S

    # lambda step: keep equal-mu entries in the eigenbasis of the lambda rotation.
    Z = k["mask"] * (k["Gl"] @ S @ k["Gr"])
    return jnp.real(k["A"] @ Z @ k["B"])


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

# Inclination marginalization.  The inclination moments are
#     qlI(L)[k] = (-1)^k / (L - k/2 + 1) 2F1(1, -k/2; L - k/2 + 2; -1)
#               = E_inc[2^L cos^k(theta/2) sin^(2L-k)(theta/2)],   theta = -inc,
# with inc isotropic on [0, pi/2] (density sin(inc)).  They do not depend on any
# parameters, so the marginalized moments are constant linear maps of ey / Ey,
# precomputed below.  Gauss-Legendre in inc converges spectrally (integrand is
# entire); NQUAD_INC nodes give machine precision well past lmax = 15.
NQUAD_INC = 256


def _inclination_quadrature(n=NQUAD_INC):
    """Nodes theta = -inc and weights (summing to 1) for inc isotropic on [0, pi/2]."""
    x, w = np.polynomial.legendre.leggauss(n)
    inc = np.pi / 4 * (x + 1.0)
    return -inc, w * np.pi / 4 * np.sin(inc)


def qlI(l, imax=None):
    """Inclination moments, computed in numpy by quadrature (no 2F1 at z = -1).

    Entries beyond 2l (if imax > 2l) follow the same formula as before.
    """
    if imax is None:
        imax = 2 * l
    theta, w = _inclination_quadrature()
    k = np.arange(imax + 1)[:, None]
    if imax <= 2 * l:
        q = (w * 2.0**l * np.cos(theta / 2) ** k * np.sin(theta / 2) ** (2 * l - k)).sum(1)
    else:   # fall back to the closed form (scipy is accurate here) for the unused tail
        import scipy.special as ss
        kk = k[:, 0]
        q = (-1.0) ** kk / ((2 * l - kk) / 2.0 + 1.0) * ss.hyp2f1(1.0, -kk / 2.0, 2.0 + (2 * l - kk) / 2.0, -1.0)
    return jnp.asarray(q)


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

_marginalized_cache = {}


def _marginalized_constants(lmax):
    """Constant maps for the inclination-marginalized moments.

    Per degree, the old code computed (Phi = diag(i^m), D = Wigner d-matrix)
        first:  ey00 = Re(U^H Phi E[D] Phi^* U ey0),                 ey0 = Re(Rx(pi/2) ey)
        second: Ey00 = Re(U^H Phi E[D X D^T] Phi conj(U)),  X = Phi^* U Ey0 U^T Phi^*,
                Ey0 = R Ey R^T with R = Rx(pi/2).
    i.e. with the real map L(theta) = Re(U^H Phi D(theta) Phi^* U) R, first = E[L] ey and
    second = E[L Ey L^T], so the marginal covariance is PSD.  (The previous loop
    version used U Ey0 U^H, i.e. conj(U)^T instead of U^T, and R^T Ey R instead of
    R Ey R^T; both made it inconsistent with the first moment.)
    With D = V diag(e^{i mu theta}) V^H (see _wigner_eigenbasis):
        E[D]         = V diag(f) V^H,           f_p  = E[exp(i mu_p theta)]
        E[D X D^T]   = V [F * (V^H X V)] V^H,   F_pq = E[exp(i (mu_p - mu_q) theta)]
    """
    lmax = int(lmax)
    if lmax not in _marginalized_cache:
        N = (lmax + 1) ** 2
        U = set_Uylm(lmax)
        V, mu_row = _wigner_eigenbasis(lmax)
        VH = V.conj().T
        m_row = np.concatenate([np.arange(-l, l + 1) for l in range(lmax + 1)])
        phase = np.exp(1j * np.pi * m_row / 2)

        theta, w = _inclination_quadrature()
        f = (w * np.exp(1j * mu_row[:, None] * theta)).sum(1)
        dmu = np.arange(-2 * lmax, 2 * lmax + 1)
        Fd = (w * np.exp(1j * dmu[:, None] * theta)).sum(1)
        F = Fd[(mu_row[:, None] - mu_row[None, :]).astype(int) + 2 * lmax]

        # Rx(pi/2) as an explicit matrix R with R @ y == Rx(lmax, pi/2, y).
        # NB: dot_rotation_matrix acts on the *rows* of a matrix argument, so
        # Rx(lmax, pi/2, eye(N)) is R^T, not R.  (The previous second moment used
        # Rx(eye) @ Ey @ Rx(eye).T = R^T Ey R, inconsistent with the first moment's
        # R @ ey, which made the marginal covariance indefinite for lmax >= 3.)
        with jax.ensure_compile_time_eval():
            R = np.asarray(Rx(lmax, jnp.pi / 2, jnp.eye(N))).T
        assert np.allclose(np.imag(R), 0.0)
        R = np.real(R)

        UH = U.conj().T
        M1 = np.real(UH @ (phase[:, None] * (V @ (f[:, None] * VH))) @ (phase.conj()[:, None] * U)) @ R

        _marginalized_cache[lmax] = dict(
            M1=M1,
            Gl=(VH * phase.conj()[None, :]) @ U @ R,          # V^H Phi^* U R
            Gr=R.T @ U.T @ (phase.conj()[:, None] * V),        # R^T U^T Phi^* V
            A=UH @ (phase[:, None] * V),                       # U^H Phi V
            B=(VH * phase[None, :]) @ U.conj(),                # V^H Phi conj(U)
            F=F,
        )
    return _marginalized_cache[lmax]


@partial(jax.jit, static_argnames="lmax")
def compute_first_moment_marginalized(ey, lmax):
    """Inclination-marginalized E[y]: a constant (N, N) matrix applied to ey."""
    return _marginalized_constants(lmax)["M1"] @ ey


@partial(jax.jit, static_argnames="lmax")
def compute_second_moment_marginalized(Ey, lmax):
    """Inclination-marginalized E[y y^T]: Re(A [F * (Gl Ey Gr)] B), constants precomputed."""
    k = _marginalized_constants(lmax)
    return jnp.real(k["A"] @ (k["F"] * (k["Gl"] @ Ey @ k["Gr"])) @ k["B"])

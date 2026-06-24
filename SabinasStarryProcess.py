import numpy as np
import jax
import jax.numpy as jnp

from jaxoplanet.starry import ylm
from jaxoplanet.starry.ylm import Ylm

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

def clmmpi(l, m, mp, i):
    """The Wigner-d matrix coefficients c^l_{m, m', i}."""
    if (m-mp-i) % 2 != 0:
        return 0.0
    elif i < max([0, m+mp, -m-mp]):
        return 0.0
    elif i > min([2*l, 2*l+m-mp,2*l-m+mp]):
        return 0.0
    else:
        log_num = 0.5 * (log_factorial(l - m) + log_factorial(l + m) + 
                      log_factorial(l - mp) + log_factorial(l + mp))
        log_denom = l * jnp.log(2) + log_factorial((i - m - mp) / 2) + \
            log_factorial((i + m + mp) / 2) + log_factorial((2*l - i - m + mp) / 2) + \
            log_factorial((2*l - i + m - mp) / 2)
        if (2*l - m + mp - i)/2 % 2 == 0:
            return jnp.exp(log_num - log_denom)
        else:
            return -jnp.exp(log_num - log_denom)

def qlphi(alpha, beta, l, i):
    if (i % 2) != 0:
        return 0.0
    else:
        term1 = jax.lax.lgamma(alpha) + jax.lax.lgamma(l + beta - i/2) \
            - jax.lax.lgamma(l + alpha + beta - i/2)
        term2 = jax.scipy.special.hyp2f1(-i / 2, alpha, l + alpha + beta - i / 2, -1)

        return jnp.exp(term1) * term2

def Qllpphi(l, lp, i, ip, alpha, beta):
    if (i+ip) % 2 != 0:
        return 0.0
    else:
        logterm1_num = jax.lax.lgamma(alpha) + jax.lax.lgamma(l + lp + beta - (i + ip) / 2)
        logterm1_denom = jax.lax.lgamma(l + lp + alpha + beta - (i + ip) / 2)
        term1 = jnp.exp(logterm1_num - logterm1_denom)
        term2 = jax.scipy.special.hyp2f1(-(i + ip) / 2, alpha, l + lp + alpha + beta - (i + ip) / 2, -1)

        return term1 * term2

def qllambda(l, i):
    if (i % 2) != 0:
        return 0.0
    else:
        log_term1 = l*jnp.log(2) + jax.lax.lgamma((1 + i) / 2) + jax.lax.lgamma(l + (1 - i) / 2)
        log_term2 = jnp.log(jnp.pi) + jax.lax.lgamma(float(l + 1))

        return jnp.exp(log_term1 - log_term2)


def Qllambda(l, lp, i, ip):
    if (i+ip) % 2 != 0:
        return 0.0
    else:
        return qllambda(l+lp, i+ip)

def ebar_func(er, l):
    er_array = jnp.array([er[l,m] for m in range(-l,l+1)])
    ebar = U(l) @ er_array

    return ebar

def plphim(l, m, ebar, alpha, beta, mu=0):
    term1 = jnp.exp(jax.lax.lgamma(alpha + beta) - (jax.lax.lgamma(alpha) + jax.lax.lgamma(beta)))
    sumterm = sum([clmmpi(l, m, 0, i) * qlphi(alpha, beta, l, i) for i in range(0, 2*l+1)])
    term2 = ebar[l] * jnp.exp(1j * jnp.pi * m / 2) * sumterm 

    return term1 * term2

def pllambdam(l, m, ebarphi):
    sumterm = sum([ebarphi[l+mp] * clmmpi(l, m, mp, i) * qllambda(l, i) for mp in range(-l, l+1) for i in range(0, 2*l+1)])

    return sumterm 

def Pllpphimmp(l, lp, m, mp, alpha, beta, ebar, ebarp):
    term1 = jnp.exp(jax.lax.lgamma(alpha + beta) - (jax.lax.lgamma(alpha) + jax.lax.lgamma(beta)))
    term2 = ebar[l] * ebarp[lp] * jnp.exp(1j * jnp.pi / 2 * (m + mp)) # ebar[l] == ebar(l=l, m=0), ebarp[lp] == ebar(l=lp,m=0)
    sumterm = sum([(clmmpi(l, m, 0, i) * clmmpi(lp, mp, 0, ip) * Qllpphi(l, lp, i, ip, alpha, beta)) for i in range(0, 2*l+1) for ip in range(0, 2*lp+1)])

    return term1 * term2 * sumterm

def Pllplambdammp(l, lp, m, mp, Ebarphi_block):
    total = 0.0

    for mu in range(-l, l + 1):
        for mup in range(-lp, lp + 1):

            E_m_mp = Ebarphi_block[l + mu, lp + mup]

            for i in range(2*l + 1):
                c1 = clmmpi(l, m, mu, i)

                for ip in range(2*lp + 1):
                    c2 = clmmpi(lp, mp, mup, ip)
                    
                    total += c1 * c2 * Qllambda(l, lp, i, ip) * E_m_mp

    return total

# def compute_radius_moment(lmax, r, c):
#     return ylm.ylm_spot(lmax)(r=r, contrast=c)

# def compute_radius_moment(lmax, r):
#     unit = ylm.ylm_spot(lmax)(contrast=1.0, r=r)
#     flat = ylm.ylm_spot(lmax)(contrast=0.0, r=r)
#     y_sr = flat.todense() - unit.todense()
#     return Ylm.from_dense(y_sr, normalize=False)

def compute_radius_moment(lmax, r, c, smoothing=0.075, spts=1000, eps=1e-9, sfac=300):
    
    from scipy.special import legendre as _Pl
 
    theta = np.linspace(0, jnp.pi, spts)
    cost = np.cos(theta)
    B = np.hstack(
        [np.sqrt(2 * l + 1) * _Pl(l)(cost).reshape(-1, 1) for l in range(lmax + 1)]
    )
    Bp = np.linalg.solve(B.T @ B + eps * np.eye(lmax + 1), B.T)
    l = np.arange(lmax + 1)
    idx = l * (l + 1)
    Sm = np.exp(-0.5 * idx * smoothing ** 2)
    Bp = Sm[:, None] * Bp
    z = sfac * (theta - r)
    b = 1.0 / (1.0 + np.exp(-z)) - 1.0
    N = (lmax + 1) ** 2
    y = np.zeros(N)
    y[idx] = Bp @ b
    return Ylm.from_dense(jnp.asarray(y), normalize=False)

def compute_first_moment(alpha, beta, r, c, lmax):
    """
    Compute E[y]
    """
    N = (lmax + 1) ** 2
    er = compute_radius_moment(lmax, r, c)

    ephi = []
    elambda = []

    for l in range(0, lmax+1):
        u = U(l)
        erl = jnp.array([er[l,m] for m in range(-l,l+1)])
        erlbar = u @ erl
        p = jnp.array([plphim(l, m, erlbar, alpha, beta) for m in range(-l, l+1)])

        ephi_ = jnp.real(jnp.conj(u.T) @ p)
        ephi.append(ephi_)
        ebarphi = u @ ephi_

        plambda = jnp.array([pllambdam(l, m, ebarphi) for m in range(-l, l+1)])
        elambda.append(jnp.real(jnp.conj(u.T) @ plambda))

    elambda = jnp.concatenate(elambda)
    ephi = jnp.concatenate(ephi)

    return elambda

def compute_second_moment(alpha, beta, r, c, lmax):
    N = (lmax + 1) ** 2
    er = compute_radius_moment(lmax, r, c)

    Ephi = np.zeros((N, N), dtype=np.complex128)
    Elambda = np.zeros((N, N), dtype=np.complex128)

    for l in range(lmax+1):

        Ul = U(l)
        erbarl = ebar_func(er, l)

        for lp in range(lmax+1): 

            Ulp = U(lp)
            erbarlp = ebar_func(er, lp)

            Pphi = np.zeros((2*l+1, 2*lp+1), dtype=np.complex128)

            for m in range(-l, l+1):
                for mp in range(-lp, lp+1):
                    Pphi[l+m, lp+mp] = Pllpphimmp(l, lp, m, mp, alpha, beta, erbarl, erbarlp)

            Ephi_block = jnp.conj(Ul.T) @ Pphi @ jnp.conj(Ulp)
            Ephi[l**2:l**2+2*l+1, lp**2:lp**2+2*lp+1] = Ephi_block

            Ebarphi_block = Pphi

            pllp = np.zeros((2*l+1, 2*lp+1), dtype=np.complex128)

            for m in range(-l, l+1):
                for mp in range(-lp, lp+1):
                    pllp[l+m, lp+mp] = Pllplambdammp(l, lp, m, mp, Ebarphi_block)

            Elambda[l**2:l**2+2*l+1, lp**2:lp**2+2*lp+1] = jnp.conj(Ul).T @ pllp @ jnp.conj(Ulp)

    return jnp.real(Elambda)

def compute_covariance_one(alpha, beta, r, c, lmax):
 
    first = compute_first_moment(alpha, beta, r, c, lmax)
    second = compute_second_moment(alpha, beta, r, c, lmax)
    # second_eigen = jnp.dot(second, second.T)
    covariance = jnp.real(second - jnp.outer(first, first)) 

    return covariance * (jnp.pi * c) ** 2

def compute_mean(alpha, beta, r, c, n, lmax):
    return n * compute_first_moment(alpha, beta, r, c, lmax) * jnp.pi * c

def compute_covariance(alpha, beta, r, c, n, lmax):
    return n * compute_covariance_one(alpha, beta, r, c, lmax)


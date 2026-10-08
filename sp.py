from __future__ import annotations

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import scipy

from SabinasStarryProcess1 import compute_mean, compute_covariance
from design_matrix_jaxoplanet import design_matrix as _design_matrix

from jaxoplanet.core.limb_dark import light_curve as _limb_dark_light_curve
from jaxoplanet.starry.core.basis import A1, A2_inv, U
from jaxoplanet.starry.core.polynomials import Pijk
from jaxoplanet.starry.core.rotation import left_project
from jaxoplanet.starry.core.solution import rT, solution_vector
from jaxoplanet.starry.surface import Surface
from jaxoplanet.starry.ylm import Ylm

LOG_ALPHA_MAX = 10.0
LOG_BETA_MAX = 10.0
LOG_BETA_MIN = np.log(0.5)

def ab_to_alpha_beta(a, b):
    """Exact Starry Process mapping from dimensionless (a,b) to Beta(alpha,beta)."""
    alpha = jnp.exp(a * LOG_ALPHA_MAX)
    beta = jnp.exp(LOG_BETA_MIN + b * (LOG_BETA_MAX - LOG_BETA_MIN))
    return alpha, beta

def alpha_beta_to_ab(alpha, beta):
    """Exact inverse of ab_to_alpha_beta."""
    a = jnp.log(alpha) / LOG_ALPHA_MAX
    b = (jnp.log(beta) - LOG_BETA_MIN) / (LOG_BETA_MAX - LOG_BETA_MIN)
    return jnp.array([a, b])


def alpha_beta_to_latitude(alpha, beta):
    """Starry Process beta2gauss map, written in JAX.

    Returns (mu_deg, sigma_deg).
    """
    term = (4.0 * alpha**2 - 8.0 * alpha - 6.0 * beta + 4.0 * alpha * beta + beta**2 + 5.0)
    mu = 2.0 * jnp.arctan(jnp.sqrt(2.0 * alpha + beta - 2.0 - jnp.sqrt(term)))
    curvature = (1.0 - alpha + beta + (beta - 1.0) * jnp.cos(mu) + (alpha - 1.0) / jnp.cos(mu) ** 2)
    sigma = jnp.sin(mu) / jnp.sqrt(curvature)
    return jnp.array([mu, sigma]) * (180.0 / jnp.pi)

def ab_to_latitude(a, b):
    return alpha_beta_to_latitude(*ab_to_alpha_beta(a, b))


def gauss_to_ab(mu_deg, sigma_deg):
    mu = jnp.deg2rad(jnp.asarray(mu_deg))
    sigma2 = jnp.square(jnp.deg2rad(jnp.asarray(sigma_deg)))

    term = 1.0 / (16.0 * sigma2 * jnp.cos(0.5 * mu) ** 4)
    alpha = (2.0 + 4.0 * sigma2 + (3.0 + 8.0 * sigma2) * jnp.cos(mu) + 2.0 * jnp.cos(2.0 * mu) + jnp.cos(3.0 * mu)) * term
    beta = (jnp.cos(mu) + 2.0 * sigma2 * (3.0 + jnp.cos(2.0 * mu)) - jnp.cos(3.0 * mu)) * term
    return alpha_beta_to_ab(alpha, beta)


# ---------------------------------------------------------------------------
# Starry Process
# ---------------------------------------------------------------------------

class StarryProcess:

    def __init__(
        self,
        alpha,
        beta,
        contrast,
        r,
        nspots,
        lmax=5,
    ):
        self.alpha = alpha
        self.beta = beta
        self.contrast = contrast
        self.r = r
        self.nspots = nspots
        self.lmax = lmax

    # ------------------------------------------------------------------
    # Latitude parameter transformations
    # ------------------------------------------------------------------

    @staticmethod
    def ab_to_alpha_beta(a, b):
        return ab_to_alpha_beta(a, b)

    @staticmethod
    def alpha_beta_to_ab(alpha, beta):
        return alpha_beta_to_ab(alpha, beta)

    @staticmethod
    def alpha_beta_to_latitude(alpha, beta):
        return alpha_beta_to_latitude(alpha, beta)

    @staticmethod
    def ab_to_latitude(a, b):
        return ab_to_latitude(a, b)

    @staticmethod
    def gauss_to_ab(mu_deg, sigma_deg):
        return gauss_to_ab(mu_deg, sigma_deg)

    # ------------------------------------------------------------------
    # Moments
    # ------------------------------------------------------------------

    def moments(self):
        mean_ylm = compute_mean(self.alpha,self.beta,self.r,self.contrast,self.nspots,lmax=self.lmax)
        cov_ylm = compute_covariance(self.alpha,self.beta,self.r,self.contrast,self.nspots,lmax=self.lmax)
        return mean_ylm, cov_ylm

    @property
    def mean_ylm(self):
        return compute_mean(self.alpha,self.beta,self.r,self.contrast,self.nspots,lmax=self.lmax)

    @property
    def cov_ylm(self):
        return compute_covariance(self.alpha,self.beta,self.r,self.contrast,self.nspots,lmax=self.lmax)

    # ------------------------------------------------------------------
    # Design matrix
    # ------------------------------------------------------------------

    def design_matrix(
        self,
        theta,
        inc=jnp.pi / 2,
        obl=0.0,
        period=1.0,
        u=(),
        r_occ=None,
        x=None,
        y=None,
        z=None,
        order=20,
        higher_precision=False,
    ):
        y = jnp.zeros((self.lmax + 1) ** 2)
        y = y.at[0].set(1.0)
        surface = Surface(y=Ylm.from_dense(y),inc=inc,obl=obl,period=period,u=u)

        return _design_matrix(surface,r=r_occ,x=x,y=y,z=z,theta=theta,order=order,higher_precision=higher_precision)

    # ------------------------------------------------------------------
    # Ylm sampling
    # ------------------------------------------------------------------
    def sample_ylm(
        self,
        key,
        nsamples,
        eps=1e-8,
    ):
        mean_ylm = self.mean_ylm
        cov_ylm = self.cov_ylm
        ny = mean_ylm.shape[0]
        L = jnp.linalg.cholesky(cov_ylm + eps * jnp.eye(ny))
        z = jax.random.normal(key,(ny, nsamples),)
        return (mean_ylm[:, None] + L @ z).T

    # ------------------------------------------------------------------
    # Flux sampling
    # ------------------------------------------------------------------

    @staticmethod
    def sample_flux(ylm_samples, M):
        return ylm_samples @ M.T

    def sample_flux_from_parameters(
        self,
        key,
        theta,
        inc=jnp.pi / 2,
        obl=0.0,
        period=1.0,
        u=(),
        nsamples=1,
        eps=1e-8,
    ):
        M = self.design_matrix(theta=theta,inc=inc,obl=obl,period=period,u=u,)
        ylm_samples = self.sample_ylm(key,nsamples=nsamples,eps=eps,)
        return self.sample_flux(ylm_samples,M,)

    # ------------------------------------------------------------------
    # Conditional Ylm distribution
    # ------------------------------------------------------------------  
    @staticmethod
    def conditional_ylm(
        mean_ylm,
        cov_ylm,
        f,
        M,
        ferr,
    ):
        f = jnp.atleast_2d(f)

        Cn = (ferr**2 * jnp.eye(M.shape[0]))
        Cf = (M @ cov_ylm @ M.T + Cn)
        K = cov_ylm @ M.T

        resid = (f - M @ mean_ylm)
        solved_resid = jnp.linalg.solve(Cf, resid.T,)

        mean_post = (mean_ylm + (K @ solved_resid).T)
        cov_post = (cov_ylm - K @ jnp.linalg.solve(Cf,K.T,))

        return mean_post, cov_post

    # ------------------------------------------------------------------
    # Conditional Ylm sampling
    # ------------------------------------------------------------------

    @staticmethod
    def sample_conditional_ylm(
        mean_ylm,
        cov_ylm,
        f,
        M,
        ferr,
        key,
        nsamples=1,
        eps=1e-8,
    ):
        mean_post, cov_post = (StarryProcess.conditional_ylm(mean_ylm,cov_ylm,f,M,ferr))

        N = mean_post.shape[-1]
        L = jnp.linalg.cholesky(cov_post + eps * jnp.eye(N))
        z = jax.random.normal(key,(mean_post.shape[0],nsamples,N,),)

        return (mean_post[:, None, :] + jnp.einsum("ij,csj->csi",L,z,))

    # ------------------------------------------------------------------
    # Conditional flux sampling
    # ------------------------------------------------------------------

    @staticmethod
    def sample_conditional_flux(
        mean_ylm,
        cov_ylm,
        f,
        M,
        ferr,
        key,
        nsamples=1,
        eps=1e-8,
    ):
        ylm_samples = (StarryProcess.sample_conditional_ylm(mean_ylm,cov_ylm,f,M,ferr,key,nsamples,eps))

        return jnp.einsum("csn,tn->cst",ylm_samples,M)

    # ------------------------------------------------------------------
    # Convenience conditional methods
    # ------------------------------------------------------------------

    def conditional(
        self,
        f,
        M,
        ferr,
    ):
        return self.conditional_ylm(self.mean_ylm,self.cov_ylm,f,M,ferr)

    def sample_conditional(
        self,
        f,
        M,
        ferr,
        key,
        nsamples=1,
        eps=1e-8,
    ):
        return self.sample_conditional_ylm(self.mean_ylm,self.cov_ylm,f,M,ferr,key,nsamples,eps)
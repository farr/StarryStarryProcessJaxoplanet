from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy

from jaxoplanet.core.limb_dark import light_curve as _limb_dark_light_curve
from jaxoplanet.starry.core.basis import A1, A2_inv, U
from jaxoplanet.starry.core.polynomials import Pijk
from jaxoplanet.starry.core.rotation import left_project, dot_rotation_matrix, sky_projection_axis_angle
from jaxoplanet.starry.core.solution import rT, solution_vector
from jaxoplanet.starry.surface import Surface
from jaxoplanet.starry.system_observable import system_observable

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary


def design_matrix(
    surface: Surface,
    r: float | None = None,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    theta: float | None = None,
    order: int = 20,
    higher_precision: bool = False,
):
    """Light curve of an occulted surface.

    Args:
        surface (Surface): Surface object
        r (float or None): radius of the occulting body, relative to the current map
           body
        x (float or None): x coordinate of the occulting body relative to the surface
           center. By default (None) 0.0
        y (float or None): y coordinate of the occulting body relative to the surface
           center. By default (None) 0.0
        z (float or None): z coordinate of the occulting body relative to the surface
           center. By default (None) 0.0
        theta (float):
            rotation angle of the map, in radians. By default 0.0
        order (int):
            order of the P integral numerical approximation. By default 20
        higher_precision (bool): whether to compute change of basis matrix as hight
            precision. By default False (only used to testing).

    Returns:
        ArrayLike: flux
    """
    if higher_precision:
        try:
            from jaxoplanet.starry.multiprecision import (
                basis as basis_mp,
                utils as utils_mp,
            )
        except ImportError as e:
            raise ImportError(
                "The `mpmath` Python package is required for higher_precision=True."
            ) from e

    total_deg = surface.deg

    rT_deg = rT(total_deg)

    x = 0.0 if x is None else x
    y = 0.0 if y is None else y
    z = 0.0 if z is None else z

    # no occulting body
    if r is None:
        b_rot = True
        theta_z = 0.0
        design_matrix_p = rT_deg

    # occulting body
    else:
        b = jnp.sqrt(jnp.square(x) + jnp.square(y))
        b_rot = jnp.logical_or(jnp.greater_equal(b, 1.0 + r), jnp.less_equal(z, 0.0))
        b_occ = jnp.logical_not(b_rot)

        # trick to avoid nan `x=jnp.where...` grad caused by nan sT
        r = jnp.where(b_rot, 1.0, r)
        b = jnp.where(b_rot, 1.0, b)

        if surface.ydeg == 0:
            if surface.udeg == 0:
                ld_u = jnp.array([])
            else:
                ld_u = jnp.concatenate(
                    [jnp.atleast_1d(jnp.asarray(u_)) for u_ in surface.u], axis=0
                )

            lc_func = partial(_limb_dark_light_curve, ld_u, order=order)
            lc = lc_func(b, r)
            return surface.amplitude * (1.0 + jnp.where(b_occ, lc, 0))

        else:
            theta_z = jnp.arctan2(x, y)
            sT = solution_vector(total_deg, order=order)(b, r)

        if total_deg > 0:
            if higher_precision:
                A2 = np.atleast_2d(utils_mp.to_numpy(basis_mp.A2(total_deg)))
            else:
                A2 = scipy.sparse.linalg.inv(A2_inv(total_deg))
                A2 = jax.experimental.sparse.BCOO.from_scipy_sparse(A2)
        else:
            A2 = jnp.array([[1]])

        design_matrix_p = jnp.where(b_occ, sT @ A2, rT_deg)

    # limb darkening
    if surface.udeg == 0:
        p_u = Pijk.from_dense(jnp.array([1]))
    else:
        u = jnp.array([1, *surface.u])
        p_u = Pijk.from_dense(u @ U(surface.udeg), degree=surface.udeg)

    # surface map * limb darkening map
    if higher_precision:
        A1_val = np.atleast_2d(utils_mp.to_numpy(basis_mp.A1(surface.ydeg)))
    else:
        A1_val = jax.experimental.sparse.BCOO.from_scipy_sparse(A1(surface.ydeg))

    norm = np.pi / (p_u.tosparse() @ rT(surface.udeg))

    N = (surface.ydeg + 1) ** 2
    eye = jnp.eye(N)

    # this version gives the same matrix with far fewer rotations, because
    # the flux is a fixed linear function of the rotated map, so it reduces to one fixed vector
    # the inclination and obliquity enter as a single rotation shared by all time steps
    # only the cheap z-axis spin rotation changes with time
    
    def flux_of(y_rot):
        p_y = Pijk.from_dense(A1_val @ y_rot, degree=surface.ydeg)
        p_yu = p_y * p_u
        return surface.amplitude * (p_yu.tosparse() @ design_matrix_p) * norm

    w = jax.vmap(flux_of)(eye)

    m_theta_z = -theta_z if theta_z is not None else theta_z
    axis_x, axis_y, axis_z, angle = sky_projection_axis_angle(surface._inc, surface._obl)
    T1 = dot_rotation_matrix(surface.ydeg, 1.0, None, None, -0.5 * jnp.pi)(eye)
    T3 = dot_rotation_matrix(surface.ydeg, axis_x, axis_y, axis_z, angle)(eye)
    T4 = dot_rotation_matrix(surface.ydeg, None, None, 1.0, m_theta_z)(eye)
    v = T3 @ (T4 @ w)

    T2 = jax.vmap(lambda th: dot_rotation_matrix(surface.ydeg, None, None, 1.0, -th)(eye))(jnp.atleast_1d(theta))
    return jnp.real(jnp.einsum("kjl,l->kj", T2, v) @ T1.T)
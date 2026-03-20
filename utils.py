import jax.numpy as jnp
import jax.lax as jlx
import jax.scipy.special as jss

def Fam(F, Fcp, l, alpha, beta, i):
    """Compute `2F1(a-1, b, c, d)` in terms of `2F1(a, b, c, d)` and `2F1(a, b, c+1,
    d)`."""
    return 2*F + ((2*(l + beta) - i) / (i - 2*(l + alpha + beta))) * Fcp

def Fcm(F, Fap, l, alpha, beta, i):
    """Compute `2F1(a, b, c-1, d)` in terms of `2F1(a, b, c, d)` and `2F1(a+1,
    b, c, d)`."""
    return (alpha + beta + l - 1) / (alpha + beta + l - (i+2)/2) * F + i / (i - 2*(alpha + beta + l - 1)) * Fap

def hyp2f1_sequence(l, alpha, beta, imax=None):
    r"""Returns a sequence of `2F1` hypergeometric function values at even
    indices.

    .. math::

        \begin{cases} {}_2 F_1\left( -\frac{i}{2}, \alpha, l + \alpha + \beta -
        \frac{i}{2}, -1 \right) & i = 0, 2, 4, \ldots, i_\mathrm{max} \\ 0 & i =
        1, 3, 5, \ldots \end{cases}

    The hypergeometric function sequences are computed using the contiguous
    relations
    (https://en.wikipedia.org/wiki/Hypergeometric_function#Gauss's_contiguous_relations)
    in a `jax.lax.scan` loop, suitable for JAX's JIT compilation and automatic
    differentiation.

    :param l: The spherical harmonic degree (or sum of degrees for second
        moments).
    :param alpha: The first parameter of the beta distribution on cosine angle.
    :param beta: The second parameter of the beta distribution on cosine angle.
    :param imax: The maximum index to compute. If None, computes up to `2*l` by
        default.

    :returns: A length `imax+1` array whose even-index entries contain the
        hypergeometric function values and odd-index entries are zero.

    .. doctest::

        >>> import jax.scipy.special as jss
        >>> l, alpha, beta = 3, 2.0, 5.0
        >>> imax = 12
        >>> result_sequence = hyp2f1_sequence(l, alpha, beta, imax)
        >>> expected_sequence = jnp.array([jss.hyp2f1(-i/2, alpha, l + alpha + beta - i/2, -1.0) if i % 2 == 0 else 0.0 for i in range(imax+1)])
        >>> jnp.allclose(result_sequence, expected_sequence)
        Array(True, dtype=bool)
    """
    if imax is None:
        imax = 2 * l

    init = (jss.hyp2f1(1.0, alpha, l + alpha + beta, -1.0), jss.hyp2f1(1.0, alpha, l + alpha + beta + 1, -1.0))

    def scan_f(state, i):
        F_, Fcp_ = state
        Fam_ = Fam(F_, Fcp_, l, alpha, beta, i)
        Fcm_ = Fcm(Fam_, F_, l, alpha, beta, i)
        return (Fcm_, Fam_), Fam_
        
    _, result_evens = jlx.scan(scan_f, init, jnp.arange(imax+1, step=2))
    result = jnp.zeros(imax+1)
    result = result.at[0::2].set(result_evens)
    return result

def gamma_sequence1(l, alpha, beta, imax=None):
    """Returns a sequence of gamma functions which is zero at odd indices, and even
    indices `i` given by `Gamma(alpha) * Gamma(l + beta - i/2) / Gamma(l + alpha +
    beta - i/2)`.

    .. doctest::

        >>> import jax.lax as jlx
        >>> import jax.numpy as jnp
        >>> l, alpha, beta = 3, 2.0, 5.0
        >>> imax = 12
        >>> result_sequence = gamma_sequence1(l, alpha, beta, imax)
        >>> expected_sequence = jnp.array([jnp.exp(jlx.lgamma(alpha) + jlx.lgamma(l + beta - i/2) - jlx.lgamma(l + alpha + beta - i/2)) if i % 2 == 0 else 0.0 for i in range(imax+1)])
        >>> jnp.allclose(result_sequence, expected_sequence)
        Array(True, dtype=bool)
    """
    if imax is None:
        imax = 2*l

    init = jnp.exp(jlx.lgamma(alpha) + jlx.lgamma(l + beta + 1) - jlx.lgamma(l + alpha + beta + 1))

    def scan_f(state, i):
        next_state = state * (l + alpha + beta - i/2) / (l + beta - i/2)
        return (next_state, next_state)
    
    _, result_evens = jlx.scan(scan_f, init, jnp.arange(imax+1, step=2))
    result = jnp.zeros(imax+1)
    result = result.at[0::2].set(result_evens)
    return result

def gamma_sequence2(l, alpha, beta, imax=None):
    """Returns a sequence of gamma functions which is zero at odd indices and at
    even indices `i` given by `2^l * Gamma((1+i)/2) * Gamma(l + (i-1)/2) /
    (jnp.pi*Gamma(l+1))`.

    .. doctest::

        >>> import jax.scipy.special as jss
        >>> import jax.numpy as jnp
        >>> l, alpha, beta = 3, 2.0, 5.0
        >>> imax = 6
        >>> result_sequence = gamma_sequence2(l, alpha, beta, imax)
        >>> expected_sequence = jnp.array([2**l * jss.gamma((1+i)/2) * jss.gamma(l + (1-i)/2) / (jnp.pi * jss.gamma(l+1)) if i % 2 == 0 else 0.0 for i in range(imax+1)])
        >>> jnp.allclose(result_sequence, expected_sequence)
        Array(True, dtype=bool)
    """
    if imax is None:
        imax = 2*l

    init = jss.gamma(-1/2)*jnp.exp(l*jnp.log(2) + jlx.lgamma(l + 3/2) - jnp.log(jnp.pi) - jlx.lgamma(l + 1.0))

    def scan_f(state, i):
        next_state = state * ((1+i)/2 - 1) / (l + (1-i)/2)
        return (next_state, next_state)
    
    _, result_evens = jlx.scan(scan_f, init, jnp.arange(imax+1, step=2))
    result = jnp.zeros(imax+1)
    result = result.at[0::2].set(result_evens)
    return result
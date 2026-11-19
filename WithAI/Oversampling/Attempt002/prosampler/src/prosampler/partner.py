"""
Design of the partner kernel ``d`` for a given kernel ``u``.

Problem
-------
Given ``u`` find ``d`` such that ``h = u * d`` is an exact Nyquist(L) kernel,

    h[n0 + L k] = delta[k]   for all k,      n0 = L * latency,

while ``d`` is as good a lowpass as possible.  For fixed ``u`` the condition is LINEAR in ``d``:

    C d = e,        C[m, j] = u[p_m - j],   p_m = n0 mod L, n0 mod L + L, ...,   e = 1 at p_m = n0, else 0.

so this is an equality-constrained quadratic program

    minimise  1/2 || M d - r ||^2      subject to   C d = e .

Objectives (``objective=``)
    ``"grid"``    least squares on a frequency grid: passband rows ask ``D(w) U(w) = L exp(-j w n0)``
                  (a flat passband), stopband rows ask ``D(w) = 0``, optional transition rows ask ``D(w) = 0``
                  (intended to limit a bump of ``d`` in the unconstrained transition band; measured: for linear-phase
                  pairs it does NOT help - the ~+1.1 dB bump grows to ~+1.6 dB and the stopband gets much worse -
                  so the default is 0).
    ``"energy"``  closed form stopband energy ``d^T Q d`` (no grid; this is what the old KKT script did),
                  optionally plus transition-band energy.

Solution methods (``method=``)
    ``"nullspace"``  parametrise all solutions of ``C d = e`` as ``d0 + Z y`` (SVD of ``C``) and solve the
                     unconstrained problem in ``y``.  Robust, the default.
    ``"kkt"``        assemble and solve the KKT system ``[H C^T; C 0] [d; lambda] = [g; e]`` directly.  Simple,
                     but the matrix can be extremely ill-conditioned (the stopband-energy matrix has a
                     condition number around 1e12 for long kernels).

Both give the same optimum; afterwards the Nyquist condition is enforced exactly by a minimum-norm correction.
When ``u`` is symmetric and the latency is centred, ``d`` is parametrised as symmetric too (halving the number of
unknowns and guaranteeing exact linear phase).
"""
from __future__ import annotations

import warnings

import numpy as np

from ._util import conv_rows, is_symmetric, mirror_matrix
from .analysis import band_energy_matrix, freqz
from .errors import ProsamplerError
from .nyquist import nyquist_error, symmetric_latency
from .spec import Spec

METHODS = ("nullspace", "kkt")
OBJECTIVES = ("grid", "energy", "minimax")


def resolve_latency(u, n_d: int, L: int, latency):
    """Latency in base-rate samples: as given, or the centred one of a linear-phase pair (else ``ValueError``)."""
    if latency is not None:
        return int(latency)
    n_u = len(u)
    if not (is_symmetric(u) and n_u % 2 == 1 and n_d % 2 == 1):
        raise ValueError("latency must be given unless u is symmetric with odd length and n_d is odd")
    return symmetric_latency(n_u, n_d, L)


def design_partner(u, spec: Spec, n_taps: int, latency=None, *, method: str = "nullspace",
                   objective: str = "grid", stop_weight: float = 1.0, transition_weight: float = 0.0,
                   n_grid: int = 3000, symmetric="auto", ridge: float = 1e-12, refine: bool = True,
                   transition_max_db=None, passband_flat_db=None) -> np.ndarray:
    """
    Kernel ``d`` of length ``n_taps`` such that ``u * d`` is exactly Nyquist(L) with centre ``L * latency``.

    Parameters
    ----------
    u : array_like
        Any kernel (it does not have to be a windowed sinc).  It should be a decent lowpass; its DC gain should be
        ``spec.L`` if ``d`` is to have DC gain 1.
    spec : Spec
        Provides ``L`` and the passband (and thereby the stopband edge).  ``atten_db`` is not used here.
    n_taps : int
        Length of ``d``.  There must be enough freedom left after the Nyquist constraints, roughly
        ``n_taps > (len(u) + n_taps) / L``; ``d`` typically needs about twice the length of ``u`` for a comparable
        stopband.
    latency : int, optional
        Roundtrip latency in base-rate samples (centre of ``u * d`` at ``L * latency``).  Default: the centre of a
        linear-phase pair (requires symmetric odd-length ``u`` and odd ``n_taps`` with ``len(u) + n_taps - 2``
        a multiple of ``2 L``; see :func:`prosampler.symmetric_latency`).
    method : {"nullspace", "kkt"}
    objective : {"grid", "energy", "minimax"}
        ``"minimax"`` minimises the worst-case stopband magnitude by linear programming (see
        :mod:`prosampler.minimax`); ``method``, ``stop_weight``, ``transition_weight``, ``symmetric``, ``ridge`` and
        ``refine`` are ignored then.  It needs symmetric odd-length ``u`` and the centred latency.
    transition_max_db, passband_flat_db : float, optional
        Only for ``"minimax"``: upper bound (dB) on ``|D|`` between the passband and stopband edges, and the
        allowed deviation (dB) of the product ``D U / L`` from 1 in the passband.
    stop_weight, transition_weight : float
        Weights of the stopband and (optional) transition-band terms relative to the passband term (``"grid"``) or
        relative to each other (``"energy"``).
    n_grid : int
        Grid size for the ``"grid"`` objective.
    symmetric : {"auto", True, False}
        Parametrise ``d`` as symmetric.  ``"auto"`` does so when ``u`` is symmetric and the latency is centred.
    ridge : float
        Tikhonov term added to the energy matrix (as in the old script) so that it is positive definite.
    refine : bool
        Apply a minimum-norm correction so that the Nyquist condition holds to machine precision.

    Raises
    ------
    ProsamplerError
        If no exact partner of this length exists (inconsistent constraints) or the solve fails.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    if objective == "minimax":
        from .minimax import design_partner_minimax
        return design_partner_minimax(u, spec, n_taps, latency, transition_max_db=transition_max_db,
                                      passband_flat_db=passband_flat_db, n_grid=n_grid)
    u = np.asarray(u, dtype=float)
    L, n_u, n_d = spec.L, len(u), int(n_taps)
    latency = resolve_latency(u, n_d, L, latency)
    n0, n_h = L * latency, n_u + n_d - 1
    if not 0 <= n0 < n_h:
        raise ValueError(f"latency {latency} puts the centre L*latency = {n0} outside of u*d (length {n_h})")

    centred = is_symmetric(u) and n_u % 2 == 1 and n_d % 2 == 1 and 2 * n0 == n_u + n_d - 2
    if symmetric == "auto":
        use_sym = centred
    elif symmetric:
        if not centred:
            raise ValueError("symmetric=True needs symmetric odd-length u, odd n_taps and the centred latency")
        use_sym = True
    else:
        use_sym = False
    T = mirror_matrix(n_d) if use_sym else np.eye(n_d)
    n_th = T.shape[1]

    # ---- exact Nyquist constraints  C th = e   (rows beyond the centre are duplicates for symmetric kernels)
    idx = np.arange(n0 % L, n_h, L)
    if use_sym:
        idx = idx[idx <= n0]
    C = conv_rows(u, n_d, idx) @ T
    e = (idx == n0).astype(float)

    # ---- objective  1/2 ||M th - rhs||^2
    spec_grid = spec.grid(n_grid)
    if objective == "grid":
        w, g = spec_grid.w, spec_grid
        F = np.exp(-1j * np.outer(w, np.arange(n_d))) @ T
        Up = freqz(u, w[g.passband])
        tgt = L * np.exp(-1j * w[g.passband] * n0) / Up
        blocks = [F[g.passband].real, F[g.passband].imag,
                  stop_weight * F[g.stopband].real, stop_weight * F[g.stopband].imag]
        rhs = [tgt.real, tgt.imag, np.zeros(2 * int(g.stopband.sum()))]
        if transition_weight > 0 and g.transition.any():
            blocks += [transition_weight * F[g.transition].real, transition_weight * F[g.transition].imag]
            rhs.append(np.zeros(2 * int(g.transition.sum())))
        M = np.vstack(blocks)
        rhs = np.concatenate([rhs[0], rhs[1], *rhs[2:]])
    else:
        Q = stop_weight ** 2 * band_energy_matrix(n_d, spec.stopband_edge, np.pi)
        if transition_weight > 0:
            Q = Q + transition_weight ** 2 * band_energy_matrix(n_d, spec.passband_edge, spec.stopband_edge)
        Qt = T.T @ Q @ T
        r = ridge
        while True:
            try:
                M = np.linalg.cholesky(Qt + r * np.eye(n_th)).T       # Qt + r I = M^T M
                break
            except np.linalg.LinAlgError:
                r *= 10
                if r > 1e-3:
                    raise ProsamplerError("stopband energy matrix is not positive definite") from None
        rhs = np.zeros(n_th)

    # ---- solve
    if method == "nullspace":
        U_, sv, Vt = np.linalg.svd(C, full_matrices=True)
        rank = int(np.sum(sv > sv[0] * 1e-14)) if sv.size and sv[0] > 0 else 0
        Vr, Ur = Vt[:rank].T, U_[:, :rank]
        th = Vr @ ((Ur.T @ e) / sv[:rank])
        for _ in range(4):                                            # iterative refinement of the particular solution
            th = th - Vr @ ((Ur.T @ (C @ th - e)) / sv[:rank])
        Z = Vt[rank:].T
        if Z.shape[1] == 0:
            warnings.warn("no degrees of freedom left: d is fully determined by the Nyquist constraints "
                          "(try a longer n_taps)", stacklevel=2)
        else:
            y = np.linalg.lstsq(M @ Z, rhs - M @ th, rcond=None)[0]
            th = th + Z @ y
    else:
        H, gvec, m = M.T @ M, M.T @ rhs, C.shape[0]
        KKT = np.block([[H, C.T], [C, np.zeros((m, m))]])
        try:
            th = np.linalg.solve(KKT, np.concatenate([gvec, e]))[:n_th]
        except np.linalg.LinAlgError:
            raise ProsamplerError("the KKT system is singular; use method='nullspace'") from None

    if refine:
        for _ in range(3):
            res = C @ th - e
            if np.max(np.abs(res)) < 1e-16:
                break
            th = th - np.linalg.lstsq(C, res, rcond=None)[0]
    d = T @ th

    err = nyquist_error(u, d, L, latency)
    if not err < 1e-8:
        raise ProsamplerError(
            f"no exact partner of length {n_d} found (Nyquist error {err:.1e}); the constraints are inconsistent or "
            f"ill-conditioned - try a longer n_taps or another latency")
    return d

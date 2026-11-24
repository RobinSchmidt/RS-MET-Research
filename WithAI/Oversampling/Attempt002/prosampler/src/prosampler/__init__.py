"""
prosampler - perfect-reconstruction oversampling filters.

Design FIR upsampling / downsampling kernel pairs ``(u, d)`` such that

    input -> zero-stuff by L -> u -> (processing) -> d -> keep every L-th sample -> output

is an exact identity (up to float roundoff, delayed by ``latency`` samples) when the processing does
nothing - and a single matched kernel ``g`` with ``u = d = g``.

Status: early development.  See the README for what is implemented.
"""
from .analysis import band_energy_matrix, freqz, group_delay_passband, passband_ripple_db, stopband_db, transition_peak_db
from .errors import ProsamplerError, SpecNotReached, VerificationError
from .kernels import design_kaiser_sinc, kaiser_beta, kaiser_length_estimate, kaiser_sinc, windowed_sinc
from .nyquist import nyquist_error, partner_length, symmetric_latency, verify_roundtrip
from .compare import Comparison, compare
from .design import design_pair
from .joint import optimise_joint, rebalance, tie
from .minimax import design_partner_minimax, optimise_joint_minimax
from .jointlp import DEFAULT_RIPPLE_DB, optimise_joint_slp
from .search import search_pair, best_pair_for_length
from .transition import TransitionShape
from .matched import design_matched, optimise_matched
from .matchedlp import optimise_matched_slp, search_matched
from .pair import FilterPair
from .partner import design_partner
from .process import reference_roundtrip
from .spec import Grid, Spec, passband_from_hz

# plotting functions live in prosampler.plots (matplotlib is optional)
__version__ = "0.1.0.dev0"

__all__ = [
    "Spec", "Grid", "passband_from_hz",
    "windowed_sinc", "kaiser_sinc", "design_kaiser_sinc", "kaiser_beta", "kaiser_length_estimate",
    "nyquist_error", "symmetric_latency", "partner_length", "verify_roundtrip", "reference_roundtrip",
    "FilterPair", "design_partner", "design_pair", "design_matched", "optimise_matched", "optimise_matched_slp", "search_matched",
    "optimise_joint", "optimise_joint_minimax", "design_partner_minimax", "optimise_joint_slp", "search_pair", "best_pair_for_length", "DEFAULT_RIPPLE_DB", "TransitionShape", "rebalance", "tie", "compare", "Comparison",
    "freqz", "band_energy_matrix", "stopband_db", "passband_ripple_db", "transition_peak_db", "group_delay_passband",
    "ProsamplerError", "VerificationError", "SpecNotReached",
]

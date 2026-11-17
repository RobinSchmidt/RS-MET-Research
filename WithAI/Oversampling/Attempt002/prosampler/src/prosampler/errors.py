"""Exceptions used by prosampler."""


class ProsamplerError(Exception):
    """Base class for all prosampler errors."""


class VerificationError(ProsamplerError, AssertionError):
    """A filter pair failed the perfect-reconstruction (identity roundtrip) check."""


class SpecNotReached(ProsamplerError):
    """A design routine could not meet the requested specification (see the message for details)."""

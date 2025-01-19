# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Version according to SemVer versioning system (https://semver.org/)."""


__all__ = [
    "VERSION",
]


_MAJOR = "0"  # Major version to increment in case of incompatible API changes

_MINOR = (
    "0"  # Minor version to increment in case of backward compatible new functionality
)

_PATCH = "1"  # Patch version to increment in case of backward compatible bug fixes

VERSION = f"{_MAJOR}.{_MINOR}.{_PATCH}"
"""The package version."""

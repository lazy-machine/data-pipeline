"""Every built-in check. Importing this package registers them, in report order."""

from . import integrity, splits, temporal, duplicates, target, poisoning  # noqa: F401

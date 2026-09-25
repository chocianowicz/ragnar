"""Admin settings that outlive a browser session.

config.yaml is the deployment's default and belongs in version control.
What an operator tunes on a running instance belongs with that instance's
data, not in the repo — so it is layered on top, in data/settings.json.

Only admin settings are stored. Per-question choices (which model answers,
how strict to be) stay in the session: they are a property of the question
being asked, and writing them back would mean one person's choice changing
another's.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


class Preferences:
    """A small JSON file layered over config.yaml."""

    def __init__(self, path: Path):
        self._path = Path(path)
        self._values = self._load()

    def _load(self) -> dict:
        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError) as exc:
            # A corrupt preferences file must not stop the app: the
            # defaults in config.yaml are a complete, working
            # configuration on their own.
            log.warning("ignoring unreadable %s: %s", self._path, exc)
            return {}

    def get(self, key: str, default=None):
        """The stored value, or `default` from config.yaml.

        The default's type wins: a hand-edited file that says "25" for a
        number should not turn a slider into a string.
        """
        if key not in self._values:
            return default
        value = self._values[key]
        if default is not None and not isinstance(value, type(default)):
            try:
                return type(default)(value)
            except (TypeError, ValueError):
                return default
        return value

    def set(self, key: str, value) -> None:
        """Store one value, writing only when it actually changed."""
        if self._values.get(key) == value:
            return
        self._values[key] = value
        self._save()

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._values, indent=2, sort_keys=True),
                encoding="utf-8")
        except OSError as exc:
            log.warning("could not save %s: %s", self._path, exc)

    def as_dict(self) -> dict:
        return dict(self._values)

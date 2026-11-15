"""Progress display that works in three places:

* an interactive notebook (Kaggle / Colab / Jupyter): live tqdm widget bars;
* a terminal: plain tqdm bars;
* a Kaggle background run ("Save Version -> Save & Run All", KAGGLE_KERNEL_RUN_TYPE=Batch), whose log is
  plain text: one status line per bar every ``every_s`` seconds (rate and ETA included), so the log
  viewer shows the progress without thousands of carriage-return updates.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, Optional, Tuple


def batch_mode() -> bool:
    """True when nobody is watching a live notebook (Kaggle background run, or forced with SIDX_PLAIN=1)."""
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() == "batch" or os.environ.get("SIDX_PLAIN") == "1"


def _fmt_s(s: float) -> str:
    s = int(max(0, s))
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


class Bars:
    """A set of named progress bars, updated with absolute values: ``bars.set(name, done, total)``."""

    def __init__(self, unit: str = "it", every_s: float = 60.0, title: str = ""):
        self.unit, self.every_s, self.title = unit, every_s, title
        self.plain = batch_mode()
        self._bars: Dict[str, object] = {}
        self._first: Dict[str, Tuple[float, int]] = {}      # name -> (time, done) when first seen
        self._state: Dict[str, Tuple[int, int, str]] = {}
        self._status = ""
        self._last_print = 0.0
        if not self.plain:
            from tqdm.auto import tqdm
            self._tqdm = tqdm
        if title:
            print(title, flush=True)

    def set(self, name: str, done: int, total: int, postfix: str = "") -> None:
        now = time.time()
        if name not in self._first:
            self._first[name] = (now, done)
            if not self.plain:
                self._bars[name] = self._tqdm(total=total, initial=done, desc=name, unit=self.unit,
                                              dynamic_ncols=True, mininterval=1)
        self._state[name] = (done, total, postfix)
        if not self.plain:
            b = self._bars[name]
            if b.total != total:
                b.total = total
            if done > b.n:
                b.update(done - b.n)
            if postfix:
                b.set_postfix_str(postfix, refresh=False)
            b.refresh()

    def status(self, text: str) -> None:
        """A free-text line (e.g. which task runs on which GPU)."""
        if not self.plain and text != self._status:
            if "__status__" not in self._bars:
                self._bars["__status__"] = self._tqdm(total=0, bar_format="{desc}", desc=text, dynamic_ncols=True)
            else:
                self._bars["__status__"].set_description_str(text)
        self._status = text

    def maybe_print(self, force: bool = False) -> None:
        """Plain mode only: print one line per bar every ``every_s`` seconds."""
        if not self.plain:
            return
        now = time.time()
        if not force and now - self._last_print < self.every_s:
            return
        self._last_print = now
        stamp = time.strftime("%H:%M:%S")
        for name, (done, total, postfix) in self._state.items():
            t0, d0 = self._first[name]
            rate = (done - d0) / max(now - t0, 1e-6)
            eta = _fmt_s((total - done) / rate) if rate > 0 and total > done else ("done" if done >= total else "?")
            pct = 100.0 * done / total if total else 100.0
            print(f"[{stamp}] {name}: {done}/{total} {self.unit}s ({pct:.0f}%) | "
                  f"{rate * 60:.1f}/min | ETA {eta}" + (f" | {postfix}" if postfix else ""), flush=True)
        if self._status:
            print(f"[{stamp}] {self._status}", flush=True)

    def close(self) -> None:
        self.maybe_print(force=True)
        for b in self._bars.values():
            b.close()
        self._bars.clear()


def tail(path: str, n: int = 25) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 20000))
            text = f.read().decode("utf-8", "replace").replace("\r", "\n")
        return "\n".join([l for l in text.splitlines() if l.strip()][-n:])
    except OSError:
        return "(no log)"


def last_fraction(path: str) -> Optional[Tuple[int, int]]:
    """The last ``n/total`` that a tqdm bar wrote into a log file (e.g. diffusers' 'Steps: 45%|...| 270/600')."""
    import re
    t = tail(path, 5)
    m = re.findall(r"(\d+)/(\d+) \[", t)
    return (int(m[-1][0]), int(m[-1][1])) if m else None


if __name__ == "__main__":  # tiny demo: python -m sidx.progress
    b = Bars(unit="sample", every_s=1, title="demo")
    for i in range(6):
        b.set("job_a", i, 5 if i < 5 else 5)
        b.status(f"running step {i}")
        b.maybe_print()
        time.sleep(0.3)
    b.close()
    sys.exit(0)

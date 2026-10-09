"""Split a finished run into phases, finer than its stamps. Local, reads the log kaggle_push.py downloaded.

  python phases.py kaggle_out/<tag>/pair-run-<tag>.log [word ...]

Kaggle timestamps every log line and tqdm prints a bar for every loop, so the gaps between the model's own messages and
the per-item rate of each loop are already in the log of a run you paid for. Prints every stamp (`[t+..s]`) and every
line containing one of the extra words, with the seconds since the previous printed line, and one line per loop: items,
seconds, seconds per item, and how long the first item took (a first item far above the average is warm-up, not work).
"""
import json, re, sys

BAR = re.compile(r"(\d+)/(\d+) \[(\d+(?::\d+)+)<")


def seconds(clock: str) -> int:
    return sum(int(x) * 60 ** i for i, x in enumerate(reversed(clock.split(":"))))


def phases(log: str, words: list) -> list:
    rows = json.load(open(log, encoding="utf-8"))
    t0 = next((r["time"] for r in rows if "[t+" in r.get("data", "")), rows[0]["time"])
    out, bar, prev = [], None, 0.0  # bar: [total, first item s, items done, elapsed s]
    def close():
        nonlocal bar
        if bar and bar[2]:
            total, first, n, el = bar
            out.append(f"{'':>17}loop: {n} of {total} items in {el}s, {el/n:.2f} s/item, first item {first}s")
        bar = None
    for r in rows:
        for line in re.split(r"[\r\n]+", r.get("data", "")):
            m = BAR.search(line)
            if m:
                n, total, el = int(m[1]), int(m[2]), seconds(m[3])
                if bar and (total != bar[0] or n < bar[2]):
                    close()
                bar = bar or [total, None, 0, 0]
                if n >= 1 and bar[1] is None:
                    bar[1] = el
                bar[2], bar[3] = n, el
            elif "[t+" in line or any(w in line for w in words):
                close()
                t = r["time"] - t0
                out.append(f"{t:7.1f}s {t-prev:+7.1f}s  {line.strip()[:110]}")
                prev = t
    close()
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print("\n".join(phases(sys.argv[1], sys.argv[2:])))

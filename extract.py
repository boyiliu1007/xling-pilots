"""Answer extraction, normalization and majority voting. Pure Python, no GPU deps."""
import math
import re
from collections import Counter

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")


def last_boxed(text):
    """Content of the LAST \\boxed{...} (or \\fbox{...}), handling nested braces. None if absent/unclosed."""
    if text is None:
        return None
    starts = [m.end() for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text)]
    for start in reversed(starts):
        depth, i = 1, start
        while i < len(text):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i]
            i += 1
        # unclosed: try an earlier one
    return None


def _strip_wrappers(s):
    s = s.strip()
    # unwrap \text{...}, \textbf{...}, \mathrm{...}, \mbox{...} repeatedly (innermost content kept)
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\\(?:text|textbf|mathrm|mathbf|mbox|operatorname)\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\$", "").replace("$", "").replace("\\%", "").replace("%", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", "").replace("~", "")
    s = s.replace("{,}", ",")  # LaTeX thousands separator 1{,}200
    s = s.replace("−", "-")  # unicode minus
    return s.strip()


def _to_number(tok):
    tok = tok.replace(",", "")
    try:
        x = float(tok)
    except ValueError:
        return None
    if not math.isfinite(x):
        return None
    return int(round(x)) if abs(x - round(x)) < 1e-9 else x


def parse_number(s):
    """Normalize a short answer string to int/float. Returns None unless it is a single number
    (optionally with surrounding unit words, e.g. '42 dollars')."""
    if s is None:
        return None
    s = _strip_wrappers(s)
    # simple LaTeX fraction \frac{a}{b} / \dfrac{a}{b}
    m = re.fullmatch(r"(-?)\\d?frac\{(-?[\d.,]+)\}\{(-?[\d.,]+)\}", s.replace(" ", ""))
    if m:
        a, b = _to_number(m.group(2)), _to_number(m.group(3))
        if a is None or not b:
            return None
        v = (-1 if m.group(1) else 1) * a / b
        return _to_number(repr(v))
    nums = _NUM_RE.findall(s)
    if len(nums) != 1:
        return None  # zero or ambiguous (e.g. '3 and 5', '2x+3=7')
    # reject things like 'x = 5' only if extra math operators surround; allow unit words
    rest = s.replace(nums[0], " ", 1)
    if re.search(r"[0-9=+*/^<>]", rest):
        return None
    return _to_number(nums[0])


def extract_answer(text):
    """Last \\boxed{} if present (and parseable); otherwise the last number in the text."""
    if text is None:
        return None
    boxed = last_boxed(text)
    if boxed is not None:
        v = parse_number(boxed)
        if v is None and "=" in boxed:  # e.g. \boxed{80 + 150 = 230 \text{ miles}} -> right-hand side
            v = parse_number(boxed.rsplit("=", 1)[1])
        return v
    # fallback: last numeric expression in the text (after stripping any thinking block)
    body = text.split("</think>")[-1]
    body = _strip_wrappers(body)
    nums = _NUM_RE.findall(body)
    # skip separators that are just commas e.g. trailing '1,' handled by _to_number
    for tok in reversed(nums):
        v = _to_number(tok.rstrip(","))
        if v is not None:
            return v
    return None


def numeric_equal(a, b, tol=1e-6):
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def majority_vote(preds, n=None):
    """preds: list of numbers/None. Returns (majority_answer, vote_counts, vote_share, is_tie).
    vote_share denominator is n (= len(preds) by default), including unparseable samples."""
    n = len(preds) if n is None else n
    counts = Counter()
    for p in preds:
        if p is None:
            continue
        # merge numerically equal values (e.g. 3 and 3.0)
        key = next((k for k in counts if numeric_equal(k, p)), p)
        counts[key] += 1
    if not counts:
        return None, {}, 0.0, False
    top = max(counts.values())
    leaders = [k for k, c in counts.items() if c == top]
    vote_counts = {str(k): c for k, c in counts.most_common()}
    if len(leaders) > 1:
        return None, vote_counts, top / n, True
    return leaders[0], vote_counts, top / n, False

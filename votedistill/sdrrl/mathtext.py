"""Split a Markdown + LaTeX solution into translatable sentences, with math protected.

NLLB corrupts raw LaTeX (e.g. `$ l $` -> `$ 1 $`) but copies `[M1]`-style placeholders, so inline math is masked
before translation and restored afterwards. Display math, rules and math-only lines are never sent to the translator.
Pure Python, no GPU dependencies.
"""
import re
from dataclasses import dataclass, field

DISPLAY_RE = re.compile(r"(\$\$.*?\$\$|\\\[.*?\\\])", re.S)
INLINE_RE = re.compile(r"\\\(.*?\\\)|\$[^$\n]+\$")
PREFIX_RE = re.compile(r"^(\s*(?:(?:#{1,6}|[-*+]|\d+[.)]|>)\s+)*)")
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")
PLACEHOLDER_RE = re.compile(r"\[\s*M\s*(\d+)\s*\]", re.I)


@dataclass
class Piece:
    """kind 'keep': emitted verbatim. kind 'text': `text` is a masked sentence to translate; `masks` restores it."""
    kind: str
    text: str
    masks: list = field(default_factory=list)


def _boxed_spans(text):
    """(start, end) of every \\boxed{...} / \\fbox{...} with balanced braces."""
    spans = []
    for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        depth, i = 1, m.end()
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        if depth == 0:
            spans.append((m.start(), i))
    return spans


def mask_math(text):
    """Replace inline math and \\boxed{} with [M1], [M2], ...; returns (masked text, [original spans])."""
    spans = [(m.start(), m.end()) for m in INLINE_RE.finditer(text)]
    spans += [s for s in _boxed_spans(text) if not any(a <= s[0] and s[1] <= b for a, b in spans)]
    spans = sorted(s for s in spans if not any(o != s and o[0] <= s[0] and s[1] <= o[1] for o in spans))
    out, masks, last = [], [], 0
    for a, b in spans:
        if a < last:
            continue
        masks.append(text[a:b])
        out.append(text[last:a] + f"[M{len(masks)}]")
        last = b
    return "".join(out) + text[last:], masks


def unmask(text, masks):
    """Put the math back. None if the translator dropped, duplicated or invented a placeholder."""
    found = [int(k) for k in PLACEHOLDER_RE.findall(text)]
    if sorted(found) != list(range(1, len(masks) + 1)):
        return None
    return PLACEHOLDER_RE.sub(lambda m: masks[int(m.group(1)) - 1], text)   # a function's return is used literally


def _has_words(masked):
    return bool(re.search(r"[A-Za-z]{2,}", PLACEHOLDER_RE.sub(" ", masked)))


def _renumber(sentence, masks):
    """Give a sentence its own [M1..Mk] numbering and mask list."""
    own = []

    def sub(m):
        own.append(masks[int(m.group(1)) - 1])
        return f"[M{len(own)}]"
    return PLACEHOLDER_RE.sub(sub, sentence), own


def split_document(text):
    """Text -> pieces whose concatenation (after translating 'text' pieces and unmasking) rebuilds the document."""
    pieces = []
    for i, block in enumerate(DISPLAY_RE.split(text)):
        if i % 2:                                   # display math
            pieces.append(Piece("keep", block))
            continue
        lines = block.split("\n")
        for j, line in enumerate(lines):
            prefix = PREFIX_RE.match(line).group(1)
            body = line[len(prefix):].replace("**", "").replace("__", "")
            masked, masks = mask_math(body)
            if not _has_words(masked):
                pieces.append(Piece("keep", line))
            else:
                pieces.append(Piece("keep", prefix))
                sents = SENT_RE.split(masked)
                for k, s in enumerate(sents):
                    s_masked, s_masks = _renumber(s, masks)
                    if _has_words(s_masked):
                        pieces.append(Piece("text", s_masked.strip(), s_masks))
                    else:
                        pieces.append(Piece("keep", unmask(s_masked, s_masks)))
                    if k < len(sents) - 1:
                        pieces.append(Piece("keep", " "))
            if j < len(lines) - 1:
                pieces.append(Piece("keep", "\n"))
    return pieces


def rebuild(pieces, translate):
    """translate: {masked sentence: translation}. Returns (text or None, n_text_pieces, n_math_lost).
    The text is None if any sentence lost its math placeholders."""
    out, n_text, lost = [], 0, 0
    for p in pieces:
        if p.kind == "keep":
            out.append(p.text)
            continue
        n_text += 1
        t = unmask(translate[p.text], p.masks)
        if t is None:
            lost += 1
        else:
            out.append(t)
    return (None if lost else "".join(out)), n_text, lost

"""Code-switching of questions with a bilingual dictionary (SDRRL section 3.2.3).

Each word of a question that is found in the dictionary is replaced by its translation with probability p.
Answers are never changed, and math is left alone. SDRRL builds its dictionaries from MUSE, which has no
Swahili, so this uses FreeDict: the eng-swh dictionary (1,450 headwords) merged with the inverted swh-eng
dictionary (2,658 headwords), and the same in the other direction.
"""
import re
import tarfile
import urllib.request

from ..paths import CACHE
from .mathtext import mask_math, unmask

FREEDICT = {  # language pair -> (version, ISO 639-3 pair name)
    ("en", "swa"): ("0.2.2", "eng-swh"),
    ("swa", "en"): ("0.4.4", "swh-eng"),
}
URL = "https://download.freedict.org/dictionaries/{name}/{ver}/freedict-{name}-{ver}.src.tar.xz"
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
ENTRY_RE = re.compile(r"<entry\b.*?</entry>", re.S)
# Matched first and left untouched: [M1] placeholders, display math, LaTeX commands. Then ordinary words.
PROTECTED_RE = re.compile(r"\[M\d+\]|\$\$.*?\$\$|\\\[.*?\\\]|\\[A-Za-z]+|" + WORD_RE.pattern, re.S)


def _download(src, tgt):
    ver, name = FREEDICT[(src, tgt)]
    path = CACHE / "freedict" / f"freedict-{name}-{ver}.src.tar.xz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL.format(name=name, ver=ver), path)
    return path, name


def parse_tei(xml):
    """FreeDict TEI -> [(headword, [translations])]. Translations are in <def> or <quote>, comma/semicolon separated."""
    out = []
    for e in ENTRY_RE.findall(xml):
        head = re.search(r"<orth>(.*?)</orth>", e, re.S)
        if not head:
            continue
        trans = []
        for d in re.findall(r"<(?:def|quote)>(.*?)</(?:def|quote)>", e, re.S):
            d = re.sub(r"\([^)]*\)|<[^>]+>", " ", d)
            trans += [t.strip() for t in re.split(r"[,;]", d) if t.strip()]
        out.append((head.group(1).strip(), trans))
    return out


def _read(src, tgt):
    path, name = _download(src, tgt)
    with tarfile.open(path) as tar:                       # read the one TEI member; nothing is extracted to disk
        return parse_tei(tar.extractfile(f"{name}/{name}.tei").read().decode("utf8"))


def _single_word(s):
    return bool(re.fullmatch(r"[A-Za-z']+", s))


def load_dictionary(src, tgt):
    """{lowercase src word: tgt word}, single words only. Forward dictionary first, then the inverted reverse one."""
    d = {}
    for head, trans in _read(src, tgt):
        t = next((t for t in trans if _single_word(t)), None)
        if _single_word(head) and t:
            d.setdefault(head.lower(), t.lower())
    for head, trans in _read(tgt, src):
        if _single_word(head):
            for t in trans:
                t = re.sub(r"^to\s+", "", t)              # English verbs are listed as "to worship"
                if _single_word(t):
                    d.setdefault(t.lower(), head.lower())
    return d


def code_switch(text, dictionary, p, rng):
    """-> (switched text, words found in the dictionary, words replaced, total words)."""
    masked, masks = mask_math(text)
    found = swapped = total = 0

    def sub(m):
        nonlocal found, swapped, total
        w = m.group(0)
        if not w[0].isalpha():                            # math placeholder, display math or a LaTeX command
            return w
        total += 1
        t = dictionary.get(w.lower()) if len(w) > 1 else None    # single letters are variables or option labels
        if t is None:
            return w
        found += 1
        if rng.random() >= p:
            return w
        swapped += 1
        return t.capitalize() if w[0].isupper() else t
    out = PROTECTED_RE.sub(sub, masked)
    return unmask(out, masks), found, swapped, total

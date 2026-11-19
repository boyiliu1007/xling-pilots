"""Natural parallel text turned into instructions (SDRRL section 3.2.4, Table 1).

SDRRL samples 1,000 sentence pairs per language from OPUS-100. OPUS-100 has no English-Swahili pair, so this uses
MAFAND-MT (Adelani et al., 2022; human-translated news, CC BY-NC 4.0), read from the masakhane-io/lafand-mt repo.
Each pair gives two translation instructions (both directions) and one sentence-completion instruction in the
target language, split at a random word.
"""
import json
import urllib.request

from ..paths import CACHE

MAFAND_URL = "https://raw.githubusercontent.com/masakhane-io/lafand-mt/main/data/json_files/en-{lang}/train.json"
LANG_NAME = {"en": "English", "swa": "Swahili", "yor": "Yoruba"}


def load_pairs(lang, n, rng, min_words=6, max_words=40):
    """n random (english, target) sentence pairs of moderate length."""
    path = CACHE / "mafand" / f"en-{lang}.train.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MAFAND_URL.format(lang=lang), path)
    pairs = []
    with open(path) as fh:
        for line in fh:
            t = json.loads(line)["translation"]
            en, tg = t["en"].strip(), t[lang].strip()
            if min_words <= len(en.split()) <= max_words and min_words <= len(tg.split()) <= max_words:
                pairs.append((en, tg))
    if n > len(pairs):
        raise SystemExit(f"only {len(pairs)} usable en-{lang} pairs, asked for {n}")
    return rng.sample(pairs, n)


def instructions(pairs, lang, rng):
    """-> [{"subset", "user", "answer"}]: translation en->tgt, tgt->en, and target-language completion."""
    en_name, tg_name = LANG_NAME["en"], LANG_NAME[lang]
    out = []
    for en, tg in pairs:
        out.append({"subset": "mt_en2tgt", "user": f"Translate the following sentence from {en_name} to {tg_name}.\n{en}",
                    "answer": tg})
        out.append({"subset": "mt_tgt2en", "user": f"Translate the following sentence from {tg_name} to {en_name}.\n{tg}",
                    "answer": en})
        words = tg.split()
        cut = rng.randint(max(1, len(words) // 4), max(1, 3 * len(words) // 4))
        out.append({"subset": "completion", "answer": tg,
                    "user": f"Complete the following sentence in {tg_name} according to its context.\n{' '.join(words[:cut])}"})
    return out

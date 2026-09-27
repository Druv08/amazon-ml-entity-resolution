"""Token-alignment ("decoy") features: HOW a candidate's name and address differ from the S1's, not only how much.

The oracle-gap audit's uncertain band is dominated by two kinds of near-identical records (docs/p3_matching.md,
"Oracle gap"):
  true duplicates  typos and transpositions, OCR confusions (0/o, 1/l, c/e), added generic words (LLC, Services,
                   Center, Shri), token shuffles
  decoys           one distinctive name token replaced or morphed at its end (e.g. Tavell -> Tavelli, Quorin ->
                   Quorinex), an injected house-number prefix ("H.no 12 ..."), a truncated / extended number
Overall string similarity scores both alike. These features align the distinctive (non-generic) tokens and record
the TYPE of difference. They use the two records only, so they are computed identically on test.
"""

import re
import unicodedata

import numpy as np
import pandas as pd

GENERIC = frozenset(
    "llc l c inc co corp corporation company ltd limited pvt private plc llp lp pc p a services service center centre "
    "partners partner group groups associates association the and of dba com www in org net shri sri smt mr mrs ms dr "
    "m s pvtltd enterprises enterprise solutions solution international intl industries industry trading traders "
    "agency holdings".split())
_OCR = str.maketrans({"0": "o", "1": "l", "|": "l", "5": "s", "c": "e", "8": "b"})
_GENERIC_OCR = frozenset(g.translate(_OCR) for g in GENERIC)
_GENERIC_LONG = tuple(g for g in GENERIC if len(g) >= 5)
HNO = re.compile(r"^\s*(h\.?\s*no|house\s*no|h\s*n)\b", re.I)
FEATURES = ["dt_s1_distinct", "dt_exact", "dt_ocr", "dt_typo", "dt_morph", "dt_none", "dt_cand_extra",
            "dt_replaced", "dt_generic_added", "dt_frac_explained", "dt_cand_hno", "dt_num_extend", "dt_num_equal"]


def _tokens(s):
    if not isinstance(s, str):
        return []
    s = "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", s)


def _one_edit(a, b):
    """-> None, or the kind of a single Damerau-Levenshtein edit between a and b: "end" when it changes the token's
    ending (last-character substitution, insertion / deletion at the end), else "interior"."""
    if a == b or abs(len(a) - len(b)) > 1:
        return None
    if len(a) == len(b):
        d = [i for i in range(len(a)) if a[i] != b[i]]
        if len(d) == 1:
            return "end" if d[0] == len(a) - 1 else "interior"
        if len(d) == 2 and d[1] == d[0] + 1 and a[d[0]] == b[d[1]] and a[d[1]] == b[d[0]]:
            return "interior"  # adjacent transposition: a typo
        return None
    s, t = (a, b) if len(a) < len(b) else (b, a)
    i = 0
    while i < len(s) and s[i] == t[i]:
        i += 1
    if s[i:] != t[i + 1:]:
        return None
    return "end" if i == len(s) else "interior"


def _generic(tok):
    return tok in GENERIC or tok.translate(_OCR) in _GENERIC_OCR or (
        len(tok) >= 5 and any(_one_edit(tok, g) for g in _GENERIC_LONG))


def _joined(tok, all_other, own):
    """tok appears inside a joined token of the other name whose remainder holds another token of its own name
    ("tavellbakery" = "tavell" + "bakery"), unlike a morph ("quorinex" = "quorin" + "ex")."""
    for x in all_other:
        if len(x) > len(tok) and tok in x:
            rest = x.replace(tok, "", 1)
            if any(len(t) >= 3 and t != tok and t in rest for t in own):
                return True
    return False


def _kind(tok, others, all_other, own):
    """Best relation of a token to the other name's distinctive tokens (all_other: all its tokens, own: all tokens
    of tok's own name, for joined forms like "#tavellbakery" or "tavellbakery.com")."""
    if tok in others:
        return "exact"
    o = tok.translate(_OCR)
    if any(o == x.translate(_OCR) for x in others):
        return "ocr"
    if len(tok) >= 4 and _joined(tok, all_other, own):
        return "exact"
    if len(tok) >= 4:
        edits = [_one_edit(tok, x) for x in others if len(x) >= 4]
        if "interior" in edits:
            return "typo"
        if "end" in edits or any(len(x) >= 4 and x[:4] == tok[:4] for x in others):
            return "morph"  # same stem, different ending
    return "none"


def _numbers(s):
    return [t.lstrip("0") or "0" for t in _tokens(s) if t.isdigit()]


def pair_features(s1_name, cand_name, s1_addr, cand_addr):
    sa, sb = _tokens(s1_name), _tokens(cand_name)
    da = [t for t in dict.fromkeys(sa) if not t.isdigit() and not _generic(t)]
    db = [t for t in dict.fromkeys(sb) if not t.isdigit() and not _generic(t)]
    kinds = [_kind(t, db, sb, sa) for t in da]
    n = max(len(da), 1)
    extra = [t for t in db if _kind(t, da, sa, sb) == "none"]
    na, nb = _numbers(s1_addr), _numbers(cand_addr)
    extend = any(x != y and (x.startswith(y) or y.startswith(x) or x.endswith(y) or y.endswith(x))
                 for x in na for y in nb)
    return [len(da), kinds.count("exact") / n, kinds.count("ocr") / n, kinds.count("typo") / n,
            kinds.count("morph") / n, kinds.count("none") / n, len(extra),
            float(kinds.count("none") + kinds.count("morph") > 0 and len(extra) > 0),
            len([t for t in dict.fromkeys(sb) if _generic(t) and t not in sa]),
            (kinds.count("exact") + kinds.count("ocr") + kinds.count("typo")) / n,
            float(bool(isinstance(cand_addr, str) and HNO.match(cand_addr))
                  and not (isinstance(s1_addr, str) and HNO.match(s1_addr))),
            float(extend), float(bool(set(na) & set(nb)))]


def decoy_features(pairs, records):
    """DataFrame(FEATURES) for pairs (s1_id, cand_id) with records (entity_id, business_name, business_address)."""
    rec = records.set_index("entity_id")
    cols = [rec[c].reindex(pairs[k]).tolist() for k, c in (("s1_id", "business_name"), ("cand_id", "business_name"),
                                                            ("s1_id", "business_address"),
                                                            ("cand_id", "business_address"))]
    cache = {}
    out = np.empty((len(pairs), len(FEATURES)), dtype=np.float32)
    for i, key in enumerate(zip(*cols)):
        if key not in cache:
            cache[key] = pair_features(*key)
        out[i] = cache[key]
    return pd.DataFrame(out, columns=FEATURES, index=pairs.index)

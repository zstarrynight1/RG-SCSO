"""One-shot appearance-order reference renumber for RG-SCSO_IEEE_draft.docx.

Confirmed by user: renumber the docx so [n] follows FIRST-APPEARANCE order in the
docx itself (true IEEE), and slot in 3 new refs. This diverges docx numbering from
the old .tex/PDF (which followed tex CITE_ORDER) — accepted.

Does NOT touch any OMML/equation. Body cites are plain text within single runs
(verified). Reference entries are manual "[n]" labels. Backup already at
/tmp/RG-SCSO_IEEE_draft.PRE_RENUMBER.docx.
"""
import copy
import re

from docx import Document
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn

SRC = "RG-SCSO_IEEE_draft.docx"

# old -> new for the 19 existing refs (derived from appearance sequence
# 1,2,3,4,5,6,7,8,12,[Islam],[Teng],14,15,16,17,13,[Ludwig],9,10,11,18,19)
REMAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
         9: 18, 10: 19, 11: 20, 12: 9, 13: 16, 14: 12,
         15: 13, 16: 14, 17: 15, 18: 21, 19: 22}

# new reference LIST order: (kind, key). old N -> entry object; new -> build it.
NEW_ISLAM = ('new', 'islam')
NEW_TENG = ('new', 'teng')
NEW_LUDWIG = ('new', 'ludwig')
ORDER = [('old', 1), ('old', 2), ('old', 3), ('old', 4), ('old', 5), ('old', 6),
         ('old', 7), ('old', 8), ('old', 12), NEW_ISLAM, NEW_TENG, ('old', 14),
         ('old', 15), ('old', 16), ('old', 17), ('old', 13), NEW_LUDWIG,
         ('old', 9), ('old', 10), ('old', 11), ('old', 18), ('old', 19)]

RPR_PLAIN = ('<w:rPr><w:rFonts w:eastAsia="Times New Roman" '
             'w:cs="Times New Roman"/><w:sz w:val="18"/></w:rPr>')
RPR_IT = ('<w:rPr><w:rFonts w:eastAsia="Times New Roman" '
          'w:cs="Times New Roman"/><w:i/><w:sz w:val="18"/></w:rPr>')

NEW_REFS = {
    'islam': ('[10] M. J. Islam, X. Li, and Y. Mei, “A time-varying transfer '
              'function for balancing the exploration and exploitation ability of a '
              'binary PSO,” ', 'Applied Soft Computing', ', vol. 59, pp. 182–196, 2017.'),
    'teng': ('[11] X. Teng, H. Dong, and X. Zhou, “Adaptive feature selection using '
             'v-shaped binary particle swarm optimization,” ', 'PLoS ONE',
             ', vol. 12, no. 3, art. e0173907, 2017.'),
    'ludwig': ('[17] S. A. Ludwig, “Guided Particle Swarm Optimization for Feature '
               'Selection: Application to Cancer Genome Data,” ', 'Algorithms',
               ', vol. 18, no. 4, art. 220, 2025.'),
}

PROSE_ISLAM = (
    "A parallel line of work makes the transfer itself adaptive. Islam et al. [10] "
    "let the V-shaped function vary with iteration so the swarm explores early and "
    "exploits late, and Teng et al. [11] adapt the flip behaviour of a V-shaped "
    "binary PSO to the search state; both report gains over static transfers. Yet "
    "the adaptation is a function of time or global search state alone: the same "
    "curve is applied to every coordinate at a given iteration, so a decisive "
    "feature and a noise probe are still binarized identically. The missing degree "
    "of freedom is per-feature, and it is exactly the one RG-SCSO supplies.")
PROSE_LUDWIG = (
    "A related and growing theme is knowledge-guided optimization, in which filter "
    "statistics steer the metaheuristic rather than merely rank features afterwards. "
    "Ludwig [17], for instance, seeds and biases a particle swarm with three filter "
    "criteria for cancer-genome selection, and prior-information schemes more broadly "
    "inject relevance at initialization or as an auxiliary objective. RG-SCSO shares "
    "the conviction that problem knowledge belongs inside the search, but places it "
    "at a more decisive point: not in the initial population or the objective, but "
    "in the binarization operator itself, where the continuous-to-binary bottleneck "
    "of Section III-B would otherwise discard it.")

CITE_RE = re.compile(r'\[(\d+)\]')


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def count_eq(doc):
    body = doc.element.body
    return (len(body.findall('.//' + qn('m:oMath'))),
            len(body.findall('.//' + qn('m:oMathPara'))))


def main():
    doc = Document(SRC)
    ps = doc.paragraphs
    eq_before = count_eq(doc)

    ref_start = next(i for i, p in enumerate(ps)
                     if p.style.name.startswith("Heading 1")
                     and p.text.strip() == "References") + 1
    ref_paras = [p for p in ps[ref_start:] if re.match(r'\s*\[\d+\]', p.text)]
    assert len(ref_paras) == 19, f"expected 19 refs, got {len(ref_paras)}"

    # locate the two prose insertion-point paragraphs by content (robust to index)
    p_transfer = next(p for p in ps if p.text.startswith(
        "The transfer function itself has been studied"))
    p_priors = next(p for p in ps if p.text.startswith(
        "Two further ingredients inform our design"))

    # template for prose paragraphs = a Normal body paragraph
    src_pPr = p_transfer._p.find(qn('w:pPr'))
    src_rPr = p_transfer.runs[0]._r.find(qn('w:rPr'))

    # ---- STEP 1: remap body cites (everything before References) ----
    remapped = 0
    for p in ps[:ref_start]:
        for r in p.runs:
            if '[' in r.text:
                nt = CITE_RE.sub(lambda m: f'[{REMAP[int(m.group(1))]}]', r.text)
                if nt != r.text:
                    r.text = nt
                    remapped += 1
    print("STEP1 body runs remapped:", remapped)

    # ---- STEP 2: insert 2 prose paragraphs ----
    def make_prose(after_p, text):
        newp = parse_xml(f'<w:p {nsdecls("w")}></w:p>')
        newp.append(copy.deepcopy(src_pPr))
        r = parse_xml(f'<w:r {nsdecls("w")}></w:r>')
        if src_rPr is not None:
            r.append(copy.deepcopy(src_rPr))
        r.append(parse_xml(
            f'<w:t {nsdecls("w")} xml:space="preserve">{esc(text)}</w:t>'))
        newp.append(r)
        after_p._p.addnext(newp)

    make_prose(p_transfer, PROSE_ISLAM)   # after II-B Transfer Functions
    make_prose(p_priors, PROSE_LUDWIG)    # after II-D Relevance Priors
    print("STEP2 inserted 2 prose paragraphs")

    # ---- STEP 3: reorder + relabel reference list, insert 3 new entries ----
    old_ps = [p._p for p in ref_paras]          # index 0..18 == old ref 1..19
    parent = old_ps[0].getparent()
    anchor = old_ps[0].getprevious()            # the "References" heading _p
    for op in old_ps:
        parent.remove(op)

    def relabel(_p, newlabel):
        t = _p.find('.//' + qn('w:t'))
        t.text = re.sub(r'^\s*\[\d+\]', f'[{newlabel}]', t.text, count=1)

    def make_ref(key):
        plain0, venue, plain2 = NEW_REFS[key]
        xml = (f'<w:p {nsdecls("w")}><w:pPr><w:jc w:val="both"/></w:pPr>'
               f'<w:r>{RPR_PLAIN}<w:t xml:space="preserve">{esc(plain0)}</w:t></w:r>'
               f'<w:r>{RPR_IT}<w:t xml:space="preserve">{esc(venue)}</w:t></w:r>'
               f'<w:r>{RPR_PLAIN}<w:t xml:space="preserve">{esc(plain2)}</w:t></w:r></w:p>')
        return parse_xml(xml)

    ordered = []
    for pos, (kind, key) in enumerate(ORDER, start=1):
        if kind == 'old':
            _p = old_ps[key - 1]
            relabel(_p, pos)
            ordered.append(_p)
        else:
            ordered.append(make_ref(key))

    prev = anchor
    for _p in ordered:
        prev.addnext(_p)
        prev = _p
    print("STEP3 reference list rebuilt:", len(ordered), "entries")

    # ---- verify equations untouched ----
    eq_after = count_eq(doc)
    print("equations before:", eq_before, "after:", eq_after,
          "-> DELTA", (eq_after[0] - eq_before[0], eq_after[1] - eq_before[1]))
    assert eq_after == eq_before, "EQUATION COUNT CHANGED — ABORT"

    doc.save(SRC)
    print("SAVED", SRC)


if __name__ == "__main__":
    main()

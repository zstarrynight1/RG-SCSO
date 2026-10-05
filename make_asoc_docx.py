"""
make_asoc_docx.py — Applied Soft Computing editable Word deliverables.

Produces (Elsevier submits Highlights + Graphical Abstract as SEPARATE files):
  RG-SCSO_ASOC.docx               — main manuscript (identical journal-agnostic body
                                     as RG-SCSO_IJCS.docx; copied, content verified)
  RG-SCSO_ASOC_Highlights.docx    — Highlights (5 bullets) + Graphical Abstract image
  RG-SCSO_ASOC_Cover_Letter.docx  — cover letter to the Editor-in-Chief

Run after build_paper_ijcs_docx.py (so RG-SCSO_IJCS.docx is current) and make_asoc.py.
"""
from __future__ import annotations

import os
import shutil

from docx import Document
from docx.shared import Pt, Inches

TITLE = ("Relevance-Guided Binarization for Parsimonious Feature Selection "
         "with Sand Cat Swarm Optimization")
AUTHORS = "Bui Quang Huy, Duong Minh Son"
AFFIL = "Dong A University, Da Nang, Vietnam"
EMAIL = "huybq@donga.edu.vn"

HIGHLIGHTS = [
    "Transfer-function binarization discards swarm search detail; we name it washout",
    "RG-SCSO injects per-feature MI relevance into the binarization decision of SCSO",
    "Best mean accuracy and second-smallest subsets on 18 datasets under a fixed budget",
    "Ablation localizes the gain to the binarization step, not init or the objective",
    "Parsimony gains persist across KNN/SVM/RF wrappers; limited on p>>n gene-expression",
]


def main_manuscript() -> None:
    """Identical body to the master docx (verified journal-agnostic: no IJCS/Springer text)."""
    shutil.copyfile("RG-SCSO_master.docx", "RG-SCSO_ASOC.docx")
    print("Wrote RG-SCSO_ASOC.docx (copy of master docx — identical body)")


def highlights_doc() -> None:
    doc = Document()
    doc.add_heading("Highlights", level=1)
    p = doc.add_paragraph()
    p.add_run(TITLE).italic = True
    doc.add_paragraph(AUTHORS)
    for h in HIGHLIGHTS:
        doc.add_paragraph(h, style="List Bullet")
    doc.add_heading("Graphical Abstract", level=1)
    if os.path.exists("figures/graphical_abstract.png"):
        doc.add_picture("figures/graphical_abstract.png", width=Inches(2.6))
    doc.save("RG-SCSO_ASOC_Highlights.docx")
    print("Wrote RG-SCSO_ASOC_Highlights.docx (5 highlights + graphical abstract)")


COVER = [
    ("Dear Editor-in-Chief, Applied Soft Computing,", None),
    ("We submit our manuscript “Relevance-Guided Binarization for Parsimonious "
     "Feature Selection with Sand Cat Swarm Optimization” for consideration as a "
     "research article.", None),
    ("Motivation and fit. Applied Soft Computing is a natural home for soft-computing "
     "feature selection. Our work addresses a structural weakness shared by virtually "
     "all swarm-based binary feature selectors: the fixed, feature-agnostic transfer "
     "function that converts continuous positions into bits discards the very "
     "adjustments the continuous search makes — a failure mode we identify and name "
     "washout, and demonstrate empirically (four well-motivated continuous-space SCSO "
     "enhancements fail to beat base SCSO under an identical feature-selection protocol).", None),
    ("Contribution. RG-SCSO replaces the fixed transfer with a per-feature, "
     "relevance-modulated binarization that injects a mutual-information prior directly "
     "at the binarization decision. A controlled injection-point ablation (V0–V4) "
     "localizes the benefit specifically to that decision stage, distinguishing our "
     "mechanism from relevance-at-initialization or relevance-in-objective approaches in "
     "the SCSO-family literature (a full-text novelty audit is included).", None),
    ("Evidence and rigor. We evaluate on 18 datasets against six metaheuristics, five "
     "classical selectors, and a deep baseline, under a nominal 15,000-evaluation budget "
     "matched across all methods and a leakage-controlled protocol (relevance prior, "
     "search, and cross-validated fitness fit on the training partition only). We report "
     "a full statistical treatment (Friedman, Wilcoxon with Holm correction, effect "
     "sizes, bootstrap confidence intervals), subset-stability analysis, and a causal "
     "shuffled-prior control. All code and raw results are released for reproducibility.", None),
    ("Honest scope. We state our boundaries openly rather than hide them: the advantage "
     "is parsimony-dominant, is bounded on extreme p>>n gene-expression data (where a "
     "classical LASSO baseline is preferable), and we make no cross-optimizer "
     "generalization claim.", None),
    ("Declarations. The manuscript is original, is not under consideration elsewhere, and "
     "all authors approve its submission. The authors declare no competing interests. A "
     "generative AI assistant was used for language and readability editing; all "
     "scientific content and results are the authors’ own and were verified by the "
     "authors.", None),
    ("We believe this work fits the scope of Applied Soft Computing and will interest its "
     "readership. Thank you for your consideration.", None),
    ("Sincerely,", None),
    (f"{AUTHORS}", None),
    (f"{AFFIL}", None),
    (EMAIL, None),
]


def cover_letter() -> None:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)
    for text, _ in COVER:
        doc.add_paragraph(text)
    doc.save("RG-SCSO_ASOC_Cover_Letter.docx")
    words = sum(len(t.split()) for t, _ in COVER)
    print(f"Wrote RG-SCSO_ASOC_Cover_Letter.docx (~{words} words)")


if __name__ == "__main__":
    main_manuscript()
    highlights_doc()
    cover_letter()

# Licensing position

## This code

Copyright (c) 2026 Shuvam. **All rights reserved** pending an explicit licence decision.

This is a deliberate placeholder, not an oversight. No open-source licence has been
chosen for this work yet, and defaulting to one silently would be the wrong call: a
permissive grant cannot be retracted once published, whereas "all rights reserved" can
be relaxed at any time. Internal use within this repository is unaffected.

**Before sharing outside the organisation, replace this section with a real licence.**
A partner's legal review will otherwise block integration by default — unlicensed code
is legally unusable, regardless of quality.

Typical choices: MIT or Apache-2.0 for maximum adoption (Apache-2.0 additionally grants
patent rights, which matters for a clinical/ML context); AGPL-3.0 if derivative works
must stay open.

## Third-party obligations

None of the datasets are included in this bundle, and `.gitignore` excludes dataset
payloads repository-wide. The obligations below attach to *data you obtain yourself*,
and potentially to models fitted on it.

| Source | Licence | Obligation |
|---|---|---|
| **MUSIC** (heart failure, PhysioNet) | Open Data Commons ODbL v1.0 | Attribution **and share-alike** on any redistributed derivative database |
| **PTB-XL** (12-lead ECG, PhysioNet) | Creative Commons Attribution | Attribution |
| **PGS Catalog** scoring files (e.g. PGS000018) | Open; cite the score and its publication | Cite `Inouye M et al. J Am Coll Cardiol (2018)` for metaGRS_CAD, per EBI terms |
| **CADICA** (angiography, not yet obtained) | Check at source before use | — |

### The ODbL question, flagged rather than answered

MUSIC is ODbL, which carries **share-alike**. Whether a model fitted on ODbL data counts
as a "Produced Work" (unencumbered) or a "Derivative Database" (share-alike applies) is
a genuine legal question that depends on what the artifact embeds. It is not one that
should be settled by an engineer's judgement.

Relevant facts for whoever does decide:

- The trained artifacts embed **aggregate statistics only** — imputer medians and a
  97-value reference vector — not patient rows.
- **No trained artifacts are included in this bundle**, partly for this reason.
- `results/*.json` contain feature names and model metrics; they were checked and carry
  **no patient-level rows**.

Get this reviewed before distributing any artifact fitted on MUSIC outside the
organisation.

## Attribution for reuse

Third-party components retain their own licences: Qiskit and Qiskit Machine Learning
(Apache-2.0), scikit-learn (BSD-3-Clause), PyTorch (BSD-style), torchvision ImageNet
ResNet18 weights (BSD-style). The dressed-circuit architecture in `hybrid_qnn.py`
follows Mari et al., *Transfer learning in hybrid classical-quantum neural networks*
(2020); the vesselness filter in `angiography.py` follows Frangi et al., *Multiscale
vessel enhancement filtering* (MICCAI 1998). Both are cited in their module docstrings.

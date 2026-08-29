# Session Handoff — Neurological Conditions Platform

**Date**: 2026-08-29
**Purpose**: Resume this work in a fresh chat without needing the original conversation history.

## The original directive (verbatim, from `/goal`)

> use the neurological conditions spec defined with all the models and start one by one according
> to priority. use as much public data you can find exhaustively and only keep as much useful for
> our cases. then find the best possible preprocessinf+hybrid classical + quantum models
> architecture fit for the task, then build the models in the most efficient way and test and
> optimise until reached an acceptable mark. doneload anyb missing stuff. use opus for thinkning
> and haiku for execution subagents. we need to optimise token usage. use ponytail skill as well.

Additional standing instruction given mid-session: when reusing existing code/tools, close any
gaps found rather than working around them — extend the reused thing properly, don't leave gaps
"just because the existing tool had that gap."

## IMPORTANT — about the `/goal` Stop hook

This session had an active `/goal` running, which installs a Stop hook that blocks the session
from ending and injects an automated critique every time the assistant tries to stop, until the
goal condition is judged satisfied. **If you don't want that behavior in the new session, do not
re-run `/goal` with this same directive** — just reference this doc and continue normally. If a
`/goal` is (or becomes) active and you want to cancel it, the only real command is `/goal clear`
(not `/goal stop`/`/goal off`/etc. — those don't exist, verified against the CLI binary directly).

## Where things stand — scorecard

| Condition | Status | Best result | Files |
|---|---|---|---|
| **P1 Stroke** (clinical, tabular) | ✅ Operational reference | Classical balanced accuracy **0.548** (full 5110-row cohort). QSVC failed real-gain gate; also failed a full circuit-ansatz/entanglement architecture sweep (10-seed validated). | `research/P1-stroke-reuse-record.md`, models `stroke-clinical-risk-tabular(-classical).json` |
| **P4 Seizure** (EEG window risk) | ✅ Operational reference | Classical balanced accuracy **0.875** (2700-window single-patient CHB-MIT cohort). QSVC failed real-gain gate. | `research/P4-seizure-reuse-record.md`, models `seizure-window-risk-tabular(-classical).json` |
| **P5 Alzheimer's** (clinical, same-visit proxy) | ✅ Operational reference | Classical balanced accuracy **0.823** (235-row OASIS-1 cohort). QSVC failed. Scoped honestly as same-visit dementia-association proxy, NOT longitudinal progression (no OASIS-1 data supports that). | `research/P5-alzheimers-reuse-record.md`, models `alzheimers-clinical-risk-tabular(-classical).json` |
| **P6 Parkinson's** (voice, diagnosed-vs-healthy) | ✅ Operational reference | Classical balanced accuracy **0.734** (195-row UCI voice cohort). QSVC failed. Scoped honestly as diagnosed-vs-healthy, NOT prodromal/PPMI-style risk. | `research/P6-parkinsons-reuse-record.md`, models `parkinsons-voice-risk-tabular(-classical).json` |
| **P3 Glioma** (MGMT-methylation from mpMRI radiomics) | ⚠️ Tested, conclusively negative — registered `not available` | Classical **0.474**, QSVC **0.451** on full 256-case UPenn-GBM cohort (CC BY 4.0, TCIA, no DUA). A 24-configuration architecture sweep (ANOVA/PCA × 4/6/8 qubits) found everything in [0.44, 0.55] — statistically at chance. This is a completed, honest negative result, not an unattempted gap. The condition's PRIMARY task (tumor segmentation/grade characterization) has no dataset and was never attempted (see below). | `research/P3-glioma-reuse-record.md`, models `glioma-mgmt-radiomics-tabular(-classical).json` |
| **P2 ICH** | ❌ Blocked — zero models | No lawful, non-DUA, license-verified dataset exists. Checked exhaustively (see below). One concrete unblock path is in progress — see "Next steps" below. | `research/P2-ich-reuse-record.md` |

**Platform verification**: registry loads clean (`load_registry()`), all 24 tests in
`backend/tests/test_platform.py` pass, `backend/tests/test_smoke.py` passes — as of last check.
Re-run these three before trusting this doc's "current" status if much time has passed:
```
cd backend
./.venv/bin/python -c "from qhealth_qml.platform.registry import load_registry; load_registry(); print('ok')"
./.venv/bin/python -m pytest tests/test_platform.py -q
./.venv/bin/python tests/test_smoke.py
```

## P2 (ICH) — exhaustive search already done, don't repeat it

Checked and rejected, each independently verified (see `research/P2-ich-reuse-record.md` for full
detail):
- **PhysioNet CT-ICH (Hssayeni)** — requires PhysioNet's Restricted Health Data License (DUA). Blocked.
- **Kaggle "mirrors" of that dataset** (`vbookshelf/computed-tomography-ct-images`,
  `coderrkj/processed-ct-ich-dataset-images`) — confirmed re-uploads of the same PhysioNet-gated
  data (verified via direct page fetch, found "PhysioNet"/"Hssayeni" in content). Using them would
  be license laundering. Rejected.
- **TCIA** — searched the full live collection catalog (156 collections); zero ICH/hemorrhage/
  stroke-relevant collections exist. (This same method found `UPENN-GBM` for P3 — it does not work
  for P2.)
- **CQ500 (Qure.ai)** — both known hosting channels are dead (qure.ai/dataset → 404;
  academictorrents mirror → "Collection not found"). Confirmed via live HTTP checks, not memory.
- **RSNA ICH Detection (Kaggle competition)** — NOT DUA-gated (competition-rules-only), but the
  full dataset is ~450GB. This is the one lead that's actually about *feasibility*, not *license* —
  see next section.

### The one live lead: RSNA ICH subset via Kaggle + external storage

User's idea (sound, in progress): download a **stratified, class-balanced sample** of RSNA ICH
slices (not the full 450GB) using the small `stage_2_train.csv` labels file to pick the sample,
sized to whatever disk is actually available. Same pattern as P4's feature-extraction-from-signal
approach — reuse the existing `qhealth_qml` tabular engine, no new imaging deep-learning code.

**Blocked on two things, both requiring the user, neither done yet as of this doc:**
1. **Kaggle account hasn't joined the competition.** `kaggle competitions list -s rsna` showed
   `userHasEntered: False` for `rsna-intracranial-hemorrhage-detection`. Fix: visit
   `https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/rules` and click
   "I Understand and Accept" (free, instant, not a DUA). Re-check with:
   `kaggle competitions files rsna-intracranial-hemorrhage-detection`
2. **No external SSD is mounted.** Root disk (`/dev/nvme0n1p4`) had only 3.2GB free at last check.
   `sda` (476.9GB) exists as a raw block device, not mounted. `/mnt/ssd` and `/mnt/brainage_ssd` are
   pre-existing empty mount-point directories (no sudo access available to mount it directly in
   this session). User needs to run, in their own terminal or via `!` in a Claude Code session:
   ```
   sudo blkid /dev/sda2          # check filesystem type first
   sudo mount /dev/sda2 /mnt/ssd
   sudo chown $(whoami):$(whoami) /mnt/ssd
   ```
   (Exact device/partition may need adjusting — check `sudo fdisk -l /dev/sda` if `sda2` isn't right.)

**Once both are done**: pull `stage_2_train.csv`, pick a balanced sample of slices, download just
those DICOMs, extract tabular features, run through the same `execution.benchmark_model()` pipeline
used for every other condition (see `backend/src/qhealth_qml/platform/execution.py` and the P3
`prepare.py` in `backend/data/p3_glioma_upenn/` as the template for the merge-then-benchmark pattern).

## The acceptance-mark question — drafted, not yet confirmed by the user

Spec text (`specs/001-neurological-conditions/spec.md` line 507): *"No minimum accuracy threshold
is shared across all conditions. Promotion criteria are declared per model."* This delegates the
threshold decision to the implementer, not the user — but the user has not explicitly approved the
specific bar declared.

**`specs/001-neurological-conditions/ACCEPTANCE-CRITERIA.md`** documents the bar actually applied
throughout this build:
1. **Baseline viability gate**: a model is only `available`/`operational_reference` if its
   production-rigor balanced accuracy is clearly above chance (0.5). (This is why P3's models are
   `not available` — they didn't clear this.)
2. **Real-gain gate** (spec FR-039/040, not invented by the agent): QML only replaces classical if
   the paired CI on the balanced-accuracy delta excludes zero in QML's favor. (No condition's QML
   candidate has passed this yet.)

**Status**: the user asked to clarify before approving/rejecting this, but never followed up with
the actual clarifying question. **This is the first thing to resolve in a fresh session** — ask the
user directly: "What did you want to clarify about the acceptance criteria?" or, if they've decided,
just get an explicit approve/adjust/reject.

## Codex session conflict (probably resolved, worth one check)

Earlier in this session, 4 concurrent `codex resume --dangerously-bypass-approvals-and-sandbox`
processes were independently editing this same repo, causing repeated unauthorized file
reappearance (an EHR subsystem nobody asked for, contaminating `dashboard.py`, `pyproject.toml`,
`platform/__init__.py`). This was cleaned up multiple times and had stabilized (0 contamination) for
many consecutive checks before this doc was written. Worth one verification in a fresh session:
```
grep -c "ehr" backend/src/qhealth_qml/dashboard.py backend/pyproject.toml
```
Both should read `0`. If not, investigate before doing further work on those files — check
`ps aux | grep codex` for any still-running sessions.

## Recommended next steps, in order

1. Re-run the three verification commands above to confirm nothing has drifted.
2. Ask the user directly what they wanted clarified about `ACCEPTANCE-CRITERIA.md` (don't re-draft
   it blind — the current draft is reasonable and spec-grounded, just needs a yes/no/adjust).
3. Check whether the user has accepted the RSNA Kaggle competition rules and mounted external
   storage. If yes to both, proceed with the P2 RSNA-subset build (see plan above).
4. If the user explicitly says P2 is permanently out of scope and/or approves the acceptance
   criteria as final, the project is done: 4 of 6 conditions have real, tested, honestly-evaluated
   operational models; P3 has a conclusive negative result; P2 is documented as blocked pending
   data access that only the user can provide.

Do not re-run the P2 exhaustive search (PhysioNet/Kaggle-mirrors/TCIA/CQ500) again — it's been done
twice with different methods and the answer won't change without new information from the user.

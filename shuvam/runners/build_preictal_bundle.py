"""Package the P4 pre-ictal (early-detection) model as a deployable bundle.

Encoder-free: this model consumes 14 engineered EEG band-power/dynamics features directly, so
`encoder_kind="none"` and the feature vector *is* the representation. Everything else — persisted
scalers, validation-selected threshold, saved score normalisation, OOD guard — is identical to the
imaging bundles.

Trained on all patients and thresholded on a held-out patient. The honest performance number is
the leave-one-patient-out result from `run_preictal.py`, NOT a within-patient split, and it is
recorded in the artifact's limitations because it is at chance.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from qhealth_qml.chbmit_preictal import FEATURE_NAMES, PREICTAL_END_S
from qhealth_qml.experiment import build_quantum_context, classification_metrics, select_threshold
from qhealth_qml.serving import SERVING_SCHEMA_VERSION, InferenceBundle, save_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/unichronic/neutral_data/p4_eeg")
    parser.add_argument("--num-qubits", type=int, default=4)
    parser.add_argument("--lopo-report", default="runtime/p4_preictal_lopo.json")
    parser.add_argument("--out", default="runtime/bundles/seizure-preictal-eeg.pkl")
    args = parser.parse_args()

    root = Path(args.data_root)
    X = np.load(root / "X.npy")
    y = np.load(root / "y.npy")
    groups = np.load(root / "groups.npy", allow_pickle=True).astype(str)
    patients = sorted(set(groups.tolist()))
    print(f"{len(y)} windows, {int(y.sum())} pre-ictal, patients {patients}", flush=True)

    # hold one patient out for thresholding so the operating point is never chosen on data the
    # model trained on; the remaining patients form the training pool
    threshold_patient = patients[-1]
    va = groups == threshold_patient
    tr = ~va

    standardizer = StandardScaler().fit(X[tr])
    Xtr, Xva = standardizer.transform(X[tr]), standardizer.transform(X[va])
    reducer = PCA(n_components=min(args.num_qubits, Xtr.shape[1])).fit(Xtr)
    Qtr, Qva = reducer.transform(Xtr), reducer.transform(Xva)
    angle = MinMaxScaler(feature_range=(-np.pi / 2 * 0.2, np.pi / 2 * 0.2)).fit(Qtr)
    Atr, Ava = angle.transform(Qtr), angle.transform(Qva)

    context = build_quantum_context(
        mode="statevector", n_qubits=args.num_qubits, shots=512, seed=7,
        feature_map_reps=1, feature_map_entanglement="linear",
    )
    from qiskit_machine_learning.algorithms import QSVC

    head = QSVC(quantum_kernel=context.kernel, C=5.0)
    head.fit(Atr, y[tr])

    raw_validation = head.decision_function(Ava)
    score_offset = float(raw_validation.min())
    score_scale = float(max(raw_validation.max() - raw_validation.min(), 1e-9))
    validation_score = np.clip((raw_validation - score_offset) / score_scale, 0.0, 1.0)
    picked = select_threshold(
        y[va], validation_score, policy="target_sensitivity", target_sensitivity=0.8
    )
    validation_metrics = classification_metrics(
        y[va], (validation_score >= picked["threshold"]).astype(int),
        validation_score, probability_score=True,
    )
    print(f"threshold patient {threshold_patient}: BA={validation_metrics['balanced_accuracy']:.4f}", flush=True)

    lopo = {}
    report_path = Path(args.lopo_report)
    if report_path.exists():
        lopo = json.loads(report_path.read_text()).get("summary", {}).get("qsvc", {})

    bundle = InferenceBundle(
        schema_version=SERVING_SCHEMA_VERSION,
        model_id="seizure-preictal-eeg",
        condition="epilepsy (seizure prediction)",
        temporal_framing="prediction",
        positive_label="preictal_seizure_imminent",
        negative_label="interictal_baseline",
        channel_names=list(FEATURE_NAMES),
        input_grid=[len(FEATURE_NAMES)],
        encoder_state={},
        encoder_kind="none",
        encoder_config={"features": len(FEATURE_NAMES), "representation": "engineered EEG band-power + dynamics"},
        head_kind="qsvc",
        head=head,
        standardizer=standardizer,
        reducer=reducer,
        angle_scaler=angle,
        threshold=float(picked["threshold"]),
        threshold_policy="target_sensitivity@0.8 (held-out patient only)",
        quantum_config={
            "n_qubits": args.num_qubits, "feature_map": "ZZFeatureMap", "feature_map_reps": 1,
            "entanglement": "linear", "C": 5.0, "angle_scale": 0.2, "backend": "statevector",
        },
        training_provenance={
            "dataset": "CHB-MIT Scalp EEG (PhysioNet, ODC-By)",
            "modality": "signal",
            "cohort_size": int(len(y)),
            "patients": patients,
            "positives": int(y.sum()),
            "lead_time_minutes": PREICTAL_END_S / 60,
            "task": "pre-ictal prediction: is a seizure coming in the next ~30 minutes",
            "preictal_definition": "35-5 min before annotated onset; 5 min seizure-prediction-horizon gap",
            "held_out_performance": {
                "protocol": "leave-one-patient-out",
                "balanced_accuracy": lopo.get("balanced_accuracy_mean"),
                "balanced_accuracy_std": lopo.get("balanced_accuracy_std"),
                "auc": lopo.get("auc_mean"),
                "per_patient": lopo.get("per_patient_ba"),
            },
            "limitations": [
                "RESEARCH USE ONLY, NOT A MEDICAL DEVICE",
                "PERFORMS AT CHANCE patient-independently: leave-one-patient-out balanced accuracy "
                "0.505 +/- 0.257, AUC 0.468 - NO demonstrated ability to predict seizures in an "
                "unseen patient, and must not be used for warning or alerting",
                "Per-patient results range 0.26 to 0.92; two of four held-out patients scored BELOW "
                "chance with inverted ranking, indicating pre-ictal signatures are patient-specific",
                "Only 4 patients (CHB-MIT has 24); patient-independent literature SOTA is AUC ~0.81 "
                "using 22 patients and spectrogram inputs rather than these 14 band-power features",
                "Pre-ictal windows come from seizure-bearing recordings and interictal from "
                "seizure-free ones, so session-level confounds are not controlled",
                "Input must be 14 features in the documented order, computed from 30 s of "
                "multi-channel scalp EEG at 256 Hz",
                "Calibration not assessed",
            ],
        },
        input_stats={"mean": float(X[tr].mean()), "std": float(X[tr].std())},
        score_offset=score_offset,
        score_scale=score_scale,
    )
    path = save_bundle(bundle, args.out)
    print(f"wrote {path}\nwrote {path}.manifest.json", flush=True)


if __name__ == "__main__":
    main()

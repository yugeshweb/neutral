"""Generates the synthetic signal/imaging fixtures for
`test_signal_imaging_adapters.py`. Run with:

    ./.venv/Scripts/python tests/pipeline/fixtures/generate_signal_imaging_fixtures.py

Regenerates deterministically (fixed seed) - committed fixtures should
match a fresh run byte-for-byte in shape/content, only float encoding may
differ slightly across library versions."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).parent
RNG = np.random.default_rng(20260107)

N_COHORT = 24


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


# ---------------------------------------------------------------------------
# ECG (wfdb)
# ---------------------------------------------------------------------------

def generate_ecg() -> None:
    import wfdb

    cohort_dir = FIXTURES / "ecg_cohort"
    _clean_dir(cohort_dir)
    # All 12 standard leads - qhealth_qml.ecg.extract_ecg_features (the
    # real, teammate-built extractor pulled in from origin/main's
    # shuvam/src/qhealth_qml/ecg.py) requires exactly this set.
    channels = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    fs = 250
    duration_s = 2
    n_samples = fs * duration_s
    t = np.linspace(0, duration_s, n_samples, endpoint=False)

    for i in range(N_COHORT):
        label = int(i % 2 == 0)
        heart_rate_hz = 1.6 if label else 1.0  # simple, deterministic signal-label relationship
        base = np.sin(2 * np.pi * heart_rate_hz * t)
        sig = np.stack([base * (0.8 + 0.1 * c) + RNG.normal(0, 0.02, n_samples) for c in range(len(channels))])
        wfdb.wrsamp(
            record_name=f"REC{i:03d}",
            fs=fs, units=["mV"] * len(channels), sig_name=channels,
            p_signal=sig.T, fmt=["16"] * len(channels),
            write_dir=str(cohort_dir),
            comments=[f"Age: {40 + i}", f"Sex: {'Male' if i % 2 else 'Female'}", f"mi_event: {label}"],
        )

    corruption_dir = FIXTURES / "ecg_corruption"
    _clean_dir(corruption_dir)
    # missing a declared lead entirely
    sig2 = np.stack([np.sin(2 * np.pi * 1.2 * t), np.sin(2 * np.pi * 1.2 * t) * 0.9]) + RNG.normal(0, 0.02, (2, n_samples))
    wfdb.wrsamp(
        record_name="MISSING_LEAD", fs=fs, units=["mV", "mV"], sig_name=["I", "II"],
        p_signal=sig2.T, fmt=["16", "16"], write_dir=str(corruption_dir),
        comments=["Age: 70", "Sex: Male"],
    )
    # all 12 declared leads present, but one (V1) flatlined - tests
    # artefact detection specifically, distinct from MISSING_LEAD's
    # channel_missing case above.
    sig3 = np.stack(
        [np.sin(2 * np.pi * 1.2 * t) * (0.8 + 0.1 * c) + RNG.normal(0, 0.02, n_samples) for c in range(len(channels))]
    )
    v1_index = channels.index("V1")
    sig3[v1_index] = np.zeros(n_samples)
    wfdb.wrsamp(
        record_name="FLATLINE", fs=fs, units=["mV"] * len(channels), sig_name=channels,
        p_signal=sig3.T, fmt=["16"] * len(channels), write_dir=str(corruption_dir),
        comments=["Age: 65", "Sex: Female"],
    )
    print(f"ECG: {N_COHORT} cohort records + 2 corruption records written to {cohort_dir}, {corruption_dir}")


# ---------------------------------------------------------------------------
# EEG (edf)
# ---------------------------------------------------------------------------

def generate_eeg() -> None:
    from edfio import Edf, EdfSignal

    cohort_dir = FIXTURES / "eeg_cohort"
    _clean_dir(cohort_dir)
    channels = ["Fp1", "Fp2", "C3"]
    fs = 100.0
    duration_s = 2
    n_samples = int(fs * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)

    for i in range(N_COHORT):
        label = int(i % 3 == 0)  # ~1/3 positive rate, deterministic
        alpha_power = 3.0 if label else 1.0
        signals = []
        for c in range(len(channels)):
            data = alpha_power * np.sin(2 * np.pi * 10 * t + c) + RNG.normal(0, 0.5, n_samples)
            signals.append(EdfSignal(data.astype(np.float64), fs, label=channels[c], physical_dimension="uV"))
        edf = Edf(signals)
        edf.write(cohort_dir / f"sub{i:03d}__label{label}.edf")

    corruption_dir = FIXTURES / "eeg_corruption"
    _clean_dir(corruption_dir)
    # missing a declared channel
    signals_missing = [EdfSignal((np.sin(2 * np.pi * 10 * t) + RNG.normal(0, 0.5, n_samples)).astype(np.float64), fs, label="Fp1", physical_dimension="uV")]
    Edf(signals_missing).write(corruption_dir / "MISSING_CHANNEL__label0.edf")
    # a flatlined channel
    signals_flat = [
        EdfSignal((np.sin(2 * np.pi * 10 * t) + RNG.normal(0, 0.5, n_samples)).astype(np.float64), fs, label="Fp1", physical_dimension="uV"),
        EdfSignal(np.zeros(n_samples, dtype=np.float64), fs, label="Fp2", physical_dimension="uV"),
        EdfSignal((np.sin(2 * np.pi * 10 * t) + RNG.normal(0, 0.5, n_samples)).astype(np.float64), fs, label="C3", physical_dimension="uV"),
    ]
    Edf(signals_flat).write(corruption_dir / "FLATLINE__label1.edf")
    print(f"EEG: {N_COHORT} cohort files + 2 corruption files written to {cohort_dir}, {corruption_dir}")


# ---------------------------------------------------------------------------
# Gait (plain txt)
# ---------------------------------------------------------------------------

def generate_gait() -> None:
    cohort_dir = FIXTURES / "gait_cohort"
    _clean_dir(cohort_dir)
    channels = ["accel_x", "accel_y", "accel_z"]
    n_samples = 200

    for i in range(N_COHORT):
        label = int(i % 2 == 1)
        stride_hz = 2.2 if label else 1.6
        t = np.arange(n_samples) / 50.0
        rows = np.stack([np.sin(2 * np.pi * stride_hz * t + c) + RNG.normal(0, 0.05, n_samples) for c in range(len(channels))]).T
        lines = [",".join(channels)] + [",".join(f"{v:.5f}" for v in row) for row in rows]
        (cohort_dir / f"subj{i:03d}__label{label}.txt").write_text("\n".join(lines), encoding="utf-8")

    corruption_dir = FIXTURES / "gait_corruption"
    _clean_dir(corruption_dir)
    lines_missing = [",".join(["accel_x", "accel_y"])] + [
        f"{np.sin(0.1 * k):.5f},{np.cos(0.1 * k):.5f}" for k in range(n_samples)
    ]
    (corruption_dir / "MISSING_CHANNEL__label0.txt").write_text("\n".join(lines_missing), encoding="utf-8")
    print(f"gait: {N_COHORT} cohort files + 1 corruption file written to {cohort_dir}, {corruption_dir}")


# ---------------------------------------------------------------------------
# MRI (nifti_volume)
# ---------------------------------------------------------------------------

def generate_mri() -> None:
    import nibabel as nib

    cohort_dir = FIXTURES / "mri_cohort"
    _clean_dir(cohort_dir)
    shape = (16, 16, 16)
    affine = np.eye(4)

    for i in range(N_COHORT):
        label = int(i % 2 == 0)
        case_dir = cohort_dir / f"case{i:03d}__label{label}"
        case_dir.mkdir(parents=True)
        base_intensity = 800.0 if label else 500.0
        for seq in ("t1", "t2"):
            data = base_intensity + RNG.normal(0, 40, shape)
            # a bright "lesion" blob for the positive class, off-center
            if label:
                data[6:10, 6:10, 6:10] += 400
            nib.save(nib.Nifti1Image(data.astype(np.float32), affine), case_dir / f"{seq}.nii.gz")

    corruption_dir = FIXTURES / "mri_corruption"
    _clean_dir(corruption_dir)
    missing_seq_dir = corruption_dir / "MISSING_SEQUENCE__label0"
    missing_seq_dir.mkdir(parents=True)
    nib.save(nib.Nifti1Image((500 + RNG.normal(0, 40, shape)).astype(np.float32), affine), missing_seq_dir / "t1.nii.gz")
    print(f"MRI: {N_COHORT} cohort cases + 1 corruption case written to {cohort_dir}, {corruption_dir}")


# ---------------------------------------------------------------------------
# CT (dicom_series)
# ---------------------------------------------------------------------------

def generate_ct() -> None:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    cohort_dir = FIXTURES / "ct_cohort"
    _clean_dir(cohort_dir)
    rows, cols, n_slices = 32, 32, 8

    def write_series(series_dir: Path, series_uid: str, patient_id: str, label_lesion: bool) -> None:
        series_dir.mkdir(parents=True)
        for z in range(n_slices):
            base = -1000 + RNG.normal(0, 20, (rows, cols))  # air-ish HU baseline
            base[8:24, 8:24] += 1050  # soft-tissue block back up near 0-50 HU
            if label_lesion and 3 <= z <= 5:
                base[14:20, 14:20] += 150  # a denser lesion region mid-volume

            file_meta = FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

            ds = FileDataset(str(series_dir / f"slice{z:03d}.dcm"), {}, file_meta=file_meta, preamble=b"\x00" * 128)
            ds.PatientID = patient_id
            ds.Modality = "CT"
            ds.SeriesInstanceUID = series_uid
            ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
            ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
            ds.InstanceNumber = z + 1
            ds.Rows, ds.Columns = rows, cols
            ds.PixelSpacing = [1.0, 1.0]
            ds.SliceThickness = 1.0
            ds.RescaleSlope = 1.0
            ds.RescaleIntercept = 0.0  # base array is already HU-scaled directly
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 1  # signed, since HU includes negatives
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.is_little_endian = True
            ds.is_implicit_VR = False
            ds.PixelData = base.astype(np.int16).tobytes()
            ds.save_as(str(series_dir / f"slice{z:03d}.dcm"), enforce_file_format=True)

    for i in range(N_COHORT):
        label = int(i % 2 == 0)
        write_series(cohort_dir / f"series{i:03d}__label{label}", generate_uid(), f"PAT{i:03d}", bool(label))

    corruption_dir = FIXTURES / "ct_corruption"
    _clean_dir(corruption_dir)
    # a series whose header modality doesn't match the declared spec modality
    mr_like = corruption_dir / "WRONG_MODALITY__label0"
    mr_like.mkdir(parents=True)
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(mr_like / "slice000.dcm"), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.PatientID = "PATWRONG"
    ds.Modality = "MR"
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.InstanceNumber = 1
    ds.Rows, ds.Columns = rows, cols
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.PixelData = np.zeros((rows, cols), dtype=np.uint16).tobytes()
    ds.save_as(str(mr_like / "slice000.dcm"), enforce_file_format=True)
    print(f"CT: {N_COHORT} cohort series + 1 corruption series written to {cohort_dir}, {corruption_dir}")


# ---------------------------------------------------------------------------
# image2d (plain PNG/JPEG - not part of the formal spec, see image_2d.py)
# ---------------------------------------------------------------------------

def generate_image2d() -> None:
    from PIL import Image

    cohort_dir = FIXTURES / "image2d_cohort"
    _clean_dir(cohort_dir)
    (cohort_dir / "yes").mkdir()
    (cohort_dir / "no").mkdir()
    size = (96, 96)

    for i in range(N_COHORT):
        is_tumor = i % 2 == 0
        arr = RNG.normal(60, 15, size)
        arr[20:76, 20:76] += 60  # a brighter "skull interior" region, like the real scans
        if is_tumor:
            arr[35:55, 35:55] += 90  # a bright lesion blob
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        folder = "yes" if is_tumor else "no"
        Image.fromarray(arr, mode="L").save(cohort_dir / folder / f"{i:03d}.jpg", quality=90)

    corruption_dir = FIXTURES / "image2d_corruption"
    _clean_dir(corruption_dir)
    # an all-black (no foreground) image - should be refused by check_foreground
    Image.fromarray(np.zeros(size, dtype=np.uint8), mode="L").save(corruption_dir / "BLANK__label0.png")
    print(f"image2d: {N_COHORT} cohort images (yes/no folders) + 1 corruption image written to {cohort_dir}, {corruption_dir}")


# ---------------------------------------------------------------------------
# angiography (2D frames, for the real qhealth_qml.angiography extractor)
# ---------------------------------------------------------------------------

def generate_angiography() -> None:
    from PIL import Image, ImageDraw

    cohort_dir = FIXTURES / "angio_cohort"
    _clean_dir(cohort_dir)
    (cohort_dir / "stenosis").mkdir()
    (cohort_dir / "normal").mkdir()
    size = (128, 128)

    for i in range(N_COHORT):
        has_stenosis = i % 2 == 0
        arr = RNG.normal(40, 8, size)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
        draw = ImageDraw.Draw(img)
        # a branching "vessel tree": a few bright line segments, narrower
        # (thinner strokes / gaps) for the stenosis class.
        rng_local = np.random.default_rng(1000 + i)
        width = 2 if has_stenosis else 5
        x, y = 20, 64
        for _ in range(6):
            dx, dy = rng_local.integers(10, 25), rng_local.integers(-15, 15)
            draw.line([(x, y), (x + dx, y + dy)], fill=200, width=int(width))
            x, y = x + dx, y + dy
        folder = "stenosis" if has_stenosis else "normal"
        img.save(cohort_dir / folder / f"{i:03d}.png")

    print(f"angiography: {N_COHORT} cohort frames (stenosis/normal folders) written to {cohort_dir}")


if __name__ == "__main__":
    generate_ecg()
    generate_eeg()
    generate_gait()
    generate_mri()
    generate_ct()
    generate_image2d()
    generate_angiography()
    print("done.")

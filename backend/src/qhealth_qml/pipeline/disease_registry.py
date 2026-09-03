"""Disease configuration registry and standardizer cross-checking."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional
from qhealth_qml.pipeline.exceptions import UnknownDiseaseError


@dataclass
class DiseaseConfig:
    """Configuration for a specific disease pipeline."""

    disease_id: str
    name: str
    short_name: str
    standardizer_disease_id: str
    benchmark_file: str
    default_qubits: int = 6
    modality: str = "Structured Tabular"
    positive_label: str = "Positive"
    negative_label: str = "Negative"


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "diseases.json"


class DiseaseRegistry:
    """Registry managing disease configurations and standardizer validation."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or _DEFAULT_CONFIG_PATH
        self._diseases: Dict[str, DiseaseConfig] = {}
        self.load()

    def load(self) -> None:
        """Load disease configs from JSON file."""
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data.get("diseases", []):
                cfg = DiseaseConfig(**item)
                self._diseases[cfg.disease_id] = cfg
        else:
            # Fallback default hard-wired config for the 4 core diseases
            default_items = [
                DiseaseConfig(
                    disease_id="heart-disease",
                    name="Heart Disease / Myocardial Infarction",
                    short_name="Heart Disease",
                    standardizer_disease_id="heart-disease",
                    benchmark_file="benchmarks/heart-disease.json",
                    default_qubits=6,
                    modality="Clinical & Hemodynamic Tabular",
                    positive_label="High Risk (Stenosis/CAD)",
                    negative_label="Normal / Low Risk",
                ),
                DiseaseConfig(
                    disease_id="breast-cancer",
                    name="Breast Cancer Early Detection",
                    short_name="Breast Cancer",
                    standardizer_disease_id="breast-cancer",
                    benchmark_file="benchmarks/breast-cancer.json",
                    default_qubits=6,
                    modality="Structured Tabular",
                    positive_label="Malignant",
                    negative_label="Benign",
                ),
                DiseaseConfig(
                    disease_id="alzheimers",
                    name="Alzheimer's Disease / Dementia Association",
                    short_name="Alzheimer's",
                    standardizer_disease_id="alzheimers",
                    benchmark_file="benchmarks/alzheimers.json",
                    default_qubits=6,
                    modality="Structured Tabular",
                    positive_label="Dementia (CDR > 0)",
                    negative_label="No Dementia (CDR 0)",
                ),
                DiseaseConfig(
                    disease_id="glioma",
                    name="Glioma MGMT Radiomics Characterization",
                    short_name="Glioma",
                    standardizer_disease_id="glioma",
                    benchmark_file="benchmarks/glioma.json",
                    default_qubits=6,
                    modality="MRI-derived Radiomics Tabular",
                    positive_label="MGMT Methylated",
                    negative_label="Unmethylated",
                ),
                DiseaseConfig(
                    disease_id="stroke",
                    name="Stroke Clinical Risk",
                    short_name="Stroke Risk",
                    standardizer_disease_id="stroke",
                    benchmark_file="benchmarks/stroke.json",
                    default_qubits=6,
                    modality="Clinical & Hemodynamic Tabular",
                    positive_label="Stroke Event",
                    negative_label="No Stroke",
                ),
                DiseaseConfig(
                    disease_id="seizure",
                    name="Epileptic Seizure & Neural Abnormality Detection",
                    short_name="Seizure",
                    standardizer_disease_id="seizure",
                    benchmark_file="benchmarks/seizure.json",
                    default_qubits=6,
                    modality="EEG Biosignal Features",
                    positive_label="Abnormality / Seizure",
                    negative_label="Baseline",
                ),
                DiseaseConfig(
                    disease_id="parkinsons",
                    name="Parkinson's Disease Prodromal Assessment",
                    short_name="Parkinson's",
                    standardizer_disease_id="parkinsons",
                    benchmark_file="benchmarks/parkinsons.json",
                    default_qubits=6,
                    modality="Acoustic & Motor Feature Tabular",
                    positive_label="Parkinsons Positive",
                    negative_label="Healthy Control",
                ),
            ]
            for cfg in default_items:
                self._diseases[cfg.disease_id] = cfg

    def get(self, disease_id: str) -> DiseaseConfig:
        """Retrieve disease config by ID, raising UnknownDiseaseError if not found."""
        if disease_id not in self._diseases:
            raise UnknownDiseaseError(
                disease_id=disease_id,
                message=f"Disease '{disease_id}' is not in the disease registry. "
                        f"Supported diseases: {list(self._diseases.keys())}",
            )
        return self._diseases[disease_id]

    def list_diseases(self) -> List[DiseaseConfig]:
        """List all registered disease configurations."""
        return list(self._diseases.values())

    def list_disease_ids(self) -> List[str]:
        """List all registered disease IDs."""
        return list(self._diseases.keys())

    def validate_against_standardizer(self, list_supported_diseases_fn: Optional[Callable[[], List[str]]] = None) -> Dict[str, bool]:
        """Cross-check disease registry against standardizer module's supported disease IDs.

        Flags any mismatch rather than guessing or renaming.
        """
        if list_supported_diseases_fn is None:
            return {d_id: True for d_id in self._diseases}

        try:
            standardizer_ids = set(list_supported_diseases_fn())
        except Exception:
            standardizer_ids = set()

        validation_status = {}
        for d_id, cfg in self._diseases.items():
            is_valid = cfg.standardizer_disease_id in standardizer_ids
            validation_status[d_id] = is_valid
        return validation_status


# Global singleton instance
disease_registry = DiseaseRegistry()

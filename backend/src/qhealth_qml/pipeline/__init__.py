"""Unified ingestion & preprocessing pipeline (spec 002-ingestion-preprocessing).

One component used unchanged by training and prediction, for one record or
a whole cohort, across tabular EHR and (in later phases) ECG, EEG, gait,
CT and MR. See the spec bundle this was built from for the full contract:
`spec.md` (requirements), `data-model.md` (I/O shape), `design.md`
(architecture and rationale).

Public surface - two calls, no third, no single-record variant:

    from qhealth_qml.pipeline import Pipeline, SourceSpec, Batch, Recipe

    spec  = SourceSpec.load("profiles/cardiac_prognosis.json")
    batch = Pipeline.read(spec)
    fit   = Pipeline.from_spec(spec).fit(batch, n_qubits=6)

    recipe = Recipe.load("runtime/cardiac.recipe.pkl")
    batch  = Pipeline.read(recipe.spec, source="incoming/case.csv")
    run    = Pipeline.from_recipe(recipe).run(batch)

This is Phase 0 + a Phase 1 tabular slice of the 6-phase build in
`tasks.md`: the tabular adapter and the full cleaning/QC/split/selection/
recipe machinery are implemented and tested; the signal (ECG/EEG/gait) and
imaging (NIfTI/DICOM) adapters in Phases 2-3, and Phase 4's consolidation
of the platform's existing code onto this component, are not yet built.
"""

from .pipeline import Pipeline, PipelineError
from .recipe import ManifestMismatchError, Recipe, RecipePredatesRecipeFormatError, RecipeVersionError
from .registry import AdapterRegistry, AdapterUnavailableError, DispatchError, default_registry
from .spec import SourceSpec, SpecValidationError
from .types import (
    Batch,
    FitResult,
    Issue,
    IssueCode,
    LABEL_EXCLUDE,
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    Ledger,
    PreparedArrays,
    QCVerdict,
    RunResult,
    Sample,
    Source,
)

__all__ = [
    "Pipeline", "PipelineError",
    "Recipe", "RecipeVersionError", "RecipePredatesRecipeFormatError", "ManifestMismatchError",
    "AdapterRegistry", "AdapterUnavailableError", "DispatchError", "default_registry",
    "SourceSpec", "SpecValidationError",
    "Batch", "Sample", "Source", "Issue", "IssueCode", "QCVerdict",
    "PreparedArrays", "Ledger", "FitResult", "RunResult",
    "LABEL_POSITIVE", "LABEL_NEGATIVE", "LABEL_EXCLUDE",
]

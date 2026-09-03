"""End-to-end trainable hybrid: a quantum circuit as a torch layer inside the network.

The platform's other hybrid paths are *decoupled* -- a CNN is trained with a classical
head, frozen, and its latents are then handed to a separately-fitted QSVC. That works,
but no gradient ever crosses the quantum boundary: the encoder never learns a
representation that suits the circuit.

This module closes that gap using `qiskit_machine_learning.connectors.TorchConnector`,
which exposes an `EstimatorQNN` as a normal `torch.nn.Module` so backprop flows through
the circuit's input parameters into the encoder beneath it.

The circuit follows the "dressed quantum circuit" of Mari et al. 2020 (*Transfer learning
in hybrid classical-quantum neural networks*), the standard architecture for quantum +
medical imaging: a classical layer projects the encoder's latent down to the qubit count,
a bounded non-linearity maps it into a sensible angle range, the circuit embeds and
entangles, and a classical layer maps the measured expectations to logits.

Note on `input_gradients=True`: this is REQUIRED. `EstimatorQNN` defaults it to False,
which silently yields zero/incorrect gradients w.r.t. the circuit's inputs -- the encoder
below would then never receive a useful learning signal, and training would appear to run
fine while learning nothing through the quantum path.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .raw_hybrid import _require_torch

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - torch is optional
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def build_estimator_qnn(
    num_qubits: int = 4,
    feature_map_reps: int = 1,
    feature_map_entanglement: str = "linear",
    ansatz_reps: int = 2,
    ansatz_entanglement: str = "linear",
) -> Any:
    """Build an EstimatorQNN whose inputs are differentiable (required for hybrid training)."""

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import real_amplitudes, zz_feature_map
    from qiskit.primitives import StatevectorEstimator
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_machine_learning.neural_networks import EstimatorQNN

    if feature_map_entanglement not in {"linear", "full", "circular"}:
        raise ValueError("feature_map_entanglement must be linear, full, or circular")
    if ansatz_entanglement not in {"linear", "full", "circular"}:
        raise ValueError("ansatz_entanglement must be linear, full, or circular")

    feature_map = zz_feature_map(
        num_qubits, reps=feature_map_reps, entanglement=feature_map_entanglement
    )
    ansatz = real_amplitudes(num_qubits, reps=ansatz_reps, entanglement=ansatz_entanglement)
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    # One single-qubit Z observable per qubit, NOT EstimatorQNN's default global Z^n.
    # The default emits a single expectation value, which forces the whole encoder to
    # communicate through one scalar -- a severe bottleneck that measurably destabilised
    # training here (gait: 0.740 +/- 0.132 with one output). Mari et al.'s dressed circuit
    # measures each qubit and feeds all n expectations to the classical output layer.
    observables = [
        SparsePauliOp.from_sparse_list([("Z", [qubit], 1.0)], num_qubits=num_qubits)
        for qubit in range(num_qubits)
    ]

    return EstimatorQNN(
        circuit=circuit,
        observables=observables,
        input_params=list(feature_map.parameters),
        weight_params=list(ansatz.parameters),
        # Without this the encoder beneath the circuit receives no gradient signal.
        input_gradients=True,
        estimator=StatevectorEstimator(),
    )


if nn is not None:

    class DressedQuantumLayer(nn.Module):
        """Mari et al. dressed quantum circuit: project -> bound angles -> circuit -> project.

        The `tanh * pi/2` step is what makes the embedding well-conditioned: it bounds every
        angle into [-pi/2, pi/2] regardless of the encoder's output scale, which is the same
        bandwidth consideration that mattered for this platform's fidelity kernels (a too-wide
        angle spread drives the circuit toward the concentrated, uninformative regime).
        """

        def __init__(
            self,
            in_features: int,
            num_qubits: int = 4,
            out_features: int = 1,
            angle_scale: float = 1.0,
            **qnn_kwargs: Any,
        ) -> None:
            super().__init__()
            from qiskit_machine_learning.connectors import TorchConnector

            self.num_qubits = int(num_qubits)
            self.angle_scale = float(angle_scale)
            self.pre = nn.Linear(in_features, num_qubits)
            qnn = build_estimator_qnn(num_qubits=num_qubits, **qnn_kwargs)
            self.quantum = TorchConnector(qnn)
            # One expectation value per qubit (see build_estimator_qnn) -- the classical
            # output layer therefore sees n_qubits measurements, per Mari et al.
            self.post = nn.Linear(num_qubits, out_features)

        def forward(self, values: Any) -> Any:
            angles = torch.tanh(self.pre(values)) * (np.pi / 2.0) * self.angle_scale
            expectations = self.quantum(angles)
            return self.post(expectations)

    class HybridEncoderQNN(nn.Module):
        """Any encoder (1-D gait CNN, 3-D imaging CNN, ...) followed by a trainable quantum head.

        `encoder` must map a batch of raw inputs to a [batch, latent_dim] tensor -- exactly the
        contract the platform's existing `RawGaitEncoder` / `VolumeEncoder` already satisfy, so
        either can be dropped in unchanged.
        """

        def __init__(
            self,
            encoder: Any,
            latent_dim: int,
            num_qubits: int = 4,
            angle_scale: float = 1.0,
            freeze_encoder: bool = False,
            **qnn_kwargs: Any,
        ) -> None:
            super().__init__()
            self.encoder = encoder
            if freeze_encoder:
                for parameter in self.encoder.parameters():
                    parameter.requires_grad_(False)
            self.quantum_head = DressedQuantumLayer(
                in_features=latent_dim,
                num_qubits=num_qubits,
                out_features=1,
                angle_scale=angle_scale,
                **qnn_kwargs,
            )

        def forward(self, values: Any) -> tuple[Any, Any]:
            latent = self.encoder(values)
            return latent, self.quantum_head(latent).squeeze(-1)

else:

    class DressedQuantumLayer:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()

    class HybridEncoderQNN:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


def quantum_parameter_count(model: Any) -> dict[str, int]:
    """Report how much of the model is quantum vs classical (useful for honest reporting)."""

    total = sum(p.numel() for p in model.parameters())
    quantum = 0
    for module in model.modules():
        if type(module).__name__ == "TorchConnector":
            quantum += sum(p.numel() for p in module.parameters())
    return {"total_parameters": int(total), "quantum_parameters": int(quantum)}

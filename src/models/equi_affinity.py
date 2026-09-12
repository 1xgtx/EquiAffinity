"""Phase 2 EquiAffinity model and a four-configuration smoke test.

The full model separates intramolecular and protein--ligand interface edges.
Its geometry block passes scalar messages parameterised by radial distances;
because every operation on coordinates uses only pairwise distances, the scalar
representations and affinity output are invariant to SE(3)/E(3) transforms
(the scalar channel is the l=0 equivariant representation).  The optional GAT
baseline deliberately discards this geometric construction.

Run the smoke test from the project root:

    python -m src.models.equi_affinity
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import scatter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE_TEST_GRAPH = PROJECT_ROOT / "outputs" / "1a0q.pt"


def _batch_vector(data: Data, device: torch.device) -> Tensor:
    """Return one graph-id per node, supporting both Data and Batch objects."""
    if hasattr(data, "batch") and data.batch is not None:
        return data.batch.to(device=device, dtype=torch.long)
    return torch.zeros(data.x.size(0), dtype=torch.long, device=device)


def _ligand_mask(data: Data, device: torch.device) -> Tensor:
    """Read the explicit mask or recover it from the final Phase-1 x feature."""
    if hasattr(data, "is_ligand") and data.is_ligand is not None:
        return data.is_ligand.to(device=device).reshape(-1).bool()
    if data.x.size(1) < 1:
        raise ValueError("Data.x needs an is_ligand feature or a separate is_ligand tensor.")
    # Phase 1 stores is_ligand as the final x column.
    return data.x[:, -1].to(device=device) > 0.5


def _interface_edges(data: Data, device: torch.device) -> Tensor:
    """Get interface edges, accepting both current and Phase-1 graph schemas."""
    if hasattr(data, "interface_edge_index") and data.interface_edge_index is not None:
        return data.interface_edge_index.to(device=device, dtype=torch.long)
    if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.size(1) >= 1:
        # In Phase 1, the final edge feature is the Interface flag.
        interface = data.edge_attr[:, -1].to(device=device) > 0.5
        return data.edge_index[:, interface].to(device=device, dtype=torch.long)
    return torch.empty((2, 0), dtype=torch.long, device=device)


def _intramolecular_edges(data: Data, device: torch.device) -> Tensor:
    """Return covalent/intramolecular edges, excluding the interface class."""
    edge_index = data.edge_index.to(device=device, dtype=torch.long)
    if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.size(1) >= 1:
        interface = data.edge_attr[:, -1].to(device=device) > 0.5
        return edge_index[:, ~interface]
    return edge_index


class RadialBasis(nn.Module):
    """Gaussian radial basis encoding of E(3)-invariant pairwise distances."""

    def __init__(self, num_basis: int, cutoff: float) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        self.register_buffer("centres", torch.linspace(0.0, cutoff, num_basis))
        self.gamma = nn.Parameter(torch.tensor((num_basis / cutoff) ** 2), requires_grad=False)

    def forward(self, distance: Tensor) -> Tensor:
        distance = distance.clamp(min=0.0, max=self.cutoff).unsqueeze(-1)
        return torch.exp(-self.gamma * (distance - self.centres).square())


class EquivariantScalarInteraction(nn.Module):
    """Distance-conditioned l=0 equivariant message-passing block.

    Node channels are scalars, so they remain unchanged under rotations and
    translations. Edge messages use only ||r_i-r_j||, an E(3)-invariant value.
    This yields an invariant affinity predictor while retaining geometric input.
    """

    def __init__(self, hidden_dim: int, radial_basis: int, cutoff: float) -> None:
        super().__init__()
        self.rbf = RadialBasis(radial_basis, cutoff)
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim + radial_basis, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.update = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: Tensor, pos: Tensor, edge_index: Tensor) -> Tensor:
        if edge_index.numel() == 0:
            return h
        source, target = edge_index
        distance = (pos[source] - pos[target]).norm(dim=-1)
        message = self.message(torch.cat((h[source], h[target], self.rbf(distance)), dim=-1))
        aggregated = scatter(message, target, dim=0, dim_size=h.size(0), reduce="mean")
        return self.norm(h + self.update(torch.cat((h, aggregated), dim=-1)))


class CrossGraphGate(nn.Module):
    """Interface-only, geometry-aware gated message exchange between molecules."""

    def __init__(self, hidden_dim: int, radial_basis: int, cutoff: float) -> None:
        super().__init__()
        self.rbf = RadialBasis(radial_basis, cutoff)
        self.message = nn.Sequential(nn.Linear(2 * hidden_dim + radial_basis, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.gate = nn.Sequential(nn.Linear(2 * hidden_dim + radial_basis, hidden_dim), nn.Sigmoid())
        self.update = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: Tensor, pos: Tensor, interface_edge_index: Tensor) -> Tensor:
        if interface_edge_index.numel() == 0:
            return h
        source, target = interface_edge_index
        radial = self.rbf((pos[source] - pos[target]).norm(dim=-1))
        pair = torch.cat((h[source], h[target], radial), dim=-1)
        message = self.message(pair) * self.gate(pair)
        aggregated = scatter(message, target, dim=0, dim_size=h.size(0), reduce="mean")
        return self.norm(h + self.update(torch.cat((h, aggregated), dim=-1)))


class CrossGraphGatedPooling(nn.Module):
    """Pool protein and ligand separately, then gate their learned interaction."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.node_attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.cross_gate = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Sigmoid())

    @staticmethod
    def _masked_pool(h: Tensor, mask: Tensor, batch: Tensor, num_graphs: int, scores: Tensor) -> Tensor:
        """Softmax attention pool, returning zeros only for a missing molecule."""
        output = h.new_zeros((num_graphs, h.size(-1)))
        for graph_id in range(num_graphs):
            selected = (batch == graph_id) & mask
            if selected.any():
                weights = torch.softmax(scores[selected], dim=0).unsqueeze(-1)
                output[graph_id] = (weights * h[selected]).sum(dim=0)
        return output

    def forward(self, h: Tensor, ligand_mask: Tensor, batch: Tensor) -> Tensor:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        scores = self.node_attention(h).squeeze(-1)
        protein = self._masked_pool(h, ~ligand_mask, batch, num_graphs, scores)
        ligand = self._masked_pool(h, ligand_mask, batch, num_graphs, scores)
        gate = self.cross_gate(torch.cat((protein, ligand), dim=-1))
        return gate * ligand + (1.0 - gate) * protein


class EquiAffinityNet(nn.Module):
    """Dual-graph affinity predictor with pooling, graph, and geometry ablations.

    Args:
        node_dim: Number of input atom features (18 for the Phase-1 graph).
        hidden_dim: Width of all scalar message-passing layers.
        num_layers: Number of intramolecular interaction blocks.
        ablation_pooling: Use ordinary graph-wise mean pooling.
        ablation_graph: Merge protein and ligand into one untyped graph.
        ablation_equivariant: Replace distance-conditioned layers with GATConv.
    """

    def __init__(
        self,
        node_dim: int = 18,
        hidden_dim: int = 96,
        num_layers: int = 3,
        radial_basis: int = 16,
        cutoff: float = 5.0,
        gat_heads: int = 4,
        ablation_pooling: bool = False,
        ablation_graph: bool = False,
        ablation_equivariant: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim % gat_heads:
            raise ValueError("hidden_dim must be divisible by gat_heads.")
        self.node_dim = node_dim
        self.ablation_pooling = ablation_pooling
        self.ablation_graph = ablation_graph
        self.ablation_equivariant = ablation_equivariant
        self.encoder = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim))
        if ablation_equivariant:
            self.layers = nn.ModuleList([GATConv(hidden_dim, hidden_dim // gat_heads, heads=gat_heads, concat=True,
                                                 dropout=0.0, add_self_loops=True) for _ in range(num_layers)])
            self.cross_interaction = None
        else:
            self.layers = nn.ModuleList([EquivariantScalarInteraction(hidden_dim, radial_basis, cutoff)
                                         for _ in range(num_layers)])
            self.cross_interaction = CrossGraphGate(hidden_dim, radial_basis, cutoff)
        self.pooling = CrossGraphGatedPooling(hidden_dim)
        self.predictor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, data: Data) -> Tensor:
        """Predict one scalar per graph; a single ``Data`` returns shape ``[1, 1]``."""
        if data.x.ndim != 2 or data.x.size(1) != self.node_dim:
            raise ValueError(f"Expected data.x with shape [N, {self.node_dim}], got {tuple(data.x.shape)}")
        device = next(self.parameters()).device
        x, pos = data.x.to(device=device, dtype=torch.float32), data.pos.to(device=device, dtype=torch.float32)
        if pos.ndim != 2 or pos.size(1) != 3 or pos.size(0) != x.size(0):
            raise ValueError("data.pos must have shape [N, 3] matching data.x.")
        batch = _batch_vector(data, device)
        ligand_mask = _ligand_mask(data, device)
        all_edges = data.edge_index.to(device=device, dtype=torch.long)
        interface_edges = _interface_edges(data, device)
        intra_edges = _intramolecular_edges(data, device)
        h = self.encoder(x)
        if self.ablation_graph:
            propagation_edges = all_edges
            for layer in self.layers:
                h = F.elu(layer(h, propagation_edges)) if self.ablation_equivariant else layer(h, pos, propagation_edges)
        else:
            for layer in self.layers:
                h = F.elu(layer(h, intra_edges)) if self.ablation_equivariant else layer(h, pos, intra_edges)
            if self.ablation_equivariant:
                # In the GAT baseline, interface topology is still available but no geometry is used.
                if interface_edges.numel():
                    h = F.elu(self.layers[-1](h, interface_edges))
            else:
                h = self.cross_interaction(h, pos, interface_edges)
        if self.ablation_pooling:
            graph_embedding = scatter(h, batch, dim=0, reduce="mean")
        elif self.ablation_graph:
            # Attention remains, but molecular membership is intentionally ignored.
            graph_embedding = self.pooling._masked_pool(h, torch.ones_like(ligand_mask), batch,
                                                        int(batch.max().item()) + 1, self.pooling.node_attention(h).squeeze(-1))
        else:
            graph_embedding = self.pooling(h, ligand_mask, batch)
        return self.predictor(graph_embedding)


def _load_smoke_test_graph(path: Path) -> Data:
    """Load a trusted local PyG graph across PyTorch 2.5/2.6 loader defaults."""
    if not path.is_file():
        raise FileNotFoundError(f"Smoke-test graph not found: {path}")
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before weights_only was introduced.
        data = torch.load(path, map_location="cpu")
    if not isinstance(data, Data):
        raise TypeError(f"Expected torch_geometric.data.Data, received {type(data)!r}")
    return data


def run_smoke_test() -> None:
    """Run four forward passes and print parameters, latency, and output shape."""
    torch.manual_seed(17)
    data = _load_smoke_test_graph(SMOKE_TEST_GRAPH)
    configurations = (
        ("Full Model (Dual-graph + Equivariant Geometry + Cross-attention)", {}),
        ("Ablation 1: Mean-pooling", {"ablation_pooling": True}),
        ("Ablation 2: Merged Single-graph", {"ablation_graph": True}),
        ("Ablation 3: Non-equivariant GAT Baseline", {"ablation_equivariant": True}),
    )
    print("\n" + "=" * 108)
    print("EquiAffinity Phase 2 — single-sample smoke test (1a0q)")
    print("=" * 108)
    print(f"{'Configuration':<64} {'Total Parameters':>18} {'Inference Time':>18} {'Output Shape':>16}")
    print("-" * 108)
    for name, options in configurations:
        model = EquiAffinityNet(node_dim=data.x.size(1), **options).eval()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        with torch.inference_mode():
            start = time.perf_counter()
            output = model(data)
            elapsed_ms = (time.perf_counter() - start) * 1_000.0
        if tuple(output.shape) != (1, 1):
            raise AssertionError(f"{name}: expected output shape [1, 1], got {tuple(output.shape)}")
        if not torch.isfinite(output).all():
            raise AssertionError(f"{name}: produced non-finite output")
        print(f"{name:<64} {parameter_count:>18,} {elapsed_ms:>15.2f} ms {str(tuple(output.shape)):>16}")
    print("-" * 108)
    print("PASS: all 4 configurations produced finite affinity predictions with shape [1, 1].")
    print("=" * 108 + "\n")


if __name__ == "__main__":
    try:
        run_smoke_test()
    except Exception as error:
        print(f"SMOKE TEST FAILED: {type(error).__name__}: {error}", file=__import__("sys").stderr)
        raise

#!/usr/bin/env python3
"""Build and visualise the atom-level protein--ligand graph for PDB ID 1a0q.

Run from the project root:
    python src/build_and_visualize.py

Required packages: numpy, scipy, biopython, rdkit, torch, torch-geometric,
networkx, matplotlib, plotly.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")  # Always work on headless servers.
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import networkx as nx
    import numpy as np
    import plotly.graph_objects as go
    import torch
    from Bio.PDB import PDBParser
    from rdkit import Chem
    from scipy.spatial import cKDTree
    from torch_geometric.data import Data
except ImportError as error:
    missing_package = error.name or "a required package"
    print(
        f"ERROR: Missing dependency '{missing_package}'. Install numpy scipy biopython rdkit "
        "torch torch-geometric networkx matplotlib plotly, then rerun this script.",
        file=sys.stderr,
    )
    sys.exit(2)


PDB_ID = "1a0q"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPLEX_DIR = PROJECT_ROOT / "data" / "P-L" / "1981-2000" / PDB_ID
PROTEIN_FILE = COMPLEX_DIR / f"{PDB_ID}_protein.pdb"
LIGAND_SDF = COMPLEX_DIR / f"{PDB_ID}_ligand.sdf"
LIGAND_MOL2 = COMPLEX_DIR / f"{PDB_ID}_ligand.mol2"
INDEX_FILE = PROJECT_ROOT / "data" / "index" / "INDEX_general_PL.2020R1.lst"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
GRAPH_FILE = OUTPUT_DIR / f"{PDB_ID}.pt"
IMAGE_FILE = OUTPUT_DIR / f"{PDB_ID}_2d_graph.png"
HTML_FILE = OUTPUT_DIR / f"{PDB_ID}_3d_interactive.html"
CUTOFF = 5.0

# x = 10 atomic-element one-hot + 6 hybridisation one-hot + formal charge + is_ligand.
ELEMENT_CLASSES = ("C", "N", "O", "S", "P", "F", "CL", "BR", "I")
HYBRIDIZATION_CLASSES = ("SP", "SP2", "SP3", "SP3D", "SP3D2")
WATER_NAMES = {"HOH", "WAT", "DOD", "H2O"}
ION_NAMES = {"NA", "K", "CA", "MG", "MN", "ZN", "FE", "CO", "NI", "CU", "CD", "HG",
             "AL", "LI", "RB", "CS", "SR", "BA", "CL", "BR", "IOD", "F"}
RADII = {"H": .31, "C": .76, "N": .71, "O": .66, "F": .57, "P": 1.07,
         "S": 1.05, "CL": 1.02, "BR": 1.20, "I": 1.39, "SE": 1.20}
NUMBER_TO_ELEMENT = {6: "C", 7: "N", 8: "O", 16: "S", 15: "P", 9: "F", 17: "CL", 35: "BR", 53: "I"}
ELEMENT_TO_NUMBER = {value: key for key, value in NUMBER_TO_ELEMENT.items()}
UNITS = {"M": 1.0, "MM": 1e-3, "UM": 1e-6, "NM": 1e-9, "PM": 1e-12, "FM": 1e-15}
AFFINITY_RE = re.compile(r"\b(?P<kind>K[di])\s*(?P<qualifier>[<>=~]*)\s*"
                         r"(?P<value>\d*\.?\d+)\s*(?P<unit>[munpf]?M)\b", re.I)


def node_feature(atomic_number: int, hybridization: str, charge: float, is_ligand: bool) -> List[float]:
    """Return the fixed, documented 18-dimensional atom feature vector."""
    element = NUMBER_TO_ELEMENT.get(int(atomic_number), "OTHER")
    element_features = [float(element == cls) for cls in ELEMENT_CLASSES] + [float(element == "OTHER")]
    hybridization = hybridization.upper()
    hybrid_features = [float(hybridization == cls) for cls in HYBRIDIZATION_CLASSES]
    hybrid_features.append(float(hybridization not in HYBRIDIZATION_CLASSES))
    return element_features + hybrid_features + [float(charge), float(is_ligand)]


def normalise_element(value: str, atom_name: str) -> str:
    """Return a robust upper-case PDB element, including malformed records."""
    element = (value or "").strip().upper()
    if element:
        return element
    letters = re.sub(r"[^A-Za-z]", "", atom_name).upper()
    return letters[:2] if letters[:2] in {"CL", "BR", "SE"} else letters[:1]


def parse_affinity(index_path: Path, pdb_id: str) -> Tuple[float, float, str, str]:
    """Return (pK, molar value, Kd/Ki type, qualifier) for the requested ID."""
    with index_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            if not fields or fields[0].lower() != pdb_id.lower():
                continue
            match = AFFINITY_RE.search(line)
            if match is None:
                raise ValueError(f"No quantitative Kd/Ki value in index entry: {line.strip()}")
            molar = float(match["value"]) * UNITS[match["unit"].upper()]
            if molar <= 0:
                raise ValueError("Binding affinity must be positive")
            return -math.log10(molar), molar, match["kind"], match["qualifier"] or "="
    raise KeyError(f"PDB ID {pdb_id} not found in {index_path}")


def parse_protein(pdb_path: Path) -> Dict[str, object]:
    """Read the first protein model, omitting water and common inorganic ions."""
    structure = PDBParser(QUIET=True).get_structure("protein", str(pdb_path))
    atoms, positions, features = [], [], []
    residue_keys, atom_names, elements, serials = [], [], [], []
    model = next(structure.get_models(), None)
    if model is None:
        raise ValueError(f"No model found in {pdb_path}")
    for chain in model:
        for residue in chain:
            residue_atoms = list(residue.get_atoms())
            residue_name = residue.get_resname().strip().upper()
            residue_elements = [normalise_element(atom.element, atom.get_name()) for atom in residue_atoms]
            is_isolated_nonmetal_hetero = (residue.id[0].strip() and len(residue_atoms) == 1
                                           and residue_elements[0] not in {"C", "N", "O", "S", "P"})
            if residue_name in WATER_NAMES or residue_name in ION_NAMES or is_isolated_nonmetal_hetero:
                continue
            for atom, element in zip(residue_atoms, residue_elements):
                if element == "H":  # keep a consistently heavy-atom graph
                    continue
                atoms.append(atom)
                positions.append(np.asarray(atom.coord, dtype=np.float64))
                features.append(node_feature(ELEMENT_TO_NUMBER.get(element, 0), "OTHER", 0.0, False))
                residue_keys.append((str(chain.id), int(residue.id[1]), str(residue.id[2]).strip()))
                atom_names.append(atom.get_name().strip().upper())
                elements.append(element)
                serials.append(int(atom.serial_number))
    if not atoms:
        raise ValueError(f"No retained protein atoms in {pdb_path}")
    return {"features": features, "positions": np.asarray(positions), "residue_keys": residue_keys,
            "atom_names": atom_names, "elements": elements, "serials": serials}


def protein_bonds(protein: Dict[str, object]) -> List[Tuple[int, int, List[float]]]:
    """Infer same-residue bonds plus peptide/disulfide links; all are single bonds.

    Restricting radius-based inference to each residue prevents accidental bonds
    between separate residues that happen to be close in a folded structure.
    """
    positions = protein["positions"]
    residue_keys, elements, atom_names = protein["residue_keys"], protein["elements"], protein["atom_names"]
    bonds = set()
    for left, right in cKDTree(positions).query_pairs(r=2.45):
        if residue_keys[left] != residue_keys[right]:
            continue
        distance = float(np.linalg.norm(positions[left] - positions[right]))
        maximum = 1.25 * (RADII.get(elements[left], .77) + RADII.get(elements[right], .77)) + .15
        if .40 < distance <= maximum:
            bonds.add((left, right))
    carbonyl_c, amide_n, thiols = {}, {}, []
    for index, (key, atom_name) in enumerate(zip(residue_keys, atom_names)):
        if atom_name == "C":
            carbonyl_c[key] = index
        elif atom_name == "N":
            amide_n[key] = index
        elif atom_name == "SG" and elements[index] == "S":
            thiols.append(index)
    for key, c_index in carbonyl_c.items():
        n_index = amide_n.get((key[0], key[1] + 1, ""))
        if n_index is not None and np.linalg.norm(positions[c_index] - positions[n_index]) <= 1.9:
            bonds.add(tuple(sorted((c_index, n_index))))
    if len(thiols) > 1:
        for local_a, local_b in cKDTree(positions[thiols]).query_pairs(r=2.25):
            bonds.add(tuple(sorted((thiols[local_a], thiols[local_b]))))
    single = [1., 0., 0., 0., 0.]
    return [(a, b, single) for a, b in sorted(bonds)]


def read_ligand(sdf_path: Path, mol2_path: Path):
    """Prefer SDF; transparently fall back to MOL2 if SDF parsing fails."""
    errors = []
    for path, reader in ((sdf_path, lambda p: Chem.MolFromMolFile(str(p), sanitize=True, removeHs=False)),
                         (mol2_path, lambda p: Chem.MolFromMol2File(str(p), sanitize=True, removeHs=False))):
        if not path.exists():
            errors.append(f"missing {path.name}")
            continue
        try:
            molecule = reader(path)
            if molecule is not None and molecule.GetNumConformers() > 0:
                return molecule, path.name
            errors.append(f"invalid {path.name}")
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    raise ValueError("Ligand parse failed: " + "; ".join(errors))


def ligand_graph(molecule) -> Tuple[List[List[float]], np.ndarray, List[str], List[Tuple[int, int, List[float]]]]:
    """Extract heavy atoms, RDKit bond orders and coordinates from a ligand."""
    conformer = molecule.GetConformer()
    original_indices = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 1]
    remap = {old: new for new, old in enumerate(original_indices)}
    features, positions, labels = [], [], []
    for old_index in original_indices:
        atom = molecule.GetAtomWithIdx(old_index)
        features.append(node_feature(atom.GetAtomicNum(), str(atom.GetHybridization()), atom.GetFormalCharge(), True))
        point = conformer.GetAtomPosition(old_index)
        positions.append([point.x, point.y, point.z])
        labels.append(atom.GetSymbol())
    if not features:
        raise ValueError("Ligand has no heavy atoms")
    bond_features = {Chem.BondType.SINGLE: [1., 0., 0., 0., 0.],
                     Chem.BondType.DOUBLE: [0., 1., 0., 0., 0.],
                     Chem.BondType.TRIPLE: [0., 0., 1., 0., 0.],
                     Chem.BondType.AROMATIC: [0., 0., 0., 1., 0.]}
    bonds = []
    for bond in molecule.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if start in remap and end in remap:
            bonds.append((remap[start], remap[end], bond_features.get(bond.GetBondType(), [1., 0., 0., 0., 0.])))
    return features, np.asarray(positions, dtype=np.float64), labels, bonds


def build_graph(protein: Dict[str, object], ligand, affinity: float, ligand_source: str) -> Tuple[Data, List[Tuple[int, int]], int, int]:
    """Build Data and return its undirected interface pairs for visualisation."""
    ligand_features, ligand_pos, ligand_labels, ligand_bonds = ligand_graph(ligand)
    protein_features, protein_pos = protein["features"], protein["positions"]
    protein_bond_list = protein_bonds(protein)
    offset = len(protein_features)
    edge_pairs, edge_features = [], []
    for a, b, feature in protein_bond_list + [(a + offset, b + offset, feature) for a, b, feature in ligand_bonds]:
        edge_pairs.extend(((a, b), (b, a)))
        edge_features.extend((feature, feature))
    interface_pairs = []
    interface_feature = [0., 0., 0., 0., 1.]
    for protein_index, ligand_neighbours in enumerate(cKDTree(protein_pos).query_ball_tree(cKDTree(ligand_pos), r=CUTOFF)):
        for ligand_index in ligand_neighbours:
            ligand_node = offset + ligand_index
            interface_pairs.append((protein_index, ligand_node))
            edge_pairs.extend(((protein_index, ligand_node), (ligand_node, protein_index)))
            edge_features.extend((interface_feature, interface_feature))
    data = Data(x=torch.tensor(protein_features + ligand_features, dtype=torch.float32),
                pos=torch.tensor(np.vstack((protein_pos, ligand_pos)), dtype=torch.float32),
                edge_index=torch.tensor(edge_pairs, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_features, dtype=torch.float32),
                y=torch.tensor(affinity, dtype=torch.float32))
    data.pdb_id, data.ligand_source = PDB_ID, ligand_source
    data.num_protein_atoms, data.num_ligand_atoms = len(protein_features), len(ligand_features)
    # These public counts are undirected chemical/contact pairs; edge_index itself
    # contains both directions for message passing.
    data.num_intra_molecular_covalent_edges = len(protein_bond_list) + len(ligand_bonds)
    data.num_interface_contact_edges = len(interface_pairs)
    data.num_covalent_edges = 2 * data.num_intra_molecular_covalent_edges
    data.num_interface_edges = 2 * data.num_interface_contact_edges
    data.ligand_elements = ligand_labels
    return data, interface_pairs, len(protein_bond_list), len(ligand_bonds)


def pocket_subgraph_edges(data: Data, interface_pairs: Sequence[Tuple[int, int]]):
    """Return visible nodes and unique covalent/interface edges for both renderers."""
    protein_count = data.num_protein_atoms
    ligand_nodes = set(range(protein_count, data.num_nodes))
    pocket_protein = {protein for protein, _ in interface_pairs}
    visible = pocket_protein | ligand_nodes
    covalent_edges, interface_edges = set(), set()
    for edge_index in range(data.edge_index.shape[1]):
        left, right = (int(data.edge_index[0, edge_index]), int(data.edge_index[1, edge_index]))
        if left not in visible or right not in visible:
            continue
        edge = tuple(sorted((left, right)))
        if data.edge_attr[edge_index, 4].item() == 1:
            interface_edges.add(edge)
        else:
            covalent_edges.add(edge)
    return pocket_protein, ligand_nodes, visible, sorted(covalent_edges), sorted(interface_edges)


def render_pocket_graph(data: Data, interface_pairs: Sequence[Tuple[int, int]], image_path: Path) -> None:
    """Render ligand + contacting protein atoms only, using an informative 2D layout."""
    protein_count = data.num_protein_atoms
    pocket_protein, ligand_nodes, visible, covalent_edges, interface_edges = pocket_subgraph_edges(data, interface_pairs)
    graph = nx.Graph()
    graph.add_nodes_from(visible)
    graph.add_edges_from(covalent_edges + interface_edges)
    # Seeded spring layout makes figures reproducible while retaining topology readability.
    positions = nx.spring_layout(graph, seed=17, k=1.35 / max(len(visible), 1) ** .45, iterations=250)
    fig, axis = plt.subplots(figsize=(14, 10), facecolor="white")
    axis.set_facecolor("white")
    nx.draw_networkx_edges(graph, positions, edgelist=covalent_edges, width=1.1, edge_color="#8A8A8A", ax=axis)
    nx.draw_networkx_edges(graph, positions, edgelist=interface_edges, width=2.1, style="dashed", edge_color="#D62728", ax=axis)
    nx.draw_networkx_nodes(graph, positions, nodelist=sorted(pocket_protein), node_size=105,
                           node_color="#A9D6E5", edgecolors="#3D7182", linewidths=.55, ax=axis)
    nx.draw_networkx_nodes(graph, positions, nodelist=sorted(ligand_nodes), node_size=180,
                           node_color="#B8E0B0", edgecolors="#38713C", linewidths=.8, ax=axis)
    ligand_labels = {node: data.ligand_elements[node - protein_count] for node in ligand_nodes}
    nx.draw_networkx_labels(graph, positions, labels=ligand_labels, font_size=7, font_weight="bold", ax=axis)
    axis.set_title(f"PDB {PDB_ID} | Pocket topology: {len(visible)} atoms, {len(interface_edges)} 5 Å contacts",
                   fontsize=15, pad=17, fontweight="bold")
    axis.legend(handles=[Line2D([0], [0], marker="o", color="w", label="Ligand atom", markerfacecolor="#B8E0B0", markersize=10),
                        Line2D([0], [0], marker="o", color="w", label="Pocket protein atom", markerfacecolor="#A9D6E5", markersize=9),
                        Line2D([0], [0], color="#8A8A8A", lw=1.5, label="Covalent bond"),
                        Line2D([0], [0], color="#D62728", lw=2, linestyle="--", label="Interface contact (≤5 Å)")],
                loc="upper left", frameon=True, framealpha=.96, edgecolor="#CCCCCC")
    axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(image_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _line_coordinates(positions: np.ndarray, edges: Sequence[Tuple[int, int]]):
    """Encode many disconnected 3D segments as one Plotly line trace."""
    x, y, z = [], [], []
    for left, right in edges:
        for coordinate in (positions[left], positions[right]):
            x.append(float(coordinate[0]))
            y.append(float(coordinate[1]))
            z.append(float(coordinate[2]))
        x.append(None)
        y.append(None)
        z.append(None)
    return x, y, z


def render_interactive_3d(data: Data, interface_pairs: Sequence[Tuple[int, int]], html_path: Path) -> None:
    """Export a standalone, draggable 3D pocket graph using physical coordinates."""
    pocket_protein, ligand_nodes, _, covalent_edges, interface_edges = pocket_subgraph_edges(data, interface_pairs)
    positions = data.pos.detach().cpu().numpy()
    protein_nodes = sorted(pocket_protein)
    ligand_nodes = sorted(ligand_nodes)
    figure = go.Figure()
    if covalent_edges:
        x, y, z = _line_coordinates(positions, covalent_edges)
        figure.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", name="Covalent bond",
                                      line=dict(color="#828282", width=3), hoverinfo="skip"))
    if interface_edges:
        x, y, z = _line_coordinates(positions, interface_edges)
        figure.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", name="Interface contact (≤5 Å)",
                                      line=dict(color="#D62728", width=5), hoverinfo="skip"))
    if protein_nodes:
        protein_coordinates = positions[protein_nodes]
        figure.add_trace(go.Scatter3d(
            x=protein_coordinates[:, 0], y=protein_coordinates[:, 1], z=protein_coordinates[:, 2],
            mode="markers", name="Pocket protein atom", hovertemplate="Protein atom<br>x=%{x:.2f}, y=%{y:.2f}, z=%{z:.2f}<extra></extra>",
            marker=dict(size=5.5, color="#8ECAE6", line=dict(color="#39738A", width=.5), opacity=.92),
        ))
    ligand_coordinates = positions[ligand_nodes]
    ligand_symbols = [data.ligand_elements[node - data.num_protein_atoms] for node in ligand_nodes]
    figure.add_trace(go.Scatter3d(
        x=ligand_coordinates[:, 0], y=ligand_coordinates[:, 1], z=ligand_coordinates[:, 2],
        mode="markers", name="Ligand atom", text=ligand_symbols,
        hovertemplate="Ligand atom: %{text}<br>x=%{x:.2f}, y=%{y:.2f}, z=%{z:.2f}<extra></extra>",
        marker=dict(size=8, color="#75B96F", line=dict(color="#2F6B32", width=1), opacity=.98),
    ))
    figure.update_layout(
        title=dict(text=f"PDB {PDB_ID}: 3D pocket interaction graph", x=.5),
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(bgcolor="rgba(255,255,255,.88)", bordercolor="#CCCCCC", borderwidth=1),
        scene=dict(xaxis=dict(title="X (Å)", backgroundcolor="white", gridcolor="#E6E6E6"),
                   yaxis=dict(title="Y (Å)", backgroundcolor="white", gridcolor="#E6E6E6"),
                   zaxis=dict(title="Z (Å)", backgroundcolor="white", gridcolor="#E6E6E6"),
                   aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=55),
    )
    figure.write_html(str(html_path), include_plotlyjs=True, full_html=True, auto_open=False)


def print_statistics(data: Data, affinity_molar: float, affinity_kind: str, qualifier: str) -> None:
    """Print a concise, aligned scientific audit table."""
    rows = [("Complex PDB ID", data.pdb_id),
            ("Experimental binding affinity (-log Kd/Ki)", f"{data.y.item():.4f} pK"),
            ("Original affinity", f"{affinity_kind}{qualifier} {affinity_molar:.4g} M"),
            ("Protein Heavy Atoms", str(data.num_protein_atoms)),
            ("Ligand Heavy Atoms", str(data.num_ligand_atoms)),
            ("Pocket Atoms within 5 Å", str(data.num_pocket_atoms)),
            ("Intra-molecular Covalent Edges", str(data.num_intra_molecular_covalent_edges)),
            ("Interface Contact Edges (≤5 Å)", str(data.num_interface_contact_edges)),
            ("Graph data saved", str(GRAPH_FILE))]
    width = max(len(key) for key, _ in rows)
    line_width = max(72, width + max(len(value) for _, value in rows) + 7)
    print("\n" + "=" * line_width)
    print("  Protein–Ligand Dual Graph: Scientific Summary")
    print("=" * line_width)
    for key, value in rows:
        print(f"  {key:<{width}} : {value}")
    print("=" * line_width)


def validate_inputs() -> None:
    for path in (PROTEIN_FILE, INDEX_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")
    if not LIGAND_SDF.is_file() and not LIGAND_MOL2.is_file():
        raise FileNotFoundError(f"Neither ligand file exists: {LIGAND_SDF} / {LIGAND_MOL2}")


def main() -> None:
    validate_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    affinity, molar, affinity_kind, qualifier = parse_affinity(INDEX_FILE, PDB_ID)
    protein = parse_protein(PROTEIN_FILE)
    ligand, ligand_source = read_ligand(LIGAND_SDF, LIGAND_MOL2)
    data, interface_pairs, _, _ = build_graph(protein, ligand, affinity, ligand_source)
    data.num_pocket_atoms = data.num_ligand_atoms + len({protein_index for protein_index, _ in interface_pairs})
    torch.save(data, GRAPH_FILE)
    render_pocket_graph(data, interface_pairs, IMAGE_FILE)
    render_interactive_3d(data, interface_pairs, HTML_FILE)
    print_statistics(data, molar, affinity_kind, qualifier)
    print(f"\nPyG graph saved: {GRAPH_FILE}")
    print(f"PNG image generated successfully: {IMAGE_FILE}\n")
    print(f"Interactive 3D HTML generated successfully: {HTML_FILE}\n")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)

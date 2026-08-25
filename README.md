# EquiAffinity: Equivariant Graph Neural Networks for Protein-Ligand Binding Affinity Prediction

EquiAffinity is a research project for predicting protein-ligand binding affinity with atom-level molecular graphs. It uses PyTorch Geometric and e3nn to learn SE(3)/E(3)-equivariant molecular representations that preserve the geometric structure of protein-ligand complexes.

## Phase 1

- Atom-level protein-ligand dual-graph construction
- Extraction of protein-ligand interface contact edges within 5 Å
- 2D pocket-topology visualisation
- Interactive 3D pocket-graph visualisation exported as standalone HTML

## Quick start

Install the scientific Python dependencies, then run:

```bash
python src/build_and_visualize.py
```

For the `1a0q` test complex, this writes the graph to `outputs/1a0q.pt`, a 2D PNG, and an interactive 3D HTML visualisation.

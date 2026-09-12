"""Extract PLIP protein--ligand interactions for the local 1a0q complex.

The script accepts any complete protein--ligand PDB.  For this repository's
Phase-1 layout it can also create a temporary complex PDB by appending the
``1a0q_ligand.sdf`` (or MOL2 fallback) coordinates to ``1a0q_protein.pdb``.

Run from the project root after installing ``plip`` and ``pandas``:

    python -m src.analysis.plip_extractor
"""

from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pandas as pd
    from plip.structure.preparation import PDBComplex
except ImportError as error:  # pragma: no cover - environment-dependent dependency
    _DEPENDENCY_ERROR = error
else:
    _DEPENDENCY_ERROR = None


PDB_ID = "1a0q"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPLEX_DIR = PROJECT_ROOT / "data" / "P-L" / "1981-2000" / PDB_ID
PROTEIN_FILE = COMPLEX_DIR / f"{PDB_ID}_protein.pdb"
SDF_FILE = COMPLEX_DIR / f"{PDB_ID}_ligand.sdf"
MOL2_FILE = COMPLEX_DIR / f"{PDB_ID}_ligand.mol2"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CSV_FILE = OUTPUT_DIR / f"{PDB_ID}_plip_interactions.csv"
GENERATED_COMPLEX_FILE = OUTPUT_DIR / f"{PDB_ID}_complex_for_plip.pdb"


def _require_dependencies() -> None:
    if _DEPENDENCY_ERROR is not None:
        raise RuntimeError(
            "PLIP analysis requires pandas and PLIP. Install compatible packages, for example: "
            "python -m pip install pandas plip"
        ) from _DEPENDENCY_ERROR


def _attribute(value: Any, *names: str, default: Any = "") -> Any:
    """Return the first available PLIP attribute, tolerating PLIP API versions."""
    for name in names:
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return default


def _residue_label(interaction: Any) -> str:
    residue_type = _attribute(interaction, "restype", "resname", default="UNK")
    residue_number = _attribute(interaction, "resnr", "resnum", default="?")
    chain = _attribute(interaction, "reschain", "chain", default="?")
    return f"{residue_type} {chain}:{residue_number}"


def _residue_chain(interaction: Any) -> str:
    """Return the protein chain identifier from a PLIP interaction object."""
    return str(_attribute(interaction, "reschain", "chain", default="?"))


def _atom_label(atom: Any, fallback: Any = "?") -> str:
    """Render atom name/index from PLIP atom descriptors without assuming fields."""
    if atom is None:
        return str(fallback)
    name = _attribute(atom, "name", "type", default=None)
    index = _attribute(atom, "idx", "orig_idx", "id", default=None)
    if name is not None and index is not None:
        return f"{name} ({index})"
    if name is not None:
        return str(name)
    return str(fallback)


def _numeric(interaction: Any, *names: str) -> Optional[float]:
    value = _attribute(interaction, *names, default=None)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _record(interaction_type: str, interaction: Any, *, protein_atom_names: Sequence[str], ligand_atom_names: Sequence[str],
            distance_names: Sequence[str] = ("distance",)) -> Dict[str, Any]:
    """Make a uniform, CSV-ready interaction record from a PLIP object."""
    donor = _attribute(interaction, "donor", "don", "d", default=None)
    acceptor = _attribute(interaction, "acceptor", "acc", "a", default=None)
    protein_atom = _attribute(interaction, "protatom", "proteinatom", "protcarbon", "bsatom", default=None)
    ligand_atom = _attribute(interaction, "ligatom", "ligandatom", "ligcarbon", default=None)
    donor_index = _attribute(interaction, "d_orig_idx", "donoridx", default="?")
    acceptor_index = _attribute(interaction, "a_orig_idx", "acceptoridx", default="?")
    if donor is not None or acceptor is not None:
        atom_a, atom_b = _atom_label(donor, donor_index), _atom_label(acceptor, acceptor_index)
    else:
        atom_a, atom_b = _atom_label(protein_atom, _attribute(interaction, "protcarbonidx", default="?")), _atom_label(ligand_atom, _attribute(interaction, "ligcarbonidx", default="?"))
    return {
        "interaction_type": interaction_type,
        "residue": _residue_label(interaction),
        "res_chain": _residue_chain(interaction),
        "distance": _numeric(interaction, *distance_names),
        "details": f"protein_atom={atom_a}; ligand_atom={atom_b}; angle={_numeric(interaction, 'angle', 'don_angle')!s}",
    }


def _interaction_set_for_ligand(complex_object: Any) -> Tuple[Any, Any]:
    """Characterise the ligand and use PLIP's dynamically generated set key."""
    ligands = list(complex_object.ligands)
    if not ligands:
        raise ValueError("PLIP found no ligand in the supplied complex PDB.")
    ligand = next((item for item in ligands if str(_attribute(item, "hetid", default="")).upper() == "LIG"), ligands[0])
    complex_object.characterize_complex(ligand)
    # PLIP keys vary by version and input: e.g. LIG:Z:1 rather than ``LIG``.
    # Never reconstruct the key from ligand metadata; inspect PLIP's result.
    if not complex_object.interaction_sets:
        raise ValueError("PLIP produced no interaction set after ligand characterisation.")
    return ligand, next(iter(complex_object.interaction_sets.values()))


def extract_plip_interactions(pdb_path: str):
    """Extract PLIP contacts and return ``(structured_dict, pandas.DataFrame)``.

    Hydrogen bonds include donor/acceptor atoms, donor angle, and atom distance.
    Hydrophobic contacts, salt bridges, and pi-stacking are represented by the
    same stable table schema, with unavailable geometry left blank.
    """
    _require_dependencies()
    input_path = Path(pdb_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Complex PDB not found: {input_path}")
    complex_object = PDBComplex()
    complex_object.load_pdb(str(input_path))
    ligand, interactions = _interaction_set_for_ligand(complex_object)
    records: List[Dict[str, Any]] = []
    # hbonds_pdon and hbonds_ldon encode the two donor orientations. They are
    # intentionally concatenated instead of using hbonds_all to avoid PLIP
    # version differences in whether that convenience collection exists.
    for hbond in list(_attribute(interactions, "hbonds_pdon", default=[])) + list(_attribute(interactions, "hbonds_ldon", default=[])):
        records.append(_record("Hydrogen Bond", hbond, protein_atom_names=(), ligand_atom_names=(),
                               distance_names=("distance_ad", "distance")))
    for contact in _attribute(interactions, "hydrophobic_contacts", default=[]):
        records.append(_record("Hydrophobic Contact", contact, protein_atom_names=(), ligand_atom_names=(),
                               distance_names=("distance",)))
    for bridge in list(_attribute(interactions, "saltbridge_lneg", default=[])) + list(_attribute(interactions, "saltbridge_pneg", default=[])):
        records.append(_record("Salt Bridge", bridge, protein_atom_names=(), ligand_atom_names=(),
                               distance_names=("distance",)))
    for stack in _attribute(interactions, "pistacking", default=[]):
        record = _record("Pi-Stacking", stack, protein_atom_names=(), ligand_atom_names=(), distance_names=("distance",))
        record["details"] += f"; offset={_numeric(stack, 'offset')!s}; type={_attribute(stack, 'type', default='')}"
        records.append(record)
    columns = ["interaction_type", "residue", "res_chain", "distance", "details"]
    dataframe = pd.DataFrame(records, columns=columns)
    result = {
        "pdb_path": str(input_path),
        "ligand": str(_attribute(ligand, "longname", default="LIG")),
        "interactions": records,
        "counts": {kind: sum(item["interaction_type"] == kind for item in records)
                   for kind in ("Hydrogen Bond", "Hydrophobic Contact", "Salt Bridge", "Pi-Stacking")},
    }
    return result, dataframe


def _parse_sdf(path: Path) -> Tuple[List[Tuple[str, float, float, float]], List[Tuple[int, int, int]]]:
    """Parse V2000 SDF coordinates and bond orders without requiring RDKit."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 4:
        raise ValueError(f"Malformed SDF: {path}")
    counts = lines[3]
    try:
        atom_count, bond_count = int(counts[:3]), int(counts[3:6])
    except ValueError as error:
        raise ValueError(f"Unsupported SDF counts line in {path}") from error
    atoms = []
    for line in lines[4:4 + atom_count]:
        atoms.append((line[31:34].strip() or "C", float(line[:10]), float(line[10:20]), float(line[20:30])))
    bonds = []
    for line in lines[4 + atom_count:4 + atom_count + bond_count]:
        bonds.append((int(line[:3]), int(line[3:6]), int(line[6:9])))
    return atoms, bonds


def _parse_mol2(path: Path) -> Tuple[List[Tuple[str, float, float, float]], List[Tuple[int, int, int]]]:
    """Parse the coordinates and connectivity needed for a minimal PDB ligand."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        atom_start = lines.index("@<TRIPOS>ATOM") + 1
        bond_start = lines.index("@<TRIPOS>BOND") + 1
    except ValueError as error:
        raise ValueError(f"Malformed MOL2: {path}") from error
    atoms = []
    for line in lines[atom_start:bond_start - 1]:
        fields = line.split()
        if len(fields) < 6:
            continue
        element = re.sub(r"[^A-Za-z]", "", fields[5]).upper()[:2]
        if element not in {"CL", "BR"}:
            element = element[:1]
        atoms.append((element or "C", float(fields[2]), float(fields[3]), float(fields[4])))
    bonds = []
    for line in lines[bond_start:]:
        if line.startswith("@<TRIPOS>"):
            break
        fields = line.split()
        if len(fields) >= 4:
            bond_type = 1 if fields[3].lower() in {"ar", "am"} else int(float(fields[3]))
            bonds.append((int(fields[1]), int(fields[2]), bond_type))
    return atoms, bonds


def build_complex_pdb(protein_path: Path, ligand_sdf: Path, ligand_mol2: Path, output_path: Path) -> Path:
    """Create a PLIP-ready PDB by combining protein records with local ligand geometry."""
    if not protein_path.is_file():
        raise FileNotFoundError(f"Protein PDB not found: {protein_path}")
    try:
        atoms, bonds = _parse_sdf(ligand_sdf)
    except Exception:
        if not ligand_mol2.is_file():
            raise
        atoms, bonds = _parse_mol2(ligand_mol2)
    protein_lines = [line.rstrip("\n") for line in protein_path.read_text(encoding="utf-8", errors="replace").splitlines()
                     if not line.startswith(("END", "CONECT"))]
    serials = [int(line[6:11]) for line in protein_lines if line.startswith(("ATOM  ", "HETATM")) and line[6:11].strip().isdigit()]
    first_serial = max(serials, default=0) + 1
    ligand_lines = []
    for index, (element, x, y, z) in enumerate(atoms, start=1):
        serial = first_serial + index - 1
        atom_name = f"{element}{index}"[:4]
        ligand_lines.append(f"HETATM{serial:5d} {atom_name:>4s} LIG Z   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}")
    conect_lines = []
    for start, end, order in bonds:
        source, target = first_serial + start - 1, first_serial + end - 1
        conect_lines.append(f"CONECT{source:5d}" + f"{target:5d}" * max(1, min(order, 3)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(protein_lines + ligand_lines + conect_lines + ["END"]) + "\n", encoding="utf-8")
    return output_path


def _markdown_table(dataframe) -> str:
    """Render a dependency-free Markdown summary of residue, type, and distance."""
    lines = ["| Residue | Chain | Interaction | Distance (Å) | Details |",
             "|---|---|---|---:|---|"]
    for _, row in dataframe.iterrows():
        distance_value = row["distance"]
        distance = "" if distance_value is None or (isinstance(distance_value, float) and math.isnan(distance_value)) else f"{distance_value:.2f}"
        lines.append(f"| {row['residue']} | {row['res_chain']} | {row['interaction_type']} | {distance} | {row['details']} |")
    return "\n".join(lines)


def main() -> None:
    """Prepare local 1a0q, extract interactions, export CSV, and print summary."""
    complex_path = build_complex_pdb(PROTEIN_FILE, SDF_FILE, MOL2_FILE, GENERATED_COMPLEX_FILE)
    result, dataframe = extract_plip_interactions(str(complex_path))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(CSV_FILE, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"PLIP analysis completed for {PDB_ID} ({result['ligand']}).")
    print("\n" + _markdown_table(dataframe))
    print(f"\nInteraction CSV saved: {CSV_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"PLIP EXTRACTION FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        raise

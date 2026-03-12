#!/usr/bin/env python3
"""
Generate OpenEdge/SEED multilayer surface-interaction tables using RustBCA.

The target is described by a TOML file. Example:

    ion = "oxygen"
    target_species = ["tungsten", "oxygen"]

    [[layer]]
    thickness_angstrom = 50.0
    densities_m3 = [6.306e28, 1.0e28]

    [[layer]]
    thickness_angstrom = 1.0e6
    densities_m3 = [6.306e28, 0.0]

The script writes total sputtering, reflection, and implantation data derived
from RustBCA's full particle output.
"""

from __future__ import annotations

import argparse
import time
import tomllib
from pathlib import Path

import numpy as np

import materials as materials_db


def _build_material_lookup() -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for value in vars(materials_db).values():
        if not isinstance(value, dict):
            continue
        symbol = value.get("symbol")
        name = value.get("name")
        if symbol:
            lookup[str(symbol).lower()] = value
        if name:
            lookup[str(name).lower()] = value
    return lookup


MATERIALS = _build_material_lookup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate OpenEdge multilayer sputtering/reflection tables from a TOML target description."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("multilayer_o_on_w.toml"),
        help="TOML file describing the incident ion and target layers.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("multilayer_surface_data.h5"),
        help="Output HDF5 file path.",
    )
    parser.add_argument("--e-min", type=float, default=10.0, help="Minimum incident energy [eV].")
    parser.add_argument("--e-max", type=float, default=1000.0, help="Maximum incident energy [eV].")
    parser.add_argument("--n-energy", type=int, default=80, help="Number of incident energy points.")
    parser.add_argument("--a-min", type=float, default=0.0, help="Minimum incident angle [deg].")
    parser.add_argument("--a-max", type=float, default=89.9, help="Maximum incident angle [deg].")
    parser.add_argument("--n-angle", type=int, default=80, help="Number of incident angle points.")
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Monte Carlo ions per (E, A) point.",
    )
    parser.add_argument(
        "--angle-floor",
        type=float,
        default=1.0e-4,
        help="Angles below this are evaluated at this floor to avoid exact-normal edge cases.",
    )
    parser.add_argument(
        "--implant-depth-max",
        type=float,
        default=1000.0,
        help="Maximum implantation depth [Angstrom] for stored histograms.",
    )
    parser.add_argument(
        "--implant-depth-bins",
        type=int,
        default=100,
        help="Number of implantation-depth bins.",
    )
    return parser.parse_args()


def resolve_material(name: str) -> dict:
    material = MATERIALS.get(name.lower())
    if material is None:
        raise KeyError(f"Unknown material '{name}'. Add it to materials.py or use an existing symbol/name.")
    required = ("Z", "m", "Ec", "Es")
    missing = [key for key in required if key not in material]
    if missing:
        raise KeyError(f"Material '{name}' is missing required keys: {missing}")
    if "Eb" not in material:
        material = dict(material)
        material["Eb"] = 0.0
    return material


def load_config(path: Path) -> tuple[dict, list[dict], np.ndarray, np.ndarray]:
    data = tomllib.loads(path.read_text())

    ion = resolve_material(str(data["ion"]))
    species = [resolve_material(str(name)) for name in data["target_species"]]
    layers = data["layer"]
    if not layers:
        raise ValueError("Config must define at least one [[layer]] table.")

    dx = []
    n2_m3 = []
    for index, layer in enumerate(layers):
        thickness = float(layer["thickness_angstrom"])
        densities = np.asarray(layer["densities_m3"], dtype=np.float64)
        if densities.shape != (len(species),):
            raise ValueError(
                f"Layer {index} densities_m3 length {densities.size} does not match target_species length {len(species)}."
            )
        dx.append(thickness)
        n2_m3.append(densities)

    return ion, species, np.asarray(n2_m3, dtype=np.float64), np.asarray(dx, dtype=np.float64)


def implantation_histogram(depths: np.ndarray, depth_edges: np.ndarray) -> np.ndarray:
    if depths.size == 0:
        return np.zeros(depth_edges.size - 1, dtype=np.float64)
    clipped = np.clip(depths, depth_edges[0], np.nextafter(depth_edges[-1], depth_edges[0]))
    hist, _ = np.histogram(clipped, bins=depth_edges)
    return hist.astype(np.float64)


def main() -> None:
    args = parse_args()
    import h5py
    from libRustBCA import compound_bca_list_1D_py

    if args.n_energy < 2 or args.n_angle < 2:
        raise ValueError("n-energy and n-angle must both be >= 2.")
    if args.samples < 1:
        raise ValueError("samples must be >= 1.")

    ion, species, n2_m3, dx = load_config(args.config)

    target_z = np.asarray([item["Z"] for item in species], dtype=np.float64)
    target_m = np.asarray([item["m"] for item in species], dtype=np.float64)
    target_ec = np.asarray([item["Ec"] for item in species], dtype=np.float64)
    target_es = np.asarray([item["Es"] for item in species], dtype=np.float64)
    target_eb = np.asarray([item.get("Eb", 0.0) for item in species], dtype=np.float64)
    energies = np.linspace(args.e_min, args.e_max, args.n_energy, dtype=np.float64)
    angles = np.linspace(args.a_min, args.a_max, args.n_angle, dtype=np.float64)
    implant_depth_edges = np.linspace(
        0.0, args.implant_depth_max, args.implant_depth_bins + 1, dtype=np.float64
    )

    spyld = np.zeros((args.n_energy, args.n_angle), dtype=np.float64)
    rfyld = np.zeros((args.n_energy, args.n_angle), dtype=np.float64)
    implyld = np.zeros((args.n_energy, args.n_angle), dtype=np.float64)
    implant_depth_counts = np.zeros(
        (args.n_energy, args.n_angle, args.implant_depth_bins), dtype=np.float64
    )
    implant_depth_pdf = np.zeros_like(implant_depth_counts)
    implant_mean_depth = np.zeros((args.n_energy, args.n_angle), dtype=np.float64)

    n2_angstrom = n2_m3 / 1.0e30
    total = args.n_energy * args.n_angle
    start = time.time()
    idx = 0

    for ie, energy in enumerate(energies):
        for ia, angle in enumerate(angles):
            angle_eval = max(float(angle), args.angle_floor)

            theta = np.deg2rad(angle_eval)
            ux = float(np.cos(theta))
            uy = float(np.sin(theta))
            uz = 0.0

            output, incident, stopped = compound_bca_list_1D_py(
                [ux] * args.samples,
                [uy] * args.samples,
                [uz] * args.samples,
                [float(energy)] * args.samples,
                [ion["Z"]] * args.samples,
                [ion["m"]] * args.samples,
                [ion["Ec"]] * args.samples,
                [ion["Es"]] * args.samples,
                target_z.tolist(),
                target_m.tolist(),
                target_ec.tolist(),
                target_es.tolist(),
                target_eb.tolist(),
                n2_angstrom.tolist(),
                dx.tolist(),
            )

            particles = np.asarray(output, dtype=np.float64)
            incident = np.asarray(incident, dtype=bool)
            stopped = np.asarray(stopped, dtype=bool)

            reflected_mask = incident & ~stopped
            sputtered_mask = ~incident
            implanted_mask = incident & stopped

            n_reflected = np.count_nonzero(reflected_mask)
            n_sputtered = np.count_nonzero(sputtered_mask)
            n_implanted = np.count_nonzero(implanted_mask)

            rfyld[ie, ia] = n_reflected / args.samples
            spyld[ie, ia] = n_sputtered / args.samples
            implyld[ie, ia] = n_implanted / args.samples

            if n_implanted > 0:
                implant_depths = particles[implanted_mask, 3]
                counts = implantation_histogram(implant_depths, implant_depth_edges)
                implant_depth_counts[ie, ia, :] = counts
                implant_mean_depth[ie, ia] = float(np.mean(implant_depths))
                bin_widths = np.diff(implant_depth_edges)
                implant_depth_pdf[ie, ia, :] = counts / (n_implanted * bin_widths)

            idx += 1

        elapsed = time.time() - start
        done_frac = idx / total
        eta = elapsed / done_frac - elapsed if done_frac > 0.0 else 0.0
        print(
            f"[{ie + 1:4d}/{args.n_energy}] E={energy:8.2f} eV | "
            f"progress={100.0 * done_frac:6.2f}% | elapsed={elapsed:8.1f}s | eta={eta:8.1f}s",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as h5f:
        h5f.create_dataset("E", data=energies)
        h5f.create_dataset("A", data=angles)
        h5f.create_dataset("species_Z", data=target_z)
        h5f.create_dataset("spyld", data=spyld)
        h5f.create_dataset("rfyld", data=rfyld)
        h5f.create_dataset("implyld", data=implyld)
        h5f.create_dataset("implant_depth_edges_angstrom", data=implant_depth_edges)
        h5f.create_dataset("implant_depth_counts", data=implant_depth_counts)
        h5f.create_dataset("implant_depth_pdf", data=implant_depth_pdf)
        h5f.create_dataset("implant_mean_depth_angstrom", data=implant_mean_depth)

        h5f.attrs["ion_symbol"] = ion["symbol"]
        h5f.attrs["config_path"] = str(args.config)
        h5f.attrs["density_unit"] = "1/m^3 in config, converted to 1/angstrom^3 for RustBCA"
        h5f.attrs["layer_thickness_unit"] = "angstrom"
        h5f.attrs["implant_depth_pdf_definition"] = "Conditional PDF in depth given an implanted incident ion."

    total_time = time.time() - start
    print(f"\nWrote {args.output}")
    print(f"Total points: {total}, samples/point: {args.samples}, walltime: {total_time:.1f}s")
    print(f"spyld range: [{spyld.min():.6g}, {spyld.max():.6g}]")
    print(f"rfyld range: [{rfyld.min():.6g}, {rfyld.max():.6g}]")
    print(f"implyld range: [{implyld.min():.6g}, {implyld.max():.6g}]")


if __name__ == "__main__":
    main()

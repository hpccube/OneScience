"""Convert an MPtrj JSON file into ASE-LMDB shards for Equiformer V3."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Iterator

import ase
import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect

from onescience.modules.func_utils.uma_graph.radius_graph_pbc import radius_graph_pbc


def _to_atoms(identifier: str, record: dict) -> Atoms:
    """Convert one MPtrj record while preserving FairChem label conventions."""

    structure = record["structure"]
    sites = structure["sites"]
    numbers = [
        ase.data.atomic_numbers[site["species"][0]["element"]] for site in sites
    ]
    positions = [site["xyz"] for site in sites]
    atoms = Atoms(
        numbers,
        positions,
        cell=structure["lattice"]["matrix"],
        pbc=True,
    )

    # MPtrj stress is reported in kbar; retain the upstream ASE sign and units.
    stress = np.asarray(record["stress"], dtype=np.float32)
    stress = stress * (-0.1 * ase.units.GPa)
    energy = record["uncorrected_total_energy"]
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=energy,
        free_energy=energy,
        forces=record["force"],
        stress=stress,
    )
    atoms.info["sid"] = identifier
    return atoms


def _has_no_isolated_atoms(
    atoms: Atoms, cutoff: float, max_neighbors: int
) -> bool:
    # Match the upstream AtomsToGraphs + radius_graph_pbc(..., True) path.
    data = SimpleNamespace(
        pos=torch.as_tensor(atoms.positions, dtype=torch.float32),
        cell=torch.as_tensor(atoms.cell.array, dtype=torch.float32).view(1, 3, 3),
        pbc=torch.as_tensor(atoms.pbc, dtype=torch.bool).view(1, 3),
        natoms=torch.tensor([len(atoms)], dtype=torch.long),
    )
    edge_index, _, _ = radius_graph_pbc(
        data,
        cutoff,
        max_neighbors,
        True,
        pbc=data.pbc[0],
    )
    counts = torch.bincount(edge_index[1], minlength=len(atoms))
    return bool(torch.all(counts > 0))


def _init_worker() -> None:
    """Keep each conversion worker single-threaded to avoid CPU oversubscription."""

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def _worker_ready() -> None:
    """Start the process pool before the parent opens an ASE database."""


def _convert_batch(
    batch: list[tuple[str, dict]], cutoff: float, max_neighbors: int
) -> list[Atoms | None]:
    converted: list[Atoms | None] = []
    for identifier, record in batch:
        atoms = _to_atoms(identifier, record)
        converted.append(
            atoms if _has_no_isolated_atoms(atoms, cutoff, max_neighbors) else None
        )
    return converted


def _iter_batches(
    records: Iterable[tuple[str, dict]], batch_size: int
) -> Iterator[list[tuple[str, dict]]]:
    iterator = iter(records)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def _convert_batches_ordered(
    records: Iterable[tuple[str, dict]],
    cutoff: float,
    max_neighbors: int,
    workers: int,
    batch_size: int,
    executor: ProcessPoolExecutor | None,
) -> Iterator[list[Atoms | None]]:
    batches = _iter_batches(records, batch_size)
    if executor is None:
        for batch in batches:
            yield _convert_batch(batch, cutoff, max_neighbors)
        return

    pending: deque[Future[list[Atoms | None]]] = deque()
    max_pending = workers * 2
    for batch in batches:
        pending.append(
            executor.submit(_convert_batch, batch, cutoff, max_neighbors)
        )
        if len(pending) >= max_pending:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


class _JSONStream:
    """Incrementally decode JSON values without loading the full MPtrj file."""

    def __init__(self, path: Path, chunk_size: int = 1 << 20):
        self.handle = path.open("r", encoding="utf-8")
        self.chunk_size = chunk_size
        self.buffer = ""
        self.decoder = json.JSONDecoder()
        self.eof = False

    def close(self) -> None:
        self.handle.close()

    def _fill(self) -> None:
        if not self.eof:
            chunk = self.handle.read(self.chunk_size)
            if chunk:
                self.buffer += chunk
            else:
                self.eof = True

    def _skip_whitespace(self) -> None:
        while True:
            stripped = self.buffer.lstrip()
            if stripped:
                self.buffer = stripped
                return
            self._fill()
            if not self.buffer and self.eof:
                raise EOFError("unexpected end of JSON input")

    def _take(self, token: str) -> None:
        self._skip_whitespace()
        if not self.buffer.startswith(token):
            raise ValueError(f"expected {token!r} in MPtrj JSON")
        self.buffer = self.buffer[len(token) :]

    def value(self):
        """Read one complete JSON value, filling until the decoder succeeds."""

        self._skip_whitespace()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer)
            except json.JSONDecodeError:
                if self.eof:
                    raise
                self._fill()
                continue
            self.buffer = self.buffer[end:]
            return value

    def iter_group_records(self) -> Iterator[tuple[str, dict]]:
        """Yield ``(record_id, record)`` pairs from the current group object."""

        self._take("{")
        self._skip_whitespace()
        if self.buffer.startswith("}"):
            self.buffer = self.buffer[1:]
            return
        while True:
            record_id = self.value()
            if not isinstance(record_id, str):
                raise ValueError("MPtrj record identifiers must be strings")
            self._take(":")
            record = self.value()
            if not isinstance(record, dict):
                raise ValueError("MPtrj records must be JSON objects")
            yield record_id, record
            self._skip_whitespace()
            if self.buffer.startswith(","):
                self.buffer = self.buffer[1:]
                continue
            self._take("}")
            return

    def iter_records(self) -> Iterator[tuple[str, dict]]:
        """Yield all records from the two-level MPtrj root object."""

        self._take("{")
        self._skip_whitespace()
        if self.buffer.startswith("}"):
            self.buffer = self.buffer[1:]
            return
        while True:
            group_id = self.value()
            if not isinstance(group_id, str):
                raise ValueError("MPtrj group identifiers must be strings")
            self._take(":")
            for record_id, record in self.iter_group_records():
                # Upstream discards the outer group key and uses this ID as sid.
                yield record_id, record
            self._skip_whitespace()
            if self.buffer.startswith(","):
                self.buffer = self.buffer[1:]
                continue
            self._take("}")
            while not self.eof:
                self._fill()
            if self.buffer.strip():
                raise ValueError("trailing data after MPtrj JSON root object")
            self.buffer = ""
            return


def _count_records(
    input_path: Path,
    max_records: int | None,
    progress_every: int,
) -> int:
    """Count records without retaining the full upstream JSON object in memory."""

    stream = _JSONStream(input_path)
    started = time.monotonic()
    try:
        records = stream.iter_records()
        if max_records is not None:
            records = islice(records, max_records)
        count = 0
        for _ in records:
            count += 1
            if progress_every and count % progress_every == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    f"counted {count} input records; "
                    f"rate={count / elapsed:.1f} records/s",
                    flush=True,
                )
        print(
            f"counted {count} input records; starting conversion",
            flush=True,
        )
        return count
    finally:
        stream.close()


def _shard_sizes(total_records: int, shards: int) -> list[int]:
    """Return the same contiguous, remainder-first split used upstream."""

    chunk_size, remainder = divmod(total_records, shards)
    return [
        chunk_size + (1 if shard_index < remainder else 0)
        for shard_index in range(shards)
    ]


def convert(
    input_path: Path,
    output_dir: Path,
    cutoff: float,
    max_neighbors: int,
    shards: int,
    max_records: int | None = None,
    progress_every: int = 10_000,
    workers: int = 1,
    batch_size: int = 16,
) -> None:
    if shards < 1:
        raise ValueError("shards must be positive")
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be positive")
    if progress_every < 0:
        raise ValueError("progress_every cannot be negative")
    if workers < 1:
        raise ValueError("workers must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.iterdir())
    if existing:
        raise FileExistsError(
            f"refusing to write into non-empty output directory {output_dir}; "
            "choose a new directory or clear it explicitly"
        )
    total_records = _count_records(input_path, max_records, progress_every)
    shard_sizes = _shard_sizes(total_records, shards)
    atom_counts: list[int] = []
    written = 0
    examined = 0
    next_progress = progress_every
    conversion_started = time.monotonic()
    executor: ProcessPoolExecutor | None = None
    if workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("fork"),
            initializer=_init_worker,
        )
        # ProcessPoolExecutor launches all workers together on its first submit.
        executor.submit(_worker_ready).result()
    stream = _JSONStream(input_path)
    try:
        records = stream.iter_records()
        if max_records is not None:
            records = islice(records, max_records)

        for shard_index, shard_size in enumerate(shard_sizes):
            # Zero padding preserves numeric shard order in AseDBDataset, whose
            # directory discovery sorts paths lexicographically.
            database = connect(
                str(output_dir / f"data_{shard_index:05d}.aselmdb")
            )
            shard_written = 0
            try:
                shard_records = islice(records, shard_size)
                for converted in _convert_batches_ordered(
                    shard_records,
                    cutoff,
                    max_neighbors,
                    workers,
                    batch_size,
                    executor,
                ):
                    for atoms in converted:
                        examined += 1
                        if atoms is not None:
                            database.write(atoms, data=atoms.info)
                            atom_counts.append(len(atoms))
                            shard_written += 1
                            written += 1
                        if progress_every and examined >= next_progress:
                            elapsed = max(
                                time.monotonic() - conversion_started, 1e-9
                            )
                            rate = examined / elapsed
                            remaining = total_records - examined
                            eta_seconds = remaining / rate if rate else float("inf")
                            print(
                                f"processed {examined}/{total_records} records; "
                                f"wrote {written}; filtered {examined - written}; "
                                f"rate={rate:.1f} records/s; "
                                f"eta={eta_seconds / 60:.1f} min",
                                flush=True,
                            )
                            next_progress += progress_every
            finally:
                database.close()
            print(
                f"finished shard {shard_index}: examined {shard_size}; "
                f"wrote {shard_written}",
                flush=True,
            )
    finally:
        stream.close()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    if examined != total_records:
        raise RuntimeError(
            f"MPtrj input changed while converting: counted {total_records} "
            f"records but read {examined}"
        )

    np.savez(
        output_dir / "metadata.npz",
        natoms=np.asarray(atom_counts, dtype=np.int64),
    )
    print(f"wrote {written} structures to {output_dir} ({shards} shards)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="MPtrj JSON file")
    parser.add_argument(
        "--output", type=Path, required=True, help="ASE-LMDB output directory"
    )
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--max-neighbors", type=int, default=1000)
    parser.add_argument("--shards", type=int, default=15)
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="optional maximum input records to examine for bounded validation",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        help="print progress after this many input records; zero disables it",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel CPU conversion workers; results remain ordered",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="records sent to each worker task",
    )
    args = parser.parse_args()
    if args.shards < 1:
        parser.error("--shards must be positive")
    if args.max_records is not None and args.max_records < 1:
        parser.error("--max-records must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every cannot be negative")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    convert(
        args.input,
        args.output,
        args.cutoff,
        args.max_neighbors,
        args.shards,
        args.max_records,
        args.progress_every,
        args.workers,
        args.batch_size,
    )


if __name__ == "__main__":
    main()

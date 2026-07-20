import contextlib
import io
import sys
import warnings
from typing import Dict, Iterable, List, Tuple, Union

from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.constraints import FixSymmetry
from ase.filters import ExpCellFilter, FrechetCellFilter, Filter
from ase.optimize import BFGS, FIRE
from ase.optimize.optimize import Optimizer
from ase.units import GPa
from deprecated import deprecated
from loguru import logger
from tqdm import tqdm

from onescience.datapipes.materials.mattersim import build_dataloader

from .potential import Potential


class Relaxer(object):
    """Relaxer is a class for structural relaxation with fixed volume."""

    SUPPORTED_OPTIMIZERS = {"BFGS": BFGS, "FIRE": FIRE}
    SUPPORTED_FILTERS = {
        "EXPCELLFILTER": ExpCellFilter,
        "FRECHETCELLFILTER": FrechetCellFilter,
    }

    def __init__(
        self,
        optimizer: Union[Optimizer, str] = "FIRE",
        filter: Union[Filter, str, None] = None,
        constrain_symmetry: bool = True,
        fix_axis: Union[bool, Iterable[bool]] = False,
    ) -> None:
        """
        Args:
            optimizer (Union[Optimizer, str]): The optimizer to use.
            filter (Union[Filter, str, None]): The filter to use.
            constrain_symmetry (bool): Whether to constrain the symmetry.
            fix_axis (Union[bool, Iterable[bool]]): Whether to fix the axis.
        """
        self.optimizer = (
            self.SUPPORTED_OPTIMIZERS[optimizer.upper()]
            if isinstance(optimizer, str)
            else optimizer
        )
        self.relax_cell = filter is not None
        if filter is not None:
            self.filter = (
                self.SUPPORTED_FILTERS[filter.upper()]
                if isinstance(filter, str)
                else filter
            )
        self.constrain_symmetry = constrain_symmetry
        self.fix_axis = fix_axis

    def relax(
        self,
        atoms: Atoms,
        steps: int = 500,
        fmax: float = 0.01,
        params_filter: dict = {},
        verbose: bool = True,
        **kwargs,
    ) -> Tuple[bool, Atoms]:
        """
        Relax the atoms object.

        Args:
            atoms (Atoms): The atoms object to relax.
            steps (int): The maximum number of steps to take.
            fmax (float): The maximum force allowed.
            params_filter (dict): The parameters for the filter.
            verbose (bool): If True, print optimizer progress. If False,
                suppress all output during relaxation.
            kwargs: Additional keyword arguments for the optimizer.
        """

        if atoms.calc is None:
            raise ValueError("Atoms object must have a calculator.")

        if self.constrain_symmetry:
            atoms.set_constraint(FixSymmetry(atoms))

        if self.relax_cell:
            # Set the mask for the fixed axis
            if isinstance(self.fix_axis, bool):
                mask = [not self.fix_axis for i in range(6)]
            else:
                assert (
                    len(self.fix_axis) == 6
                ), "The length of fix_axis list not equal 6."
                mask = [not elem for elem in self.fix_axis]

            # check if the scalar_pressure is provided
            if (
                "scalar_pressure" in params_filter
                and params_filter["scalar_pressure"] > 1
            ):
                warnings.warn(
                    "The scalar_pressure used in ExpCellFilter assumes "
                    "eV/A^3 unit and 1 eV/A^3 is already 160 GPa. "
                    "Please make sure you have converted your pressure "
                    "from GPa to eV/A^3 by dividing by 160.21766208."
                )
            ecf = self.filter(atoms, mask=mask, **params_filter)
        else:
            ecf = atoms

        stream = sys.stdout if verbose else io.StringIO()
        with contextlib.redirect_stdout(stream):
            optimizer = self.optimizer(ecf, **kwargs)
            optimizer.run(fmax=fmax, steps=steps)

        converged = optimizer.get_number_of_steps() < steps

        if self.constrain_symmetry:
            atoms.set_constraint(None)

        return converged, atoms

    @classmethod
    @deprecated(reason="Use cli/applications/relax_structure.py instead.")
    def relax_structures(
        cls,
        atoms: Union[Atoms, Iterable[Atoms]],
        optimizer: Union[Optimizer, str] = "FIRE",
        filter: Union[Filter, str, None] = None,
        constrain_symmetry: bool = False,
        fix_axis: Union[bool, Iterable[bool]] = False,
        pressure_in_GPa: Union[float, None] = None,
        **kwargs,
    ) -> Union[Tuple[bool, Atoms], Tuple[List[bool], List[Atoms]]]:
        """
        Args:
            atoms: (Union[Atoms, Iterable[Atoms]]):
                The Atoms object or an iterable of Atoms objects to relax.
            optimizer (Union[Optimizer, str]): The optimizer to use.
            filter (Union[Filter, str, None]): The filter to use.
            constrain_symmetry (bool): Whether to constrain the symmetry.
            fix_axis (Union[bool, Iterable[bool]]): Whether to fix the axis.
            **kwargs: Additional keyword arguments for the relax method.
        Returns:
            converged (Union[bool, List[bool]]):
                Whether the relaxation converged or a list of them
            Atoms (Union[Atoms, List[Atoms]]):
                The relaxed atoms object or a list of them
        """
        params_filter = {}

        if filter is None and pressure_in_GPa is None:
            pass
        elif filter is None and pressure_in_GPa is not None:
            filter = "ExpCellFilter"
            params_filter["scalar_pressure"] = (
                pressure_in_GPa * GPa
            )  # GPa = 1 / 160.21766208
        elif filter is not None and pressure_in_GPa is None:
            params_filter["scalar_pressure"] = 0.0
        else:
            params_filter["scalar_pressure"] = (
                pressure_in_GPa * GPa
            )  # GPa = / 160.21766208

        relaxer = Relaxer(
            optimizer=optimizer,
            filter=filter,
            constrain_symmetry=constrain_symmetry,
            fix_axis=fix_axis,
        )

        if isinstance(atoms, (list, tuple)):
            relaxed_results = relaxed_results = [
                relaxer.relax(atom, params_filter=params_filter, **kwargs)
                for atom in atoms
            ]
            converged, relaxed_atoms = zip(*relaxed_results)
            return list(converged), list(relaxed_atoms)
        else:
            return relaxer.relax(atoms, params_filter=params_filter, **kwargs)
class DummyBatchCalculator(Calculator):
    def __init__(self):
        super().__init__()

    def calculate(self, atoms=None, properties=None, system_changes=None):
        pass

    def get_potential_energy(self, atoms=None):
        return atoms.info["total_energy"]

    def get_forces(self, atoms=None):
        return atoms.arrays["forces"]

    def get_stress(self, atoms=None):
        return units.GPa * atoms.info["stress"]


class BatchRelaxer(object):
    """BatchRelaxer is a class for batch structural relaxation.
    It is more efficient than Relaxer when relaxing a large number of structures."""

    SUPPORTED_OPTIMIZERS = {"BFGS": BFGS, "FIRE": FIRE}
    SUPPORTED_FILTERS = {
        "EXPCELLFILTER": ExpCellFilter,
        "FRECHETCELLFILTER": FrechetCellFilter,
    }

    def __init__(
        self,
        potential: Potential,
        optimizer: Union[str, type[Optimizer]] = "FIRE",
        filter: Union[type[Filter], str, None] = None,
        fmax: float = 0.05,
        max_natoms_per_batch: int = 512,
        max_n_steps: int = 1_000_000,
    ):
        self.potential = potential
        self.device = potential.device
        if isinstance(optimizer, str):
            if optimizer.upper() not in self.SUPPORTED_OPTIMIZERS:
                raise ValueError(f"Unsupported optimizer: {optimizer}")
            self.optimizer = self.SUPPORTED_OPTIMIZERS[optimizer.upper()]
        elif issubclass(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        if isinstance(filter, str):
            if filter.upper() not in self.SUPPORTED_FILTERS:
                raise ValueError(f"Unsupported filter: {filter}")
            self.filter = self.SUPPORTED_FILTERS[filter.upper()]
        elif filter is None or issubclass(filter, Filter):
            self.filter = filter
        else:
            raise ValueError(f"Unsupported filter: {filter}")
        self.fmax = fmax
        self.max_natoms_per_batch = max_natoms_per_batch
        self.optimizer_instances: List[Optimizer] = []
        self.is_active_instance: List[bool] = []
        self.finished = False
        self.total_converged = 0
        self.trajectories: Dict[int, List[Atoms]] = {}
        self.max_n_steps = max_n_steps 

    def insert(self, atoms: Atoms):
        atoms.calc = DummyBatchCalculator()
        optimizer_instance = self.optimizer(
            self.filter(atoms) if self.filter else atoms
        )
        optimizer_instance.fmax = self.fmax
        optimizer_instance.nsteps = 0
        self.optimizer_instances.append(optimizer_instance)
        self.is_active_instance.append(True)

    def step_batch(self):
        atoms_list = []
        for idx, opt in enumerate(self.optimizer_instances):
            if self.is_active_instance[idx]:
                atoms_list.append(opt.atoms)

        # Note: we use a batch size of len(atoms_list)
        # because we only want to run one batch at a time
        dataloader = build_dataloader(
            atoms_list, batch_size=len(atoms_list), only_inference=True
        )
        energy_batch, forces_batch, stress_batch = self.potential.predict_properties(
            dataloader, include_forces=True, include_stresses=True
        )

        counter = 0
        self.finished = True
        for idx, opt in enumerate(self.optimizer_instances):
            if self.is_active_instance[idx]:
                # Set the properties so the dummy calculator can
                # return them within the optimizer step
                opt.atoms.info["total_energy"] = energy_batch[counter]
                opt.atoms.arrays["forces"] = forces_batch[counter]
                opt.atoms.info["stress"] = stress_batch[counter]
                try:
                    self.trajectories[opt.atoms.info["structure_index"]].append(
                        opt.atoms.copy()
                    )
                except KeyError:
                    self.trajectories[opt.atoms.info["structure_index"]] = [
                        opt.atoms.copy()
                    ]

                opt.step()
                opt.nsteps += 1
                # Get gradient for convergence check
                # Note: gradient = -forces for the optimizable object
                gradient = opt.optimizable.get_gradient()
                if opt.converged(gradient) or opt.nsteps >= self.max_n_steps:
                    self.is_active_instance[idx] = False
                    self.total_converged += 1
                    if self.total_converged % 100 == 0:
                        logger.info(f"Relaxed {self.total_converged} structures.")
                else:
                    self.finished = False
                counter += 1

        # remove inactive instances
        self.optimizer_instances = [
            opt
            for opt, active in zip(self.optimizer_instances, self.is_active_instance)
            if active
        ]
        self.is_active_instance = [True] * len(self.optimizer_instances)

    def relax(
        self,
        atoms_list: List[Atoms],
    ) -> Dict[int, List[Atoms]]:
        self.trajectories = {}
        self.tqdmcounter = tqdm(total=len(atoms_list), file=sys.stdout)
        pointer = 0
        atoms_list_ = []
        for i in range(len(atoms_list)):
            atoms_list_.append(atoms_list[i].copy())
            atoms_list_[i].info["structure_index"] = i

        while (
            pointer < len(atoms_list) or not self.finished
        ):  # While there are unfinished instances or atoms left to insert
            while pointer < len(atoms_list) and (
                sum([len(opt.atoms) for opt in self.optimizer_instances])
                + len(atoms_list[pointer])
                <= self.max_natoms_per_batch
            ):
                # While there are enough n_atoms slots in the
                # batch and we have not reached the end of the list.
                self.insert(
                    atoms_list_[pointer]
                )  # Insert new structure to fire instances
                self.tqdmcounter.update(1)
                pointer += 1
            self.step_batch()
        self.tqdmcounter.close()

        return self.trajectories

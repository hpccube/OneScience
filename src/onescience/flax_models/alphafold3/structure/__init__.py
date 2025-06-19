

"""Structure module initialization."""

# pylint: disable=g-importing-member
from onescience.flax_models.alphafold3.structure.bioassemblies import BioassemblyData
from onescience.flax_models.alphafold3.structure.bonds import Bonds
from onescience.flax_models.alphafold3.structure.chemical_components import ChemCompEntry
from onescience.flax_models.alphafold3.structure.chemical_components import ChemicalComponentsData
from onescience.flax_models.alphafold3.structure.chemical_components import get_data_for_ccd_components
from onescience.flax_models.alphafold3.structure.chemical_components import populate_missing_ccd_data
from onescience.flax_models.alphafold3.structure.mmcif import BondParsingError
from onescience.flax_models.alphafold3.structure.parsing import BondAtomId
from onescience.flax_models.alphafold3.structure.parsing import from_atom_arrays
from onescience.flax_models.alphafold3.structure.parsing import from_mmcif
from onescience.flax_models.alphafold3.structure.parsing import from_parsed_mmcif
from onescience.flax_models.alphafold3.structure.parsing import from_res_arrays
from onescience.flax_models.alphafold3.structure.parsing import from_sequences_and_bonds
from onescience.flax_models.alphafold3.structure.parsing import ModelID
from onescience.flax_models.alphafold3.structure.parsing import NoAtomsError
from onescience.flax_models.alphafold3.structure.parsing import SequenceFormat
from onescience.flax_models.alphafold3.structure.structure import ARRAY_FIELDS
from onescience.flax_models.alphafold3.structure.structure import AuthorNamingScheme
from onescience.flax_models.alphafold3.structure.structure import Bond
from onescience.flax_models.alphafold3.structure.structure import CascadeDelete
from onescience.flax_models.alphafold3.structure.structure import concat
from onescience.flax_models.alphafold3.structure.structure import enumerate_residues
from onescience.flax_models.alphafold3.structure.structure import fix_non_standard_polymer_residues
from onescience.flax_models.alphafold3.structure.structure import GLOBAL_FIELDS
from onescience.flax_models.alphafold3.structure.structure import make_empty_structure
from onescience.flax_models.alphafold3.structure.structure import MissingAtomError
from onescience.flax_models.alphafold3.structure.structure import MissingAuthorResidueIdError
from onescience.flax_models.alphafold3.structure.structure import multichain_residue_index
from onescience.flax_models.alphafold3.structure.structure import stack
from onescience.flax_models.alphafold3.structure.structure import Structure
from onescience.flax_models.alphafold3.structure.structure_tables import Atoms
from onescience.flax_models.alphafold3.structure.structure_tables import Chains
from onescience.flax_models.alphafold3.structure.structure_tables import Residues

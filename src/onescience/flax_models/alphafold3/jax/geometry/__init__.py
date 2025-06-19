

"""Geometry Module."""

from onescience.flax_models.alphafold3.jax.geometry import rigid_matrix_vector
from onescience.flax_models.alphafold3.jax.geometry import rotation_matrix
from onescience.flax_models.alphafold3.jax.geometry import struct_of_array
from onescience.flax_models.alphafold3.jax.geometry import vector

Rot3Array = rotation_matrix.Rot3Array
Rigid3Array = rigid_matrix_vector.Rigid3Array

StructOfArray = struct_of_array.StructOfArray

Vec3Array = vector.Vec3Array
square_euclidean_distance = vector.square_euclidean_distance
euclidean_distance = vector.euclidean_distance
dihedral_angle = vector.dihedral_angle
dot = vector.dot
cross = vector.cross

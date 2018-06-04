from pressure_solver_2D import MpfaD2D
from pressure_solver_2D import InterpolMethod
from mesh_preprocessor import MeshManager
import numpy as np


K_1 = np.array([100.0, 78.0, 0.0,
                78.0,   1.0, 0.0,
                0.0,    0.0,  1.0])

mesh = MeshManager('anisotropic_problem.msh', dim=2)
mesh.set_media_property('Permeability', {1: K_1}, dim_target=2)
mesh.set_boundary_condition('Dirichlet', {102: 1.0, 101: 0.0},
                            dim_target=1, set_nodes=True)
mesh.set_boundary_condition('Neumann', {201: 0.0},
                            dim_target=1, set_nodes=True)
mpfad = MpfaD2D(mesh)
imd = InterpolMethod(mesh, 1.0)

mpfad.run_solver(imd.by_lpew2)

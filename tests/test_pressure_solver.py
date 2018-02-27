import unittest
import numpy as np
from pressure_solver_2D import MpfaD2D
from mesh_preprocessor import MeshManager


class PressureSolverTest(unittest.TestCase):

    def setUp(self):

        K_1 = np.array([1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0])

        # K_2 = np.array([2.0, 0.0, 0.0,
        #                 0.0, 2.0, 0.0,
        #                 0.0, 0.0, 2.0])

        self.mesh_1 = MeshManager('mesh_test_1.msh', dim=2)
        self.mesh_1.set_media_property('Permeability', {1: K_1}, dim_target=2)
        self.mesh_1.set_boundary_condition('Dirichlet', {102: 1.0, 101: 0.0},
                                           dim_target=1, set_nodes=True)
        self.mesh_1.set_boundary_condition('Neumann', {201: 0.0},
                                           dim_target=1, set_nodes=True)
        self.mpfad_1 = MpfaD2D(self.mesh_1)

    # def test_linear_problem_mesh_1(self):
    #     self.mpfad_1.run_solver(self.mesh_1)
    #     for a_volume in self.mesh_1.all_volumes:
    #         local_pressure = self.mesh_1.mb.tag_get_data(
    #                          self.mpfad_1.pressure_tag, a_volume)
    #         coord_x = self.mesh_1.get_centroid(a_volume)[0]
    #         self.assertAlmostEqual(
    #             local_pressure[0][0], 1 - coord_x, delta=1e-10)

    def test_if_method_has_all_dirichlet_nodes(self):
        self.assertEqual(len(self.mpfad_1.dirichlet_nodes), 8)

    def test_if_method_has_all_neumann_nodes(self):
        self.assertEqual(len(self.mpfad_1.neumann_nodes), 4)

    def test_if_method_has_all_intern_nodes(self):
        self.assertEqual(len(self.mpfad_1.intern_nodes), 4)

    def test_if_method_has_all_dirichlet_faces(self):
        self.assertEqual(len(self.mpfad_1.dirichlet_faces), 6)
    #
    # def test_if_method_has_all_neumann_faces(self):
    #     self.assertEqual(len(self.mpfad_1.neumann_faces), 6)
    #
    # def test_if_method_has_all_intern_faces(self):
    #     self.assertEqual(len(self.mpfad_1.intern_faces), 12)

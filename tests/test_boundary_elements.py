import unittest
import numpy as np
from pressure_solver_2D import MpfaD2D
from mesh_preprocessor import MeshManager
# import nonconform_mesh_generator_test as nmg

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

        # self.mesh_2 = MeshManager('mesh_nonconform_test.vtk', dim=2)
        # mb = nmg.crazy_mesh()
        # self.mesh_2.load_data(mb)
        # self.mpfad_2 = MpfaD2D(self.mesh_2)

    def test_if_method_has_all_dirichlet_nodes(self):
        self.assertEqual(len(self.mpfad_1.dirichlet_nodes), 8)

    def test_if_method_has_all_neumann_nodes(self):
        self.assertEqual(len(self.mpfad_1.neumann_nodes), 4)

    def test_if_method_has_all_intern_nodes(self):
        self.assertEqual(len(self.mpfad_1.intern_nodes), 4)

    def test_if_method_has_all_dirichlet_faces(self):
        self.assertEqual(len(self.mpfad_1.dirichlet_faces), 6)

    def test_if_method_has_all_neumann_faces(self):
        self.assertEqual(len(self.mpfad_1.neumann_faces), 6)

    def test_if_method_has_all_faces(self):
        self.assertEqual(len(self.mpfad_1.all_faces), 24)

    def test_if_method_has_all_intern_faces(self):
        self.assertEqual(len(self.mpfad_1.intern_faces), 12)

    def test_matplot(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.plot([1,2,3])
        plt.savefig('myfig')

    # def test_load_data_neumann_faces_from_vtk_with_usable_data_on_it(self):
    #     self.assertEqual(len(self.mpfad_2.neumann_faces), 28)
    #
    # def test_load_data_dirichlet_faces_from_vtk_with_usable_data_on_it(self):
    #     self.assertEqual(len(self.mpfad_2.dirichlet_faces), 28)
    #
    # def test_load_data_dirichlet_nodes_from_vtk_with_usable_data_on_it(self):
    #     self.assertEqual(len(self.mpfad_2.dirichlet_nodes), 30)
    #
    # def test_load_data_neumann_nodes_from_vtk_with_usable_data_on_it(self):
    #     self.assertEqual(len(self.mpfad_2.neumann_nodes), 26)

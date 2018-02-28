import unittest
import numpy as np
from pressure_solver_2D import MpfaD2D
from mesh_preprocessor import MeshManager
from pressure_solver_2D import InterpolMethod


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
        self.imd = InterpolMethod(self.mpfad_1)

    def test_linear_problem_with_mesh_1(self):
        self.mpfad_1.run_solver()
        for a_volume in self.mesh_1.all_volumes:
            local_pressure = self.mesh_1.mb.tag_get_data(
                             self.mpfad_1.pressure_tag, a_volume)
            coord_x = self.mesh_1.get_centroid(a_volume)[0]
            self.assertAlmostEqual(
                local_pressure[0][0], 1 - coord_x, delta=1e-10)

    def test_lpew2_presents_equal_weights_for_equal_quads(self):
        all_nodes = self.mesh_1.all_nodes
        intern_nodes = np.asarray([all_nodes[12],
                                 all_nodes[13],
                                 all_nodes[14],
                                 all_nodes[15]], dtype='uint64')
        # intern_nodes = all_nodes[[12, 13, 14, 15]]
        for a_node in intern_nodes:
            node_weights = self.imd.by_lpew2(a_node)
            for volume, weight in node_weights.items():
                self.assertAlmostEqual(weight, 0.25, delta=1e-10)

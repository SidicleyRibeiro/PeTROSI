import unittest
import numpy as np
from pressure_solver_2D import MpfaD2D
from mesh_preprocessor import MeshManager
from pressure_solver_2D import InterpolMethod
# import nonconform_mesh_generator_test as nmg

class PressureSolverTest(unittest.TestCase):

    def setUp(self):

        K_1 = np.array([1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0])

        K_2 = np.array([2.0, 0.0, 0.0,
                        0.0, 2.0, 0.0,
                        0.0, 0.0, 2.0])

        self.mesh_1 = MeshManager('mesh_test_1.msh', dim=2)
        self.mesh_1.set_media_property('Permeability', {1: K_1}, dim_target=2)
        self.mesh_1.set_boundary_condition('Dirichlet', {102: 1.0, 101: 0.0},
                                           dim_target=1, set_nodes=True)
        self.mesh_1.set_boundary_condition('Neumann', {201: 0.0},
                                           dim_target=1, set_nodes=True)
        self.mpfad_1 = MpfaD2D(self.mesh_1)
        self.imd_1 = InterpolMethod(self.mesh_1)

        self.mesh_2 = MeshManager('mesh_test_2.msh', dim=2)
        self.mesh_2.set_media_property('Permeability', {1: K_1}, dim_target=2)
        self.mesh_2.set_boundary_condition('Dirichlet', {102: 1.0, 101: 0.0},
                                           dim_target=1, set_nodes=True)
        self.mesh_2.set_boundary_condition('Neumann', {201: 0.0},
                                           dim_target=1, set_nodes=True)
        self.mpfad_2 = MpfaD2D(self.mesh_2)
        self.imd_2 = InterpolMethod(self.mesh_2)

        self.mesh_3 = MeshManager('mesh_test_struct_layers.msh', dim=2)
        self.mesh_3.set_media_property('Permeability', {1: K_2, 2: K_1},
                                       dim_target=2)
        self.mesh_3.set_boundary_condition('Dirichlet', {102: 1.0, 101: 0.0},
                                           dim_target=1, set_nodes=True)
        self.mesh_3.set_boundary_condition('Neumann', {201: 0.0},
                                           dim_target=1, set_nodes=True)
        self.mpfad_3 = MpfaD2D(self.mesh_3)

        self.mesh_4 = MeshManager('mesh_test_unstruct_layers.msh', dim=2)
        self.mesh_4.set_media_property('Permeability', {1: K_2, 2: K_1},
                                       dim_target=2)
        self.mesh_4.set_boundary_condition('Dirichlet', {102: 1.0, 101: 0.0},
                                           dim_target=1, set_nodes=True)
        self.mesh_4.set_boundary_condition('Neumann', {201: 0.0},
                                           dim_target=1, set_nodes=True)
        self.mpfad_4 = MpfaD2D(self.mesh_4)
        # self.mesh_xconf = MeshManager('mesh_nonconform_test.vtk', dim=2)
        # mb = nmg.crazy_mesh()
        # self.mesh_xconf.load_data(mb)
        # self.mpfad_xconf = MpfaD2D(self.mesh_xconf)
        # self.imd_xconf = InterpolMethod(self.mesh_xconf)

    def test_linear_prob_with_struct_mesh_and_dyn_point_at_two_thirds(self):
        imd = InterpolMethod(self.mesh_1, 2.0/3.0)
        self.mpfad_1.run_solver(imd.by_lpew2)
        for a_volume in self.mesh_1.all_volumes:
            local_pressure = self.mesh_1.mb.tag_get_data(
                             self.mpfad_1.pressure_tag, a_volume)
            coord_x = self.mesh_1.get_centroid(a_volume)[0]
            self.assertAlmostEqual(
                local_pressure[0][0], 1 - coord_x, delta=1e-15)

    def test_linear_prob_with_unstruct_mesh_and_dyn_point_at_two_thirds(self):
        imd = InterpolMethod(self.mesh_2, 2.0/3.0)
        self.mpfad_2.run_solver(imd.by_lpew2)
        for a_volume in self.mesh_2.all_volumes:
            local_pressure = self.mesh_2.mb.tag_get_data(
                             self.mpfad_2.pressure_tag, a_volume)
            coord_x = self.mesh_2.get_centroid(a_volume)[0]
            self.assertAlmostEqual(
                local_pressure[0][0], 1 - coord_x, delta=1e-15)

    def test_linear_prob_with_struct_mesh_and_dyn_point_at_adj_node(self):
        imd = InterpolMethod(self.mesh_1, 1.0)
        self.mpfad_1.run_solver(imd.by_lpew2)
        for a_volume in self.mesh_1.all_volumes:
            local_pressure = self.mesh_1.mb.tag_get_data(
                             self.mpfad_1.pressure_tag, a_volume)
            coord_x = self.mesh_1.get_centroid(a_volume)[0]
            self.assertAlmostEqual(
                local_pressure[0][0], 1 - coord_x, delta=1e-15)

    def test_linear_prob_with_unstruct_mesh_and_dyn_point_at_adj_node(self):
        imd = InterpolMethod(self.mesh_2, 1.0)
        self.mpfad_2.run_solver(imd.by_lpew2)
        for a_volume in self.mesh_2.all_volumes:
            local_pressure = self.mesh_2.mb.tag_get_data(
                             self.mpfad_2.pressure_tag, a_volume)
            coord_x = self.mesh_2.get_centroid(a_volume)[0]
            self.assertAlmostEqual(
                local_pressure[0][0], 1 - coord_x, delta=1e-15)

    def test_linear_prob_with_unstruct_mesh_and_dyn_point_half_way(self):
        self.mpfad_2.run_solver(self.imd_2.by_lpew2)
        for a_volume in self.mesh_2.all_volumes:
            local_pressure = self.mesh_2.mb.tag_get_data(
                             self.mpfad_2.pressure_tag, a_volume)
            coord_x = self.mesh_2.get_centroid(a_volume)[0]
            self.assertAlmostEqual(
                local_pressure[0][0], 1 - coord_x, delta=1e-15)

    def test_lpew2_presents_equal_weights_for_equal_quads(self):
        all_nodes = self.mesh_1.all_nodes
        intern_nodes = np.asarray([all_nodes[12],
                                   all_nodes[13],
                                   all_nodes[14],
                                   all_nodes[15]], dtype='uint64')
        # intern_nodes = all_nodes[[12, 13, 14, 15]]
        for a_node in intern_nodes:
            node_weights = self.imd_1.by_lpew2(a_node)
            for volume, weight in node_weights.items():
                self.assertAlmostEqual(weight, 0.25, delta=1e-12)

    def test_piece_wise_lin_prob_struct_with_dyn_point_at_two_thirds(self):
        imd = InterpolMethod(self.mesh_3, 2/3.0)
        vols_press = self.mpfad_3.run_solver(imd.by_lpew2)
        all_volumes = self.mesh_3.all_volumes
        for a_volume, vol_pres in zip(all_volumes, vols_press):
            coord_x = self.mesh_3.get_centroid(a_volume)[0]
            if coord_x < 0.5:
                self.assertAlmostEqual(
                    vol_pres[0], (-2/3.0)*coord_x + 1, delta=1e-15)
            else:
                self.assertAlmostEqual(
                    vol_pres[0], (4/3.0)*(1 - coord_x), delta=1e-15)

    def test_piece_wise_lin_prob_struct_with_dyn_point_at_adj_point(self):
        imd = InterpolMethod(self.mesh_3, 1.0)
        vols_press = self.mpfad_3.run_solver(imd.by_lpew2)
        all_volumes = self.mesh_3.all_volumes
        for a_volume, vol_pres in zip(all_volumes, vols_press):
            coord_x = self.mesh_3.get_centroid(a_volume)[0]
            if coord_x < 0.5:
                self.assertAlmostEqual(
                    vol_pres[0], (-2/3.0)*coord_x + 1, delta=1e-15)
            else:
                self.assertAlmostEqual(
                    vol_pres[0], (4/3.0)*(1 - coord_x), delta=1e-15)

    def test_piece_wise_lin_prob_unstruct_with_dyn_point_at_two_thirds(self):
        imd = InterpolMethod(self.mesh_4, 2/3.0)
        vols_press = self.mpfad_4.run_solver(imd.by_lpew2)
        all_volumes = self.mesh_4.all_volumes
        for a_volume, vol_pres in zip(all_volumes, vols_press):
            coord_x = self.mesh_4.get_centroid(a_volume)[0]
            if coord_x < 0.5:
                self.assertAlmostEqual(
                    vol_pres[0], (-2/3.0)*coord_x + 1, delta=1e-14)
            else:
                self.assertAlmostEqual(
                    vol_pres[0], (4/3.0)*(1 - coord_x), delta=1e-14)

    def test_piece_wise_lin_prob_unstruct_with_dyn_point_at_adj_point(self):
        imd = InterpolMethod(self.mesh_4, 1.0)
        vols_press = self.mpfad_4.run_solver(imd.by_lpew2)
        all_volumes = self.mesh_4.all_volumes
        for a_volume, vol_pres in zip(all_volumes, vols_press):
            coord_x = self.mesh_4.get_centroid(a_volume)[0]
            if coord_x < 0.5:
                self.assertAlmostEqual(
                    vol_pres[0], (-2/3.0)*coord_x + 1, delta=1e-15)
            else:
                self.assertAlmostEqual(
                    vol_pres[0], (4/3.0)*(1 - coord_x), delta=1e-15)

    def test_flux_conservative_dyn_point_at_adj_point(self):
        imd = InterpolMethod(self.mesh_2, 1.0)
        vols_press = self.mpfad_2.run_solver(imd.by_lpew2)
        A = self.mpfad_2.A
        B = self.mpfad_2.B
        for i in range(len(B)):
            residue = np.dot(A[i], vols_press) - B[i]
            self.assertAlmostEqual(residue[0], 0.0, delta=1e-15)

    def test_flux_conservative_dyn_point_at_two_thirds(self):
        imd = InterpolMethod(self.mesh_2, 2.0/3.0)
        vols_press = self.mpfad_2.run_solver(imd.by_lpew2)
        A = self.mpfad_2.A
        B = self.mpfad_2.B
        for i in range(len(B)):
            residue = np.dot(A[i], vols_press) - B[i]
            self.assertAlmostEqual(residue[0], 0.0, delta=1e-15)

    # def test_linear_problem_vtk_non_conform_mesh(self):
    #     self.mpfad_xconf.run_solver(self.imd_xconf.by_lpew2)
    #     for a_volume in self.mesh_xconf.all_volumes:
    #         local_pressure = self.mesh_xconf.mb.tag_get_data(
    #                          self.mpfad_xconf.pressure_tag, a_volume)
    #         coord_x = self.mesh_xconf.get_centroid(a_volume)[0]
    #         self.assertAlmostEqual(
    #             local_pressure[0][0], 1 - coord_x, delta=1e-15)

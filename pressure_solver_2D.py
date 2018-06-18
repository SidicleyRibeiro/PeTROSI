# coding: utf-8
from math import sqrt
from pymoab import types
from interpolation_methods import InterpolMethods
import numpy as np


class MpfaD2D(InterpolMethods):

    def __init__(self, mesh_data):
        self.mesh_data = mesh_data
        self.mb = mesh_data.mb
        self.mtu = mesh_data.mtu

        self.dirichlet_tag = mesh_data.dirichlet_tag
        self.neumann_tag = mesh_data.neumann_tag
        self.perm_tag = mesh_data.perm_tag

        self.pressure_tag = self.mb.tag_get_handle(
            "Pressure", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

        self.dirichlet_nodes = set(self.mb.get_entities_by_type_and_tag(
            0, types.MBVERTEX, self.dirichlet_tag, np.array((None,))))

        self.neumann_nodes = set(self.mb.get_entities_by_type_and_tag(
            0, types.MBVERTEX, self.neumann_tag, np.array((None,))))
        self.neumann_nodes = self.neumann_nodes - self.dirichlet_nodes

        self.all_nodes = mesh_data.all_nodes
        boundary_nodes = (self.dirichlet_nodes | self.neumann_nodes)
        self.intern_nodes = set(self.all_nodes) - boundary_nodes

        self.dirichlet_faces = mesh_data.dirichlet_faces
        self.neumann_faces = mesh_data.neumann_faces

        self.all_faces = mesh_data.all_faces
        boundary_faces = (self.dirichlet_faces | self.neumann_faces)
        # print('ALL FACES', self.all_faces, len(self.all_faces))
        self.intern_faces = set(self.all_faces) - boundary_faces

        self.all_volumes = mesh_data.all_volumes

        self.A = np.zeros([len(self.all_volumes), len(self.all_volumes)])
        self.B = np.zeros([len(self.all_volumes), 1])

    def neumann_weight(self, neumann_node):
        adjacent_faces = self.mtu.get_bridge_adjacencies(neumann_node, 0, 1)
        # print(len(face_adj))
        crds_neum_node = self.mb.get_coords([neumann_node])
        neumann_term = 0
        for face in adjacent_faces:
            if face not in set(self.neumann_nodes):
                continue
            neu_flow_rate = self.mb.tag_get_data(self.neumann_tag, face)
            face_nodes = self.mtu.get_bridge_adjacencies(face, 0, 0)
            other_node = np.extract(face_nodes != np.array([neumann_node]),
                                    face_nodes)
            # print(pts_adj_face[0], pts_adj_face[1], neumann_node, other_node)
            other_node = np.asarray(other_node, dtype='uint64')
            coords_other_node = self.mb.get_coords(other_node)
            # print(other_node, coords_other_node, len(face_adj))
            half_face = self.mtu.get_average_position([face])

            csi_neu = self._get_face_weight(neumann_node, face)
            # print(csi_neu, half_face, crds_neum_node)
            # print (csi_neu, crds_neum_node,
            #        coords_other_node, half_face, len(face_adj))
            module_half_face = self.get_vect_module(half_face - crds_neum_node)
            neumann_term += (1 + csi_neu) * module_half_face * neu_flow_rate
            # print("Teste neumann: ", half_face, neu_flow_rate)

        adjacent_blocks = self.mtu.get_bridge_adjacencies(neumann_node, 0, 2)
        block_weight_sum = 0
        for a_block in adjacent_blocks:
            block_weight = self._get_volume_weight(neumann_node, a_block, 0.5)
            block_weight_sum += block_weight
        neumann_term = neumann_term / block_weight_sum
        # print("Neumann term: ", neumann_term)
        return neumann_term

    def set_global_id(self):
        vol_ids = {}
        range_of_ids = range(len(self.all_volumes))
        for id_, volume in zip(range_of_ids, self.all_volumes):
            vol_ids[volume] = id_
        return vol_ids

    def _get_nodes_weights(self, method):
        nodes_weights = {}
        for a_node in self.intern_nodes | self.neumann_nodes:
            nodes_weights[a_node] = method(a_node)

        for node, weights in nodes_weights.items():
            coord = self.mb.get_coords([node])
            # print("NODE WEIGHTS:", coord, weights)

        return nodes_weights

    def _node_treatment(self, node, nodes_weights, id_1st, id_2nd, v_ids,
                        transm, face_area, cross_term, is_2nd=1.0):
        if node in self.dirichlet_nodes:
            node_press = self.mb.tag_get_data(self.dirichlet_tag, node)
            value = transm * face_area * cross_term * node_press[0][0]
            self.B[id_1st][0] += value * is_2nd
            self.B[id_2nd][0] += - value * is_2nd

        if node in self.intern_nodes | self.neumann_nodes:
            for vol, weight in nodes_weights[node].items():
                value = transm * face_area * cross_term * weight
                self.A[id_1st][v_ids[vol]] += - value * is_2nd
                self.A[id_2nd][v_ids[vol]] += value * is_2nd

        if node in self.neumann_nodes:
            neumann_factor = self.neumann_weight(node)
            # print("NEU NODE FACTOR:", neumann_factor)
            value = transm * face_area * cross_term * (- neumann_factor)
            self.B[id_1st][0] += value * is_2nd
            self.B[id_2nd][0] += - value * is_2nd

    def run_solver(self, method):
        nodes_weights = self._get_nodes_weights(method)

        # if len(self.mesh_data.all_pressure_well_vols) > 0:
        #     for well_volume in self.mesh_data.all_pressure_well_vols:
        #         well_pressure = self.mb.tag_get_data(self.mesh_data.pressure_well_tag, well_volume)
        #         self.mb.tag_set_data(self.mesh_data.pressure_tag, well_volume, well_pressure)
        #         print("WELL VOLUME", self.mesh_data.all_pressure_well_vols, well_volume)
        #         self.all_volumes = self.all_volumes - set([well_volume])
        #
        #         well_volume_faces = self.mb.get_adjacencies(well_volume, 1, True)
        #
        #         well_faces_in_boundary = set(well_volume_faces) & (self.dirichlet_faces | self.neumann_faces)
        #         well_dirichlet_faces = set(well_volume_faces) - well_faces_in_boundary
        #         self.dirichlet_faces = self.dirichlet_faces | well_dirichlet_faces
        #
        #         self.dirichlet_faces = self.dirichlet_faces - well_faces_in_boundary
        #         self.neumann_faces = self.neumann_faces - well_faces_in_boundary
        #         self.intern_faces = self.intern_faces - well_dirichlet_faces
        #
        #         for new_dirich_face in well_dirichlet_faces:
        #             new_dirich_nodes = self.mtu.get_bridge_adjacencies(new_dirich_face, 0, 0)
        #             self.dirichlet_nodes = self.dirichlet_nodes | set(new_dirich_nodes)
        #             self.mb.tag_set_data(self.dirichlet_tag, new_dirich_nodes, np.repeat([well_pressure], 2))
        #             self.mb.tag_set_data(self.dirichlet_tag, new_dirich_face, well_pressure)
        #
        #     self.neumann_nodes = self.neumann_nodes - self.dirichlet_nodes
        #     self.intern_nodes = self.intern_nodes - self.dirichlet_nodes

        v_ids = self.set_global_id()

        for well_volume in self.mesh_data.all_flow_rate_well_vols:
            # print("ALL WELLS: ", len(self.mesh_data.all_well_volumes))
            print("WELL POS: ", self.mesh_data.get_centroid(well_volume), len(self.mesh_data.all_flow_rate_well_vols))
            well_src_term = self.mb.tag_get_data(self.mesh_data.flow_rate_well_tag, well_volume)
            self.B[v_ids[well_volume]][0] += well_src_term

        # print("FACE COUNT: ", len(self.dirichlet_faces), len(self.neumann_faces), len(self.intern_faces))
        for face in self.all_faces:
            adjacent_entities = np.asarray(self.mb.get_adjacencies(face, 2), dtype='uint64')
            if face in self.neumann_faces:
                neumann_flux = self.mb.tag_get_data(self.neumann_tag, face)
                self.B[v_ids[adjacent_entities[0]]][0] += -neumann_flux

            if face in self.dirichlet_faces:

                if len(adjacent_entities) == 2:
                    for a_entity in adjacent_entities:
                        print("ADJ_CENTROIDS: ", self.mesh_data.get_centroid(a_entity), a_entity)
                        print("IDS: ", list(v_ids.keys()))
                        if a_entity in set(list(v_ids.keys())):
                            adjacent_entities = [a_entity]
                            print("ENCONTROU!", adjacent_entities, self.mesh_data.get_centroid(adjacent_entities))
                            break

                centroid_adjacent_entity = self.mesh_data.get_centroid(adjacent_entities)
                face_nodes = np.asarray(self.mb.get_adjacencies(face, 0), dtype='uint64')
                coord_face_nodes = np.reshape(self.mb.get_coords(face_nodes), (2, 3))
                count_wise_face = self.cross_area_vector(coord_face_nodes[0], coord_face_nodes[1], centroid_adjacent_entity)
                if np.dot(count_wise_face, coord_face_nodes[1] - coord_face_nodes[0]) < 0:
                    coord_face_nodes[[0,1]] = coord_face_nodes[[1,0]]
                    face_nodes[[0,1]] = face_nodes[[1,0]]

                pressure_first_node = self.mb.tag_get_data(self.dirichlet_tag, face_nodes[0])
                pressure_second_node = self.mb.tag_get_data(self.dirichlet_tag, face_nodes[1]);
                perm_adjacent_entity = self.mb.tag_get_data(self.perm_tag, adjacent_entities).reshape([3, 3])
                # print("PERMEAB: ", perm_adjacent_entity)
                face_normal = self.area_vector(
                    coord_face_nodes[0], coord_face_nodes[1], centroid_adjacent_entity)

                K_n_B = self._get_conormal_prod(coord_face_nodes, centroid_adjacent_entity, perm_adjacent_entity)
                #print("K_n_B: ", K_n_B, "coords: ", coord_face_nodes, "nodes: ", face_nodes)
                K_t_B = self.K_t_X(coord_face_nodes, centroid_adjacent_entity, perm_adjacent_entity)
                h_B = self.get_face_dist(coord_face_nodes, centroid_adjacent_entity, perm_adjacent_entity)
                module_face_normal = sqrt(np.dot(face_normal, face_normal))

                aux_dot_product_first = np.dot(
                    centroid_adjacent_entity - coord_face_nodes[1], coord_face_nodes[0] - coord_face_nodes[1])
                # print("B2Ob: ", centroid_adjacent_entity - coord_face_nodes[1])
                # print("B2B1: ", coord_face_nodes[0] - coord_face_nodes[1])
                self.B[v_ids[adjacent_entities[0]]][0] += (
                    (K_n_B * aux_dot_product_first)/(h_B * module_face_normal) - K_t_B) * pressure_first_node
                # print("aux_prod: ", aux_dot_product_first)
                # print("B1: ", ((K_n_B * aux_dot_product_first)/(h_B * module_face_normal) - K_t_B) * pressure_first_node)
                aux_dot_product_second = np.dot(
                    centroid_adjacent_entity - coord_face_nodes[0], coord_face_nodes[1] - coord_face_nodes[0])
                self.B[v_ids[adjacent_entities[0]]][0] += (
                    (K_n_B * aux_dot_product_second)/(h_B * module_face_normal) + K_t_B) * pressure_second_node
                # print("B2: ", ((K_n_B * aux_dot_product_second)/(h_B * module_face_normal) + K_t_B) * pressure_second_node)
                self.A[v_ids[adjacent_entities[0]]][v_ids[adjacent_entities[0]]] += K_n_B * module_face_normal/h_B
                #print("Dirich. coefic.: ", self.A, K_n_B * module_face_normal/h_B)

            if face in self.intern_faces:

                first_volume = adjacent_entities[0]
                second_volume = adjacent_entities[1]

                cent_first_volume = self.mesh_data.get_centroid([first_volume])
                cent_second_volume = self.mesh_data.get_centroid([second_volume])

                perm_first_volume = self.mb.tag_get_data(self.perm_tag, first_volume).reshape([3, 3])
                perm_second_volume = self.mb.tag_get_data(self.perm_tag, second_volume).reshape([3, 3])

                face_nodes = np.asarray(self.mb.get_adjacencies(face, 0), dtype='uint64')
                coord_face_nodes = np.reshape(self.mb.get_coords(face_nodes), (2, 3))

                count_wise_face = self.cross_area_vector(coord_face_nodes[0], coord_face_nodes[1], cent_first_volume)
                if np.dot(count_wise_face, coord_face_nodes[1] - coord_face_nodes[0]) < 0:
                    coord_face_nodes[[0,1]] = coord_face_nodes[[1,0]]
                    face_nodes[[0,1]] = face_nodes[[1,0]]

                K_n_first = self._get_conormal_prod(coord_face_nodes, cent_first_volume, perm_first_volume)
                K_n_second = self._get_conormal_prod(coord_face_nodes, cent_second_volume, perm_second_volume)

                K_t_first = self.K_t_X(coord_face_nodes, cent_first_volume, perm_first_volume)
                K_t_second = self.K_t_X(coord_face_nodes, cent_second_volume, perm_second_volume)

                h_first = self.get_face_dist(coord_face_nodes, cent_first_volume, perm_first_volume)
                h_second = self.get_face_dist(coord_face_nodes, cent_second_volume, perm_second_volume)

                K_transm = (K_n_first * K_n_second) / (K_n_first * h_second + K_n_second * h_first)
                #print('K_transm: ', K_transm)
                face_normal = self.area_vector(
                    coord_face_nodes[0], coord_face_nodes[1], cent_first_volume)
                face_area = sqrt(np.dot(face_normal, face_normal))
                aux_dot_product = np.dot(
                    coord_face_nodes[1] - coord_face_nodes[0], cent_second_volume - cent_first_volume)
                aux_K_term = (K_t_second * h_second / K_n_second) + (K_t_first * h_first / K_n_first)

                D_ab = (
                    aux_dot_product / face_area**2) - (1/face_area) * aux_K_term

                vol_id_1st = v_ids[first_volume]
                vol_id_2nd = v_ids[second_volume]

                self.A[vol_id_1st][vol_id_1st] += K_transm * face_area

                self.A[vol_id_1st][vol_id_2nd] += - K_transm * face_area

                self.A[vol_id_2nd][vol_id_2nd] +=  K_transm * face_area

                self.A[vol_id_2nd][vol_id_1st] +=  - K_transm * face_area


                self._node_treatment(face_nodes[0], nodes_weights, vol_id_1st,
                                     vol_id_2nd, v_ids, K_transm,
                                     face_area, D_ab)

                self._node_treatment(face_nodes[1], nodes_weights, vol_id_1st,
                                     vol_id_2nd, v_ids, K_transm,
                                     face_area, D_ab, is_2nd=-1)

                if len(self.mesh_data.all_pressure_well_vols) > 0:
                    for well_volume in self.mesh_data.all_pressure_well_vols:
                        well_pressure = self.mb.tag_get_data(self.mesh_data.pressure_well_tag, well_volume)
                        self.A[v_ids[well_volume], :] = 0
                        self.A[v_ids[well_volume]][v_ids[well_volume]] = 1.0
                        self.B[v_ids[well_volume]][0] = well_pressure
            # print("Calculou face ", count, " de ", len(self.all_faces))

        # print(self.A)
        # print(self.B)
        volume_pressures = np.linalg.solve(self.A, self.B)
        # print(volume_pressures)
        self.mb.tag_set_data(self.pressure_tag, self.all_volumes, volume_pressures.flatten())
        self.mb.write_file("pressure_field.vtk")
        return volume_pressures

    # def get_nodes_pressures(self, mesh_data):
    #
    #     nodes_pressures = {}
    #     for node in self.all_nodes:
    #
    #         if node in mesh_data.dirich_nodes:
    #             nodes_pressures[node] = self.mb.tag_get_data(self.dirichlet_tag, node)
    #             self.mb.tag_set_data(mesh_data.node_pressure_tag, node, nodes_pressures[node])
    #             # print("Dirichlet nodes: ", self.mb.get_coords([node]))
    #
    #         if node in mesh_data.neu_nodes - mesh_data.dirich_nodes:
    #             neumann_term = self.neumann_weight(node)
    #             volume_weight = self.explicit_weights(node)
    #             pressure_node = 0
    #             for vol,  weight in volume_weight:
    #                 vol_pressure = self.mb.tag_get_data(mesh_data.pressure_tag, vol)
    #                 pressure_node += vol_pressure * weight
    #             nodes_pressures[node] = pressure_node - neumann_term
    #             self.mb.tag_set_data(mesh_data.node_pressure_tag, node, nodes_pressures[node])
    #             # print("Neumann nodes: ", self.mb.get_coords([node]))
    #
    #         if node in set(self.all_nodes) - mesh_data.neu_nodes - mesh_data.dirich_nodes:
    #             volume_weight = self.explicit_weights(node)
    #             pressure_node = 0
    #             for vol,  weight in volume_weight:
    #                 vol_pressure = self.mb.tag_get_data(mesh_data.pressure_tag, vol)
    #                 pressure_node += vol_pressure * weight
    #             nodes_pressures[node] = pressure_node
    #             self.mb.tag_set_data(mesh_data.node_pressure_tag, node, nodes_pressures[node])
    #             # print("Intern nodes: ", self.mb.get_coords([node]))
    #     self.mb.write_file("node_pressure_field.vtk")
    #     return nodes_pressures
    #
    #
    # def pressure_grad(self, mesh_data):
    #
    #     self.all_faces = self.mb.get_entities_by_dimension(mesh_data.root_set, 1)
    #
    #     face_grad = {}
    #     for face in self.all_faces:
    #
    #         node_I, node_J = self.mtu.get_bridge_adjacencies(face, 0, 0)
    #         adjacent_volumes = self.mb.get_adjacencies(face, 2)
    #
    #         coords_I = self.mb.get_coords([node_I])
    #         coords_J = self.mb.get_coords([node_J])
    #
    #         pressure_I = self.mb.tag_get_data(mesh_data.node_pressure_tag, node_I)
    #         pressure_J = self.mb.tag_get_data(mesh_data.node_pressure_tag, node_J)
    #
    #         face_grad[face] = {}
    #         for a_volume in adjacent_volumes:
    #
    #             volume_centroid = self.mesh_data.get_centroid(a_volume)
    #             centroid_pressure = self.mb.tag_get_data(mesh_data.pressure_tag, a_volume)
    #
    #             normal_IJ = self.area_vector(coords_I, coords_J, volume_centroid)
    #             normal_JC = self.area_vector(coords_J, volume_centroid, coords_I)
    #             normal_CI = self.area_vector(volume_centroid, coords_I, coords_J)
    #
    #             area_iter = self.mesh_data.get_centroid(coords_I, coords_J, volume_centroid)
    #
    #             grad_p = (-1/(2 * area_iter)) * (
    #                     pressure_I * normal_JC +
    #                     pressure_J * normal_CI +
    #                     centroid_pressure * normal_IJ)
    #
    #             face_grad[face][a_volume] = grad_p
    #
    #     return face_grad

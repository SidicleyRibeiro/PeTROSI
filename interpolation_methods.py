# coding: utf-8
from math import sqrt
from geometric_methods import GeoMethods
import numpy as np


class InterpolMethods(GeoMethods):

    def __init__(self, mesh_data, dist_factor=0.5):
        self.mesh_data = mesh_data
        self.dist_factor = dist_factor

    def _flux_term(self, vector_1st, permeab, vector_2nd, face_area):
        aux_1 = np.dot(vector_1st, permeab)
        aux_2 = np.dot(aux_1, vector_2nd)
        flux_term = aux_2 / face_area
        return flux_term

    def _get_face_weight(self, interp_node, face):
        crds_node = self.mesh_data.mb.get_coords([interp_node])
        adjacent_volumes = self.mesh_data.mb.get_adjacencies(face, 2)
        half_face = self.mesh_data.mtu.get_average_position([face])
        csi_num = 0
        csi_den = 0
        for adjacent_volume in adjacent_volumes:
            centroid_volume = self.mesh_data.get_centroid(adjacent_volume)
            perm_volume = self.mesh_data.mb.tag_get_data(
                self.mesh_data.perm_tag, adjacent_volume).reshape([3, 3])

            interp_node_adj_faces = set(
                self.mesh_data.mtu.get_bridge_adjacencies(interp_node, 0, 1))
            adj_vol_adj_faces = set(self.mesh_data.mtu.get_bridge_adjacencies(
                                    adjacent_volume, 1, 1))

            faces = np.asarray(list(
                interp_node_adj_faces & adj_vol_adj_faces), dtype='uint64')
            other_face = faces[faces != face]
            half_other_face = self.mesh_data.mtu.get_average_position(other_face)

            K_bar_n = self._get_conormal_prod(
                np.asarray([half_other_face, half_face]),
                crds_node, perm_volume)

            aux_dot_num = np.dot(
                crds_node - half_other_face, half_face - half_other_face)
            cot_num = aux_dot_num / (2.0 * self.trian_area(
                half_face, half_other_face, crds_node))
            # print("cot_num: ", cot_num, crds_node, half_face, self.mesh_data.get_centroid(adjacent_volume))

            norm_face = self.area_vector(half_face,
                                         half_other_face, crds_node)
            tan_face = half_face - half_other_face
            sqrd_tan = np.dot(tan_face, tan_face)
            K_bar_t = np.dot(np.dot(norm_face, perm_volume), tan_face)/sqrd_tan

            # print("COMP NUM:", K_bar_n, K_bar_t, centroid_volume, half_face, crds_node)
            csi_num += K_bar_n * cot_num + K_bar_t
            # print("CSI NUM:", csi_num)
            K_den_n = self._get_conormal_prod(
                np.asarray([crds_node, half_face]),
                centroid_volume, perm_volume)

            aux_dot_den = np.dot(half_face - crds_node,
                                 centroid_volume - crds_node)
            cot_den = aux_dot_den / (
                2.0 * self.trian_area(half_face,
                                      centroid_volume,
                                      crds_node))

            norm_vect = self.area_vector(half_face, crds_node, centroid_volume)
            tan_vect = half_face - crds_node
            sqrd_vec = np.dot(tan_vect, tan_vect)
            K_den_t = np.dot(np.dot(norm_vect, perm_volume), tan_vect)/sqrd_vec

            # K_den_t = self.K_t_X(np.asarray([crds_node, half_face]),
            #                                       centroid_volume,
            #                                       perm_volume)

            # print("COMP DEN:", K_den_n, K_den_t, centroid_volume, half_face, crds_node)
            csi_den += K_den_n * cot_den + K_den_t
            # print("CSI DEN:", csi_den)

        csi = csi_num / csi_den
        # print("csi : ", csi, crds_node, half_face)
        return csi

    def _get_neta(self, face, intern_node, cent_node, cent_adj_vol, vol_perm):
        half_face_vect = cent_node - intern_node
        half_face = sqrt(np.dot(half_face_vect, half_face_vect))
        face_nodes = self.mesh_data.mb.get_adjacencies(face, 0)
        face_nodes_crds = self.mesh_data.mb.get_coords(face_nodes).reshape([2, 3])
        height = self.get_face_dist(face_nodes_crds,
                                          cent_adj_vol,
                                          vol_perm)
        neta = half_face / height
        return neta

    def _get_dynamic_point(self, crds_node, node, face, dist_factor):
        face_nodes = self.mesh_data.mb.get_adjacencies(face, 0)
        face_nodes = np.asarray(face_nodes, dtype='uint64')
        other_node = face_nodes[face_nodes != node]
        other_crds = self.mesh_data.mb.get_coords(other_node)
        tan_vector = other_crds - crds_node
        dynamic_point = crds_node + dist_factor * tan_vector
        return dynamic_point

    def _get_volume_weight(self, node, adjacent_volume, dist_factor):
        crds_node = self.mesh_data.mb.get_coords([node])
        perm_adjacent_volume = self.mesh_data.mb.tag_get_data(
            self.mesh_data.perm_tag, adjacent_volume).reshape([3, 3])
        cent_adj_vol = self.mesh_data.get_centroid(adjacent_volume)

        adjacent_faces = list(set(self.mesh_data.mtu.get_bridge_adjacencies(node, 0, 1)) &
                              set(self.mesh_data.mtu.get_bridge_adjacencies(adjacent_volume, 1, 1)))
        first_face = adjacent_faces[0]
        second_face = adjacent_faces[1]
        dyn_point_1st = self._get_dynamic_point(crds_node,
                                                node,
                                                first_face,
                                                dist_factor)
        dyn_point_2nd = self._get_dynamic_point(crds_node,
                                                node,
                                                second_face,
                                                dist_factor)
        K_ni_first = self._get_conormal_prod(np.asarray([crds_node,
                                                   dyn_point_1st]),
                                                   cent_adj_vol,
                                                   perm_adjacent_volume)
        K_ni_second = self._get_conormal_prod(np.asarray([crds_node,
                                                    dyn_point_2nd]),
                                                    cent_adj_vol,
                                                    perm_adjacent_volume)

        neta_1st = self._get_neta(first_face, crds_node, dyn_point_1st,
                                  cent_adj_vol, perm_adjacent_volume)
        neta_2nd = self._get_neta(second_face, crds_node, dyn_point_2nd,
                                  cent_adj_vol, perm_adjacent_volume)

        csi_first = self._get_face_weight(node, first_face)
        csi_second = self._get_face_weight(node, second_face)

        node_weight = K_ni_first * neta_1st * csi_first + \
                      K_ni_second * neta_2nd * csi_second
        # node_weight = abs(node_weight)
        # print("VALUES: ", K_ni_first, neta_1st, csi_first)
        #
        # print("weight: ", node_weight, crds_node, cent_adj_vol)
        return node_weight

    def by_lpew2(self, node):
        # node = np.asarray([node], dtype='uint64')
        # print("NODE", node, self.mesh_data.mb.get_coords([node]))
        vols_around = self.mesh_data.mb.get_adjacencies([node], 2)
        weight_sum = 0.0
        weights = np.array([])
        for a_volume in vols_around:
            weight = self._get_volume_weight(node, a_volume, self.dist_factor)
            weights = np.append(weights, weight)
            weight_sum += weight
        weights = weights / weight_sum
        node_weights = {
            vol: weight for vol, weight in zip(vols_around, weights)}
        return node_weights

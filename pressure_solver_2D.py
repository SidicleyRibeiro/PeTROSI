# coding: utf-8
from math import sqrt
from pymoab import types
import numpy as np


class MpfaD2D:

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

        boundary_nodes = (self.dirichlet_nodes | self.neumann_nodes)
        self.intern_nodes = set(mesh_data.all_nodes) - boundary_nodes

        self.dirichlet_faces = mesh_data.dirichlet_faces
        self.neumann_faces = mesh_data.neumann_faces

        self.all_faces = mesh_data.all_faces
        boundary_faces = (self.dirichlet_faces | self.neumann_faces)
        # print('ALL FACES', self.all_faces, len(self.all_faces))
        self.intern_faces = set(self.all_faces) - boundary_faces

        self.all_volumes = self.mesh_data.all_volumes

        self.A = np.zeros([len(self.all_volumes), len(self.all_volumes)])
        self.B = np.zeros([len(self.all_volumes), 1])


    def get_vect_module(self, vector):
        vector = np.array(vector)
        dot_product = np.dot(vector, vector)
        module = sqrt(dot_product)
        return module

    def get_vectors_angle(self, vect_1, vect_2):
        dot_product = np.dot(vect_1, vect_2)
        module_prod = (self.get_vect_module(vect_1) *
                       self.get_vect_module(vect_2))
        try:
            arc = dot_product / module_prod
            if np.fabs(arc) > 1:
                raise ValueError('Arco maior que 1 !!!')
        except ValueError:
            arc = np.around(arc)

        angle = np.arccos(arc)
        return angle

    def area_vector(self, p_1, p_2, ref):
        tan_vector = p_2 - p_1
        ref_vector = p_2 - ref
        area_normal = np.array([-tan_vector[1],
                                tan_vector[0],
                                tan_vector[2]])
        if np.dot(area_normal, ref_vector) <= 0:
            area_normal = -1.0*area_normal

        return area_normal

    def cross_area_vector(self, p_1, p_2, ref):
        area_normal = self.area_vector(p_1, p_2, ref)
        tan_vector = np.array([-area_normal[1],
                       area_normal[0],
                       area_normal[2]])
        return tan_vector

    def trian_area(self, p_1, p_2, vert):
        vect_1 = p_2 - vert
        vect_2 = p_1 - vert
        normal_vector = np.cross(vect_1, vect_2)
        area = sqrt(np.dot(normal_vector, normal_vector)) / 2.0
        # re_sin = np.sin(self.get_vectors_angle(vect_1, vect_2))# + 1e-25
        # area = (0.5)*self.get_vect_module(vect_1)*self.get_vect_module(vect_2)*re_sin
        # if area < 0:
        #     area = -area
        return area

    def mid_point(self, p1, p2):
        coords_p1 = self.mb.get_coords(p1)
        coords_p2 = self.mb.get_coords(p2)
        mid_p = (coords_p1 + coords_p2)/2.0
        return mid_p

    def _get_conormal_prod(self, node_face_coords, vol_centroid, perm):
        face_normal = self.area_vector(
            node_face_coords[0], node_face_coords[1], vol_centroid)
        squared_face = np.dot(face_normal, face_normal)
        #print("face_normal: ", face_normal, "perm:", perm, "module_2: ", squared_face)
        conormal = np.dot(np.dot(face_normal, perm), face_normal)/squared_face
        # print("conormal: ", conormal, conormal == 1.0)
        return conormal

    def K_t_X(self, node_face_coords, vol_centroid, perm):
        count_wise_face = self.cross_area_vector(
            node_face_coords[0], node_face_coords[1], vol_centroid)
        face_normal = self.area_vector(
            node_face_coords[0], node_face_coords[1], vol_centroid)

        squared_face = np.dot(face_normal, face_normal)
        K_t = np.dot(np.dot(face_normal, perm), count_wise_face)/squared_face
        # print("K_t: ", K_t, K_t == 0.0)
        return K_t

    def get_face_dist(self, node_face_coords, vol_centroid, perm):
        face_normal = self.area_vector(node_face_coords[0], node_face_coords[1], vol_centroid)
        face_centroid = (node_face_coords[0] + node_face_coords[1])/2.0
        module_face_normal = sqrt(np.dot(face_normal, face_normal))

        h = np.dot(face_normal, face_centroid - vol_centroid)/module_face_normal
        # print("h: ", h)
        # print(node_face_coords)
        # print(vol_centroid)
        return h


    def face_weight(self, interp_node, face):
        crds_node = self.mb.get_coords([interp_node])
        adjacent_volumes = self.mb.get_adjacencies(face, 2)
        half_face = self.mtu.get_average_position([face])
        csi_num = 0
        csi_den = 0
        for adjacent_volume in adjacent_volumes:
            centroid_volume = self.mesh_data.get_centroid(adjacent_volume)
            perm_volume = self.mb.tag_get_data(self.perm_tag, adjacent_volume).reshape([3, 3])

            interp_node_adj_faces = set(self.mtu.get_bridge_adjacencies(interp_node, 0, 1))
            adjacent_volume_adj_faces = set(self.mtu.get_bridge_adjacencies(adjacent_volume, 1, 1))

            faces = np.asarray(list(interp_node_adj_faces & adjacent_volume_adj_faces), dtype='uint64')
            other_face = faces[faces != face]
            half_other_face = self.mtu.get_average_position(other_face)

            K_bar_n = self._get_conormal_prod(np.asarray([half_other_face, half_face]), centroid_volume, perm_volume)

            aux_dot_num = np.dot(crds_node - half_other_face, half_face - half_other_face)
            cot_num = aux_dot_num / (2.0 * self.trian_area(half_face, half_other_face, crds_node))
            # print("cot_num: ", cot_num, crds_node, half_face, self.mesh_data.get_centroid(adjacent_volume))
            vector_pseudo_face = half_face - half_other_face
            normal_pseudo_face = self.area_vector(half_face, half_other_face, crds_node)
            module_squared_vector = np.dot(vector_pseudo_face, vector_pseudo_face)
            K_bar_t = np.dot(np.dot(normal_pseudo_face, perm_volume), vector_pseudo_face)/module_squared_vector

            csi_num += K_bar_n * cot_num + K_bar_t
            # print("")
            K_den_n = self._get_conormal_prod(np.asarray([crds_node, half_face]), centroid_volume, perm_volume)

            aux_dot_den = np.dot(half_face - crds_node, centroid_volume - crds_node)
            cot_den = aux_dot_den / (2.0 * self.trian_area(half_face, centroid_volume, crds_node))

            K_den_t = self.K_t_X(np.asarray([crds_node, half_face]), centroid_volume, perm_volume)

            csi_den += K_den_n * cot_den + K_den_t

        csi = csi_num / csi_den
        # print("csi: ", csi, crds_node, half_face)
        return csi



    def partial_weight(self, node, adjacent_volume):
        crds_node = self.mb.get_coords([node])
        perm_adjacent_volume = self.mb.tag_get_data(self.perm_tag, adjacent_volume).reshape([3, 3])
        cent_adj_vol = self.mesh_data.get_centroid(adjacent_volume)

        adjacent_faces = list(set(self.mtu.get_bridge_adjacencies(node, 0, 1)) &
                              set(self.mtu.get_bridge_adjacencies(adjacent_volume, 1, 1)))
        first_face = adjacent_faces[0]
        second_face = adjacent_faces[1]

        half_first_face = self.mtu.get_average_position([first_face])
        half_second_face = self.mtu.get_average_position([second_face])

        K_ni_first = self._get_conormal_prod(np.asarray([crds_node, half_first_face]),
                           cent_adj_vol, perm_adjacent_volume)
        K_ni_second = self._get_conormal_prod(np.asarray([crds_node, half_second_face]),
                            cent_adj_vol, perm_adjacent_volume)

        nodes_first_face = self.mb.get_adjacencies(first_face, 0)
        nodes_second_face = self.mb.get_adjacencies(second_face, 0)

        coords_nodes_first_face = self.mb.get_coords(nodes_first_face).reshape([2, 3])
        coords_nodes_second_face = self.mb.get_coords(nodes_second_face).reshape([2, 3])

        h_first = self.get_face_dist(coords_nodes_first_face, cent_adj_vol, perm_adjacent_volume)
        h_second = self.get_face_dist(coords_nodes_second_face, cent_adj_vol, perm_adjacent_volume)

        half_vect_first_face = half_first_face - crds_node
        half_vect_second_node = half_second_face - crds_node

        half_modsqrd_first_face = sqrt(np.dot(half_vect_first_face, half_vect_first_face))
        half_modsqrd_second_face = sqrt(np.dot(half_vect_second_node, half_vect_second_node))

        neta_first = half_modsqrd_first_face / h_first
        neta_second = half_modsqrd_second_face / h_second
        # print("netas: ", neta_first, neta_second, h_first, h_second, crds_node, cent_adj_vol)
        csi_first = self.face_weight(node, first_face)
        csi_second = self.face_weight(node, second_face)

        node_weight = K_ni_first * neta_first * csi_first + K_ni_second * neta_second * csi_second
        # print("weight: ", node_weight, crds_node, cent_adj_vol)
        return node_weight


    def neumann_weight(self, neumann_node):
        adjacent_faces = self.mtu.get_bridge_adjacencies(neumann_node, 0, 1)
        #print len(face_adj)
        coords_neumann_node = self.mb.get_coords([neumann_node])
        neumann_term = 0
        for face in adjacent_faces:
            try:
                neu_flow_rate = self.mb.tag_get_data(self.neumann_tag, face)
                face_nodes = self.mtu.get_bridge_adjacencies(face, 0, 0)
                other_node = np.extract(face_nodes != np.array([neumann_node]), face_nodes)
                #print pts_adj_face[0], pts_adj_face[1], neumann_node, other_node
                other_node = np.asarray(other_node, dtype='uint64')
                coords_other_node = self.mb.get_coords(other_node)
                #print other_node, coords_other_node, len(face_adj)
                half_face = self.mtu.get_average_position([face])

                csi_neu = self.face_weight(neumann_node, face)
                #print csi_neu, half_face, coords_neumann_node
                #print csi_neu, coords_neumann_node, coords_other_node, half_face, len(face_adj)
                module_half_face = self.get_vect_module(half_face - coords_neumann_node)
                neumann_term += (1 + csi_neu) * module_half_face * neu_flow_rate
                # print("Teste neumann: ", half_face, neu_flow_rate)
            except RuntimeError:
                continue

        adjacent_blocks = self.mtu.get_bridge_adjacencies(neumann_node, 0, 2)
        block_weight_sum = 0
        for a_block in adjacent_blocks:
            block_weight = self.partial_weight(neumann_node, a_block)
            block_weight_sum += block_weight
        neumann_term = neumann_term / block_weight_sum
        # print("Neumann term: ", neumann_term)
        return neumann_term



    def explicit_weights(self, interp_node):
        adjacent_volumes = self.mtu.get_bridge_adjacencies(interp_node, 0, 2)
        weight_sum = 0
        weights = []
        for adjacent_volume in adjacent_volumes:
            weight = self.partial_weight(interp_node, adjacent_volume)
            weights.append(weight)
            weight_sum += weight
            # print("volumes: ", self.mb.get_coords([interp_node]), get_centroid(adjacent_volume))
        weights = weights / weight_sum
        # print("weights: ", weights, )
        volumes_weights = [[vol, weigh] for vol, weigh in zip(adjacent_volumes, weights)]
        # print("pesos: ", self.mb.get_coords([interp_node]), volumes_weights)
        return volumes_weights

    def set_global_id(self):
        vol_ids = {}
        range_of_ids = range(len(self.mesh_data.all_volumes))
        for id_, volume in zip(range_of_ids, self.mesh_data.all_volumes):
            vol_ids[volume] = id_
        return vol_ids

    def run_solver(self):

        nodes_weights = {}
        ncount = 0
        for intern_node in self.intern_nodes:
            intern_node_weight = self.explicit_weights(intern_node)
            nodes_weights[intern_node] = intern_node_weight
            # print("No ", ncount, " de", len(self.intern_nodes))
            ncount = ncount + 1
            # print("Pesos: ", intern_node, self.mb.get_coords([intern_node]), nodes_weights)
        neumann_nodes_weights = {}

        for neumann_node in self.neumann_nodes:
            node_weight = self.explicit_weights(neumann_node)
            neumann_nodes_weights[neumann_node] = node_weight
            # print("Neumann node:  ", self.mb.get_coords([neumann_node]))

        count = 0
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

        print("FACE COUNT: ", len(self.dirichlet_faces), len(self.neumann_faces), len(self.intern_faces))
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
                module_face_normal = sqrt(np.dot(face_normal, face_normal))
                aux_dot_product = np.dot(
                    coord_face_nodes[1] - coord_face_nodes[0], cent_second_volume - cent_first_volume)
                aux_K_term = (K_t_second * h_second / K_n_second) + (K_t_first * h_first / K_n_first)

                D_ab = (
                    aux_dot_product / module_face_normal**2) - (1/module_face_normal) * aux_K_term
                # print("D_ab:", D_ab, cent_first_volume, cent_second_volume)
                # print("mod: ", module_face_normal, cent_first_volume, cent_second_volume)
                # print("A_antes: ")
                # print(self.A)
                self.A[v_ids[first_volume]][v_ids[first_volume]] += K_transm * module_face_normal
                # print("self.A, mesmo vol:")
                # print(self.A)
                self.A[v_ids[first_volume]][v_ids[second_volume]] += - K_transm * module_face_normal
                # print("V_ids: ", v_ids[first_volume], v_ids[second_volume])
                # print(- K_transm * module_face_normal)
                # print("Linha primo para seg")
                # print(self.A)
                self.A[v_ids[second_volume]][v_ids[second_volume]] +=  K_transm * module_face_normal
                # print("Seg vol")
                # print(self.A)
                self.A[v_ids[second_volume]][v_ids[first_volume]] +=  - K_transm * module_face_normal
                # print("V_ids: ", v_ids[first_volume], v_ids[second_volume])
                # print(- K_transm * module_face_normal)
                # print("Linha seg para primo")
                # print(self.A)

                if face_nodes[0] in self.intern_nodes:
                    # print("No 1 volume 1: ", self.mb.get_coords([face_nodes[0]]), self.mb.get_coords([face_nodes[1]]), cent_first_volume)
                    for vol, weigh in nodes_weights[face_nodes[0]]:
                        self.A[v_ids[first_volume]][v_ids[vol]] += - K_transm * module_face_normal * D_ab * weigh
                        self.A[v_ids[second_volume]][v_ids[vol]] +=  K_transm * module_face_normal * D_ab * weigh
                        # print("Aval: ", K_transm * module_face_normal * D_ab * weigh, D_ab)

                elif face_nodes[0] in self.dirichlet_nodes:
                    pressure_first_node = self.mb.tag_get_data(self.dirichlet_tag, face_nodes[0])
                    self.B[v_ids[first_volume]][0] += K_transm * module_face_normal * D_ab * pressure_first_node[0][0]
                    self.B[v_ids[second_volume]][0] += - K_transm * module_face_normal * D_ab * pressure_first_node[0][0]

                elif face_nodes[0] in self.neumann_nodes:
                    neumann_node_factor = self.neumann_weight(face_nodes[0])
                    self.B[v_ids[first_volume]][0] += K_transm * module_face_normal * D_ab * (-neumann_node_factor)
                    self.B[v_ids[second_volume]][0] += - K_transm * module_face_normal * D_ab * (-neumann_node_factor)
                    for vol, weigh in neumann_nodes_weights[face_nodes[0]]:
                        self.A[v_ids[first_volume]][v_ids[vol]] += - K_transm * module_face_normal * D_ab * weigh
                        self.A[v_ids[second_volume]][v_ids[vol]] +=  K_transm * module_face_normal * D_ab * weigh


                if face_nodes[1] in self.intern_nodes:
                    for vol, weigh in nodes_weights[face_nodes[1]]:
                        self.A[v_ids[first_volume]][v_ids[vol]] += K_transm * module_face_normal * D_ab * weigh
                        self.A[v_ids[second_volume]][v_ids[vol]] += - K_transm * module_face_normal * D_ab * weigh

                elif face_nodes[1] in self.dirichlet_nodes:
                    pressure_second_node = self.mb.tag_get_data(self.dirichlet_tag, face_nodes[1])
                    self.B[v_ids[first_volume]][0] += - K_transm * module_face_normal * D_ab * pressure_second_node[0][0]
                    self.B[v_ids[second_volume]][0] += K_transm * module_face_normal * D_ab * pressure_second_node[0][0]

                elif face_nodes[1] in self.neumann_nodes:
                    neumann_node_factor = self.neumann_weight(face_nodes[1])
                    self.B[v_ids[first_volume]][0] += - K_transm * module_face_normal * D_ab * (-neumann_node_factor)
                    self.B[v_ids[second_volume]][0] += K_transm * module_face_normal * D_ab * (-neumann_node_factor)
                    for vol, weigh in neumann_nodes_weights[face_nodes[1]]:
                        # print("Peso neumann: ", weigh, self.mb.get_coords([face_nodes[1]]), self.mesh_data.get_centroid(vol), self.mesh_data.get_centroid(first_volume), self.mesh_data.get_centroid(second_volume))
                        self.A[v_ids[first_volume]][v_ids[vol]] += K_transm * module_face_normal * D_ab * weigh
                        self.A[v_ids[second_volume]][v_ids[vol]] += - K_transm * module_face_normal * D_ab * weigh

                if len(self.mesh_data.all_pressure_well_vols) > 0:
                    for well_volume in self.mesh_data.all_pressure_well_vols:
                        well_pressure = self.mb.tag_get_data(self.mesh_data.pressure_well_tag, well_volume)
                        self.A[v_ids[well_volume], :] = 0
                        self.A[v_ids[well_volume]][v_ids[well_volume]] = 1.0
                        self.B[v_ids[well_volume]][0] = well_pressure
            # print("Calculou face ", count, " de ", len(self.all_faces))
            count = count + 1

        print(self.A)
        print(self.B)
        volume_pressures = np.linalg.solve(self.A, self.B)
        print(volume_pressures)
        self.mb.tag_set_data(self.pressure_tag, self.all_volumes, volume_pressures.flatten())
        self.mb.write_file("pressure_field.vtk")
        return volume_pressures


    def get_nodes_pressures(self, mesh_data):

        nodes_pressures = {}
        for node in mesh_data.all_nodes:

            if node in mesh_data.dirich_nodes:
                nodes_pressures[node] = self.mb.tag_get_data(self.dirichlet_tag, node)
                self.mb.tag_set_data(mesh_data.node_pressure_tag, node, nodes_pressures[node])
                # print("Dirichlet nodes: ", self.mb.get_coords([node]))

            if node in mesh_data.neu_nodes - mesh_data.dirich_nodes:
                neumann_term = self.neumann_weight(node)
                volume_weight = self.explicit_weights(node)
                pressure_node = 0
                for vol,  weight in volume_weight:
                    vol_pressure = self.mb.tag_get_data(mesh_data.pressure_tag, vol)
                    pressure_node += vol_pressure * weight
                nodes_pressures[node] = pressure_node - neumann_term
                self.mb.tag_set_data(mesh_data.node_pressure_tag, node, nodes_pressures[node])
                # print("Neumann nodes: ", self.mb.get_coords([node]))

            if node in set(mesh_data.all_nodes) - mesh_data.neu_nodes - mesh_data.dirich_nodes:
                volume_weight = self.explicit_weights(node)
                pressure_node = 0
                for vol,  weight in volume_weight:
                    vol_pressure = self.mb.tag_get_data(mesh_data.pressure_tag, vol)
                    pressure_node += vol_pressure * weight
                nodes_pressures[node] = pressure_node
                self.mb.tag_set_data(mesh_data.node_pressure_tag, node, nodes_pressures[node])
                # print("Intern nodes: ", self.mb.get_coords([node]))
        self.mb.write_file("node_pressure_field.vtk")
        return nodes_pressures


    def pressure_grad(self, mesh_data):

        self.all_faces = self.mb.get_entities_by_dimension(mesh_data.root_set, 1)

        face_grad = {}
        for face in self.all_faces:

            node_I, node_J = self.mtu.get_bridge_adjacencies(face, 0, 0)
            adjacent_volumes = self.mb.get_adjacencies(face, 2)

            coords_I = self.mb.get_coords([node_I])
            coords_J = self.mb.get_coords([node_J])

            pressure_I = self.mb.tag_get_data(mesh_data.node_pressure_tag, node_I)
            pressure_J = self.mb.tag_get_data(mesh_data.node_pressure_tag, node_J)

            face_grad[face] = {}
            for a_volume in adjacent_volumes:

                volume_centroid = self.mesh_data.get_centroid(a_volume)
                centroid_pressure = self.mb.tag_get_data(mesh_data.pressure_tag, a_volume)

                normal_IJ = self.area_vector(coords_I, coords_J, volume_centroid)
                normal_JC = self.area_vector(coords_J, volume_centroid, coords_I)
                normal_CI = self.area_vector(volume_centroid, coords_I, coords_J)

                area_iter = self.mesh_data.get_centroid(coords_I, coords_J, volume_centroid)

                grad_p = (-1/(2 * area_iter)) * (
                        pressure_I * normal_JC +
                        pressure_J * normal_CI +
                        centroid_pressure * normal_IJ)

                face_grad[face][a_volume] = grad_p

        return face_grad


class InterpolMethod:

    def __init__(self, mpfad):
        pass

    def by_lpew2(self, node, adjacent_volume):
        vols_around = self.mpfad.mesh_data.get_adjacencies(node, 2)
        crds_node = self.mb.get_coords([node])

        for a_volume in vols_around:
            perm_adjacent_volume = self.mb.tag_get_data(self.perm_tag, adjacent_volume).reshape([3, 3])

            pass





        crds_node = self.mb.get_coords([node])
        perm_adjacent_volume = self.mb.tag_get_data(self.perm_tag, adjacent_volume).reshape([3, 3])
        cent_adj_vol = self.mesh_data.get_centroid(adjacent_volume)

        adjacent_faces = list(set(self.mtu.get_bridge_adjacencies(node, 0, 1)) &
                              set(self.mtu.get_bridge_adjacencies(adjacent_volume, 1, 1)))
        first_face = adjacent_faces[0]
        second_face = adjacent_faces[1]

        half_first_face = self.mtu.get_average_position([first_face])
        half_second_face = self.mtu.get_average_position([second_face])

        K_ni_first = self._get_conormal_prod(np.asarray([crds_node, half_first_face]),
                           cent_adj_vol, perm_adjacent_volume)
        K_ni_second = self._get_conormal_prod(np.asarray([crds_node, half_second_face]),
                            cent_adj_vol, perm_adjacent_volume)

        nodes_first_face = self.mb.get_adjacencies(first_face, 0)
        nodes_second_face = self.mb.get_adjacencies(second_face, 0)

        coords_nodes_first_face = self.mb.get_coords(nodes_first_face).reshape([2, 3])
        coords_nodes_second_face = self.mb.get_coords(nodes_second_face).reshape([2, 3])

        h_first = self.get_face_dist(coords_nodes_first_face, cent_adj_vol, perm_adjacent_volume)
        h_second = self.get_face_dist(coords_nodes_second_face, cent_adj_vol, perm_adjacent_volume)

        half_vect_first_face = half_first_face - crds_node
        half_vect_second_node = half_second_face - crds_node

        half_modsqrd_first_face = sqrt(np.dot(half_vect_first_face, half_vect_first_face))
        half_modsqrd_second_face = sqrt(np.dot(half_vect_second_node, half_vect_second_node))

        neta_first = half_modsqrd_first_face / h_first
        neta_second = half_modsqrd_second_face / h_second
        # print("netas: ", neta_first, neta_second, h_first, h_second, crds_node, cent_adj_vol)
        csi_first = self.face_weight(node, first_face)
        csi_second = self.face_weight(node, second_face)

        node_weight = K_ni_first * neta_first * csi_first + K_ni_second * neta_second * csi_second
        # print("weight: ", node_weight, crds_node, cent_adj_vol)
        return node_weight

# coding: utf-8
from math import sqrt
import numpy as np


class GeoMethods:

    def _get_conormal_prod(self, node_face_coords, vol_centroid, perm):
        face_normal = self.area_vector(
            node_face_coords[0], node_face_coords[1], vol_centroid)
        squared_face = np.dot(face_normal, face_normal)
        # print("face_normal: ", face_normal, "perm:",
        #         perm, "module_2: ", squared_face)
        conormal = np.dot(np.dot(face_normal, perm), face_normal)/squared_face
        # print("conormal: ", conormal, conormal == 1.0)
        return conormal

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
        return area + 1e-15

    def mid_point(self, p1, p2):
        coords_p1 = self.mb.get_coords(p1)
        coords_p2 = self.mb.get_coords(p2)
        mid_p = (coords_p1 + coords_p2)/2.0
        return mid_p

    def K_t_X(self, node_face_coords, vol_centroid, perm):
        count_wise_face = self.cross_area_vector(
            node_face_coords[0], node_face_coords[1], vol_centroid)
        face_normal = self.area_vector(
            node_face_coords[0], node_face_coords[1], vol_centroid)

        squared_face = np.dot(face_normal, face_normal)
        K_t = np.dot(np.dot(face_normal, perm), count_wise_face)/squared_face
        # print("K_t: ", K_t, K_t == 0.0)
        return K_t

    def get_face_dist(self, face_nodes, vol_cent, perm):
        face_normal = self.area_vector(face_nodes[0], face_nodes[1], vol_cent)
        face_centroid = (face_nodes[0] + face_nodes[1])/2.0
        area_vector = sqrt(np.dot(face_normal, face_normal))
        h = np.dot(face_normal, face_centroid - vol_cent)/area_vector
        return h

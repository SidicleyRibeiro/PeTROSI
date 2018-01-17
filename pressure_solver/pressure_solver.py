# coding: utf-8

# In[114]:

from pymoab import core
from pymoab import topo_util
from pymoab import types
from pymoab.rng import Range
from math import pi
from math import sqrt
from math import log10
from math import trunc
from math import floor
from math import ceil
import numpy as np
import sys
import random

mb = core.Core()
mtu = topo_util.MeshTopoUtil(mb)
mb.load_file('geometry_test.msh')
root_set = mb.get_root_set()

# Tag que vai aparecer no VTK
pressure_tag = mb.tag_get_handle(
    "pressure", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)  #Tag para a solucao

# Tag pra condição de dirichlet
dirichlet_tag = mb.tag_get_handle(
    "dirichlet", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)  #Tag para a valores da condicao de contorno de dirichlet

neumann_tag = mb.tag_get_handle(
    "neumann", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True) #Tag para os valores de contorno de Neumann

error_tag = mb.tag_get_handle(
    "error", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

node_pressure_tag = mb.tag_get_handle(
    "node_pressure", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

ref_degree_tag = mb.tag_get_handle(
    "ref_degree", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

hanging_nodes_tag = mb.tag_get_handle(
    "hanging_nodes", 1, types.MB_TYPE_HANDLE, types.MB_TAG_SPARSE, True)

full_edges_tag = mb.tag_get_handle(
    "full_edges", 1, types.MB_TYPE_HANDLE, types.MB_TAG_SPARSE, True)

perm_tag = mb.tag_get_handle(
    "PERM", 9, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

weights_tag = mb.tag_get_handle(
    "weights", 2, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

#para problema sem poço

physical_tag = mb.tag_get_handle("MATERIAL_SET")

physical_sets = mb.get_entities_by_type_and_tag(
    0, types.MBENTITYSET, np.array((physical_tag,)), np.array((None,)))

dirichlet_nodes = set()
neumann_nodes = set()
for tag in physical_sets:
    tag_id = mb.tag_get_data(physical_tag, np.array([tag]), flat=True)

    if tag_id == 201:
        entity_set_201 = mb.get_entities_by_handle(tag, True)
        for ent_201 in entity_set_201:
            mb.tag_set_data(neumann_tag, ent_201, [0.0,])
            bound_nodes_201 = mtu.get_bridge_adjacencies(ent_201, 0, 0)
            neumann_nodes = neumann_nodes | set(bound_nodes_201)
            mb.tag_set_data(neumann_tag, bound_nodes_201, np.repeat([0.0], len(bound_nodes_201)))

    if tag_id == 202:
        entity_set_202 = mb.get_entities_by_handle(tag, True)
        for ent_202 in entity_set_202:
            mb.tag_set_data(neumann_tag, ent_202, [1.0,])
            bound_nodes_202 = mtu.get_bridge_adjacencies(ent_202, 0, 0)
            neumann_nodes = neumann_nodes | set(bound_nodes_202)
            mb.tag_set_data(neumann_tag, bound_nodes_202, np.repeat([1.0], len(bound_nodes_202)))

for tag in physical_sets:
    tag_id = mb.tag_get_data(physical_tag, np.array([tag]), flat=True)

    if tag_id == 101:
        entity_set_101 = mb.get_entities_by_handle(tag, True)
        for ent_101 in entity_set_101:
            mb.tag_set_data(dirichlet_tag, ent_101, [0.0,])
            bound_nodes_101 = mtu.get_bridge_adjacencies(ent_101, 0, 0)
            dirichlet_nodes = dirichlet_nodes | set(bound_nodes_101)
            mb.tag_set_data(dirichlet_tag, bound_nodes_101, np.repeat([0.0], len(bound_nodes_101)))
    if tag_id == 102:
        entity_set_102 = mb.get_entities_by_handle(tag, True)
        for ent_102 in entity_set_102:
            mb.tag_set_data(dirichlet_tag, ent_102, [1.0,])
            bound_nodes_102 = mtu.get_bridge_adjacencies(ent_102, 0, 0)
            dirichlet_nodes = dirichlet_nodes | set(bound_nodes_102)
            mb.tag_set_data(dirichlet_tag, bound_nodes_102, np.repeat([1.0], len(bound_nodes_102)))

neumann_nodes = neumann_nodes - dirichlet_nodes
# intern_nodes includes neumann_nodes
intern_nodes = set(mb.get_entities_by_dimension(root_set, 0)) - dirichlet_nodes
intern_nodes = intern_nodes - neumann_nodes

dirichlet_nodes = list(dirichlet_nodes)
neumann_nodes = list(neumann_nodes)
intern_nodes = list(intern_nodes)

#print("Coords: ", mb.get_coords([3]))
#new_node_tag_dirichlet = mb.tag_get_data(dirichlet_tag, [3])
perm_tensor = [1.0, 0.0, 0.0,
               0.0, 1.0, 0.0,
               0.0, 0.0, 1.0]

entities = mb.get_entities_by_dimension(root_set, 2)
for ent in entities:
    full_edges = mb.get_adjacencies(ent, 1, True)
    full_edge_meshset = mb.create_meshset()
    mb.add_entities(full_edge_meshset, full_edges)
    mb.tag_set_data(full_edges_tag, ent, full_edge_meshset)
    mb.tag_set_data(perm_tag, ent, perm_tensor)

#all_verts = mb.get_entities_by_dimension(root_set, 0)
#mtu.construct_aentities(all_verts)

def get_centroid_reg(entity):
    points = mb.get_adjacencies(entity, 0)
    coords = mb.get_coords(points)
    qtd_pts = len(points)
    #print qtd_pts, 'qtd_pts'
    coords = np.reshape(coords, (qtd_pts, 3))
    centroid = sum(coords)/qtd_pts

    return centroid

def norma(vector):
    vector = np.array(vector)
    dot_product = np.dot(vector, vector)
    mag = sqrt(dot_product)
    return mag

def ang_vectors(u, v):
    u = np.array(u)
    v = np.array(v)
    dot_product = np.dot(u,v)
    norms = norma(u)*norma(v)

    try:
        arc = dot_product/norms
        if np.fabs(arc) > 1:
            raise ValueError('Arco maior que 1 !!!')
    except ValueError:
        arc = np.around(arc)

    ang = np.arccos(arc)
    #print ang, arc, dot_product, norms, u, v
    return ang

def norm_vec(u, v, p):
    u = np.array(u)
    v = np.array(v)
    p = np.array(p)
    uv = v - u
    pv = v - p
    Normal_vu = np.array([-uv[1], uv[0], uv[2]])
    #print ang_vectors(Normal_vu, v-p)
    if np.dot(Normal_vu, pv) <= 0:
        #print 'Oposto!', ang_vectors(Normal_vu, pv)
        Normal_vu = -1.0*Normal_vu

    return Normal_vu

def count_wise(u, v, p):
    normal_uv = norm_vec(u, v, p)
    uv = np.array([-normal_uv[1], normal_uv[0], normal_uv[2]])
    return uv

def area(u, v, p):
    #u = np.array(u)
    #v = np.array(v)
    #p = np.array(p)
    pv = v - p
    pu = u - p

    re_sin = np.sin(ang_vectors(pv, pu))# + 1e-25
    area = (0.5)*norma(pv)*norma(pu)*re_sin
    if area < 0:
        area = -area
    #print area, 'area', area == 0.0, re_sin, re_sin == 0.0, 're_sin'
    return area

def mid_point(p1, p2):
    coords_p1 = mb.get_coords(p1)
    coords_p2 = mb.get_coords(p2)
    mid_p = (coords_p1 + coords_p2)/2.0
    return mid_p

def get_centroid(entity):

    # verts = mb.get_adjacencies(entity, 0)
    # coords = mb.get_coords(verts).reshape(len(verts), 3)
    # # print("Coords test: ", coords)
    # centroide = sum(coords)/(len(verts))
    verts = mb.get_adjacencies(entity, 0)
    coords = np.array([mb.get_coords([vert]) for vert in verts])
    pseudo_cent = get_centroid_reg(entity)
    vectors = np.array([coord - pseudo_cent for coord in coords])
    vectors = vectors.flatten()
    vectors = np.reshape(vectors, (len(verts), 3))
    directions = np.zeros(len(vectors))
    for j in range(len(vectors)):
        direction = ang_vectors(vectors[j], [1,0,0])
        if vectors[j, 1] <= 0:
            directions[j] = directions[j] + 2.0*pi - direction
        else:
            directions[j] = directions[j] + direction
    indices = np.argsort(directions)
    vect_std = vectors[indices]
    total_area = 0
    wgtd_cent = 0
    for i in range(len(vect_std)):
        norma1 = norma(vect_std[i])
        norma2 = norma(vect_std[i-1])
        ang_vect = ang_vectors(vect_std[i], vect_std[i-1])
        area_tri = (0.5)*norma1*norma2*np.sin(ang_vect)
        cent_tri = pseudo_cent + (1/3.0)*(vect_std[i] + vect_std[i-1])
        wgtd_cent = wgtd_cent + area_tri*cent_tri
        total_area = total_area + area_tri

    centroide = wgtd_cent/total_area
    return centroide


def K_n_X(node_face_coords, entity_centroid, perm):
    face_normal = norm_vec(
        node_face_coords[0], node_face_coords[1], entity_centroid)

    module_face_normal_squared = np.dot(face_normal, face_normal)
    #print("face_normal: ", face_normal, "perm:", perm, "module_2: ", module_face_normal_squared)
    K_n = np.dot(np.dot(face_normal, perm), face_normal)/module_face_normal_squared

    # print("K_n: ", K_n, K_n == 1.0)
    return K_n

def K_t_X(node_face_coords, entity_centroid, perm):
    count_wise_face = count_wise(
        node_face_coords[0], node_face_coords[1], entity_centroid)
    face_normal = norm_vec(
        node_face_coords[0], node_face_coords[1], entity_centroid)

    module_face_normal_squared = np.dot(face_normal, face_normal)
    K_t = np.dot(np.dot(face_normal, perm), count_wise_face)/module_face_normal_squared
    # print("K_t: ", K_t, K_t == 0.0)
    return K_t

def h_X(node_face_coords, entity_centroid, perm):
    face_normal = norm_vec(node_face_coords[0], node_face_coords[1], entity_centroid)
    face_centroid = (node_face_coords[0] + node_face_coords[1])/2.0
    module_face_normal = sqrt(np.dot(face_normal, face_normal))

    h = np.dot(face_normal, face_centroid - entity_centroid)/module_face_normal
    # print("h: ", h)
    # print(node_face_coords)
    # print(entity_centroid)
    return h


def face_weight(interp_node, face):
    coords_interp_node = mb.get_coords([interp_node])
    adjacent_volumes = mb.get_adjacencies(face, 2)
    half_face = mtu.get_average_position([face])
    csi_num = 0
    csi_den = 0
    for adjacent_volume in adjacent_volumes:
        centroid_volume = get_centroid(adjacent_volume)
        perm_volume = mb.tag_get_data(perm_tag, adjacent_volume).reshape([3, 3])

        interp_node_adj_faces = set(mtu.get_bridge_adjacencies(interp_node, 0, 1))
        adjacent_volume_adj_faces = set(mtu.get_bridge_adjacencies(adjacent_volume, 1, 1))

        faces = np.asarray(list(interp_node_adj_faces & adjacent_volume_adj_faces), dtype='uint64')
        other_face = faces[faces != face]
        half_other_face = mtu.get_average_position(other_face)

        K_bar_n = K_n_X(np.asarray([half_other_face, half_face]), centroid_volume, perm_volume)

        aux_dot_num = np.dot(coords_interp_node - half_other_face, half_face - half_other_face)
        cot_num = aux_dot_num / (2.0 * area(half_face, half_other_face, coords_interp_node))
        # print("cot_num: ", cot_num, coords_interp_node, half_face, get_centroid(adjacent_volume))
        vector_pseudo_face = half_face - half_other_face
        normal_pseudo_face = norm_vec(half_face, half_other_face, coords_interp_node)
        module_squared_vector = np.dot(vector_pseudo_face, vector_pseudo_face)
        K_bar_t = np.dot(np.dot(normal_pseudo_face, perm_volume), vector_pseudo_face)/module_squared_vector

        csi_num += K_bar_n * cot_num + K_bar_t
        # print("")
        K_den_n = K_n_X(np.asarray([coords_interp_node, half_face]), centroid_volume, perm_volume)

        aux_dot_den = np.dot(half_face - coords_interp_node, centroid_volume - coords_interp_node)
        cot_den = aux_dot_den / (2.0 * area(half_face, centroid_volume, coords_interp_node))

        K_den_t = K_t_X(np.asarray([coords_interp_node, half_face]), centroid_volume, perm_volume)

        csi_den += K_den_n * cot_den + K_den_t

    csi = csi_num / csi_den
    # print("csi: ", csi, coords_interp_node, half_face)
    return csi



def partial_weight(interp_node, adjacent_volume):
    coords_interp_node = mb.get_coords([interp_node])
    perm_adjacent_volume = mb.tag_get_data(perm_tag, adjacent_volume).reshape([3, 3])
    centroid_adjacent_volume = get_centroid(adjacent_volume)

    adjacent_faces = list(set(mtu.get_bridge_adjacencies(interp_node, 0, 1)) &
                          set(mtu.get_bridge_adjacencies(adjacent_volume, 1, 1)))
    first_face = adjacent_faces[0]
    second_face = adjacent_faces[1]

    half_first_face = mtu.get_average_position([first_face])
    half_second_face = mtu.get_average_position([second_face])

    K_ni_first = K_n_X(np.asarray([coords_interp_node, half_first_face]), centroid_adjacent_volume, perm_adjacent_volume)
    K_ni_second = K_n_X(np.asarray([coords_interp_node, half_second_face]), centroid_adjacent_volume, perm_adjacent_volume)

    nodes_first_face = mb.get_adjacencies(first_face, 0)
    nodes_second_face = mb.get_adjacencies(second_face, 0)

    coords_nodes_first_face = mb.get_coords(nodes_first_face).reshape([2, 3])
    coords_nodes_second_face = mb.get_coords(nodes_second_face).reshape([2, 3])

    h_first = h_X(coords_nodes_first_face, centroid_adjacent_volume, perm_adjacent_volume)
    h_second = h_X(coords_nodes_second_face, centroid_adjacent_volume, perm_adjacent_volume)

    half_vect_first_face = half_first_face - coords_interp_node
    half_vect_second_node = half_second_face - coords_interp_node

    half_modsqrd_first_face = sqrt(np.dot(half_vect_first_face, half_vect_first_face))
    half_modsqrd_second_face = sqrt(np.dot(half_vect_second_node, half_vect_second_node))

    neta_first = half_modsqrd_first_face / h_first
    neta_second = half_modsqrd_second_face / h_second
    # print("netas: ", neta_first, neta_second, h_first, h_second, coords_interp_node, centroid_adjacent_volume)
    csi_first = face_weight(interp_node, first_face)
    csi_second = face_weight(interp_node, second_face)

    node_weight = K_ni_first * neta_first * csi_first + K_ni_second * neta_second * csi_second
    # print("weight: ", node_weight, coords_interp_node, centroid_adjacent_volume)
    return node_weight


def neumann_boundary_weight(neumann_node, neumann_tag):
    adjacent_faces = mtu.get_bridge_adjacencies(neumann_node, 0, 1)
    #print len(face_adj)
    coords_neumann_node = mb.get_coords([neumann_node])
    neumann_term = 0
    for face in adjacent_faces:
        try:
            neu_flow_rate = mb.tag_get_data(neumann_tag, face)
            face_nodes = mtu.get_bridge_adjacencies(face, 0, 0)
            other_node = np.extract(face_nodes != np.array([neumann_node]), face_nodes)
            #print pts_adj_face[0], pts_adj_face[1], neumann_node, other_node
            other_node = np.asarray(other_node, dtype='uint64')
            coords_other_node = mb.get_coords(other_node)
            #print other_node, coords_other_node, len(face_adj)
            half_face = mtu.get_average_position([face])

            csi_neu = face_weight(neumann_node, face)
            #print csi_neu, half_face, coords_neumann_node
            #print csi_neu, coords_neumann_node, coords_other_node, half_face, len(face_adj)
            module_half_face = norma(half_face - coords_neumann_node)
            neumann_term += (1 + csi_neu) * module_half_face * neu_flow_rate
            print("Teste neumann: ", half_face, neu_flow_rate)
        except RuntimeError:
            continue
    # print("Neumann term: ", neumann_term)
    return neumann_term



def explicit_weights(interp_node):
    adjacent_volumes = mtu.get_bridge_adjacencies(interp_node, 0, 2)
    weight_sum = 0
    weights = []
    for adjacent_volume in adjacent_volumes:
        weight = partial_weight(interp_node, adjacent_volume)
        weights.append(weight)
        weight_sum += weight
        # print("volumes: ", mb.get_coords([interp_node]), get_centroid(adjacent_volume))
    weights = weights / weight_sum
    # print("weights: ", weights, )
    volumes_weights = [[vol, weigh] for vol, weigh in zip(adjacent_volumes, weights)]
    print("pesos: ", mb.get_coords([interp_node]), volumes_weights)
    return volumes_weights

def well_influence(coord_x, coord_y, coord_z, radius, source_term):
    all_volumes = mb.get_entities_by_dimension(root_set, 2)
    for a_volume in all_volumes:
        volume_centroid = get_centroid(a_volume)

    pass


def MPFA_D(dirichlet_nodes, neumann_nodes, intern_nodes):
    nodes_weights = {}
    ncount = 0
    for intern_node in intern_nodes:
        intern_node_weight = explicit_weights(intern_node)
        nodes_weights[intern_node] = intern_node_weight
        print("No ", ncount, " de", len(intern_nodes))
        ncount = ncount + 1
        # print("Pesos: ", intern_node, mb.get_coords([intern_node]), nodes_weights)
    neumann_nodes_weights = {}
    print("-------------------------------------------------------------------")
    print("Calculou pesos de nos internos!")
    print("-------------------------------------------------------------------")

    for neumann_node in neumann_nodes:
        neumann_node_weight = explicit_weights(neumann_node)
        neumann_nodes_weights[neumann_node] = neumann_node_weight
        print("neu_node:  ", mb.get_coords([neumann_node]))
    print("-------------------------------------------------------------------")
    print("Calculou pesos de nos em contorno de neumann!")
    print("-------------------------------------------------------------------")
    two_d_entities = mb.get_entities_by_dimension(0, 2)
    v_ids = dict(zip(two_d_entities, np.arange(0, len(two_d_entities))))

    for ent in two_d_entities:
        print("v_ids: ", v_ids[ent], get_centroid(ent))
    A = np.zeros([len(two_d_entities), len(two_d_entities)])
    B = np.zeros([len(two_d_entities), 1])
    all_faces = mb.get_entities_by_dimension(root_set, 1)
    count = 0
    for face in all_faces:
        adjacent_entitites = np.asarray(mb.get_adjacencies(face, 2), dtype='uint64')
        if len(adjacent_entitites) == 1:
            try:
                neumann_flux = mb.tag_get_data(neumann_tag, face)
                B[v_ids[adjacent_entitites[0]]][0] += -neumann_flux
            except RuntimeError:
                centroid_adjacent_entity = get_centroid(adjacent_entitites)
                face_nodes = np.asarray(mb.get_adjacencies(face, 0), dtype='uint64')
                coord_face_nodes = np.reshape(mb.get_coords(face_nodes), (2, 3))
                count_wise_face = count_wise(coord_face_nodes[0], coord_face_nodes[1], centroid_adjacent_entity)
                if np.dot(count_wise_face, coord_face_nodes[1] - coord_face_nodes[0]) < 0:
                    coord_face_nodes[[0,1]] = coord_face_nodes[[1,0]]
                    face_nodes[[0,1]] = face_nodes[[1,0]]

                pressure_first_node = mb.tag_get_data(dirichlet_tag, face_nodes[0])
                pressure_second_node = mb.tag_get_data(dirichlet_tag, face_nodes[1]);
                perm_adjacent_entity = mb.tag_get_data(perm_tag, adjacent_entitites).reshape([3, 3])
                face_normal = norm_vec(
                    coord_face_nodes[0], coord_face_nodes[1], centroid_adjacent_entity)

                K_n_B = K_n_X(coord_face_nodes, centroid_adjacent_entity, perm_adjacent_entity)
                #print("K_n_B: ", K_n_B, "coords: ", coord_face_nodes, "nodes: ", face_nodes)
                K_t_B = K_t_X(coord_face_nodes, centroid_adjacent_entity, perm_adjacent_entity)
                h_B = h_X(coord_face_nodes, centroid_adjacent_entity, perm_adjacent_entity)
                module_face_normal = sqrt(np.dot(face_normal, face_normal))

                aux_dot_product_first = np.dot(
                    centroid_adjacent_entity - coord_face_nodes[1], coord_face_nodes[0] - coord_face_nodes[1])
                # print("B2Ob: ", centroid_adjacent_entity - coord_face_nodes[1])
                # print("B2B1: ", coord_face_nodes[0] - coord_face_nodes[1])
                B[v_ids[adjacent_entitites[0]]][0] += (
                    (K_n_B * aux_dot_product_first)/(h_B * module_face_normal) - K_t_B) * pressure_first_node
                # print("aux_prod: ", aux_dot_product_first)
                # print("B1: ", ((K_n_B * aux_dot_product_first)/(h_B * module_face_normal) - K_t_B) * pressure_first_node)
                aux_dot_product_second = np.dot(
                    centroid_adjacent_entity - coord_face_nodes[0], coord_face_nodes[1] - coord_face_nodes[0])
                B[v_ids[adjacent_entitites[0]]][0] += (
                    (K_n_B * aux_dot_product_second)/(h_B * module_face_normal) + K_t_B) * pressure_second_node
                # print("B2: ", ((K_n_B * aux_dot_product_second)/(h_B * module_face_normal) + K_t_B) * pressure_second_node)
                A[v_ids[adjacent_entitites[0]]][v_ids[adjacent_entitites[0]]] += K_n_B * module_face_normal/h_B
                #print("Dirich. coefic.: ", A, K_n_B * module_face_normal/h_B)

        if len(adjacent_entitites) == 2:

            first_volume = adjacent_entitites[0]
            second_volume = adjacent_entitites[1]

            cent_first_volume = get_centroid([first_volume])
            cent_second_volume = get_centroid([second_volume])

            perm_first_volume = mb.tag_get_data(perm_tag, first_volume).reshape([3, 3])
            perm_second_volume = mb.tag_get_data(perm_tag, second_volume).reshape([3, 3])

            face_nodes = np.asarray(mb.get_adjacencies(face, 0), dtype='uint64')
            coord_face_nodes = np.reshape(mb.get_coords(face_nodes), (2, 3))

            count_wise_face = count_wise(coord_face_nodes[0], coord_face_nodes[1], cent_first_volume)
            if np.dot(count_wise_face, coord_face_nodes[1] - coord_face_nodes[0]) < 0:
                coord_face_nodes[[0,1]] = coord_face_nodes[[1,0]]
                face_nodes[[0,1]] = face_nodes[[1,0]]

            K_n_first = K_n_X(coord_face_nodes, cent_first_volume, perm_first_volume)
            K_n_second = K_n_X(coord_face_nodes, cent_second_volume, perm_second_volume)

            K_t_first = K_t_X(coord_face_nodes, cent_first_volume, perm_first_volume)
            K_t_second = K_t_X(coord_face_nodes, cent_second_volume, perm_second_volume)

            h_first = h_X(coord_face_nodes, cent_first_volume, perm_first_volume)
            h_second = h_X(coord_face_nodes, cent_second_volume, perm_second_volume)

            K_transm = (K_n_first * K_n_second) / (K_n_first * h_second + K_n_second * h_first)
            #print('K_transm: ', K_transm)
            face_normal = norm_vec(
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
            # print(A)
            A[v_ids[first_volume]][v_ids[first_volume]] += K_transm * module_face_normal
            # print("A, mesmo vol:")
            # print(A)
            A[v_ids[first_volume]][v_ids[second_volume]] += - K_transm * module_face_normal
            # print("V_ids: ", v_ids[first_volume], v_ids[second_volume])
            # print(- K_transm * module_face_normal)
            # print("Linha primo para seg")
            # print(A)
            A[v_ids[second_volume]][v_ids[second_volume]] +=  K_transm * module_face_normal
            # print("Seg vol")
            # print(A)
            A[v_ids[second_volume]][v_ids[first_volume]] +=  - K_transm * module_face_normal
            # print("V_ids: ", v_ids[first_volume], v_ids[second_volume])
            # print(- K_transm * module_face_normal)
            # print("Linha seg para primo")
            # print(A)

            if face_nodes[0] in intern_nodes:
                # print("No 1 volume 1: ", mb.get_coords([face_nodes[0]]), mb.get_coords([face_nodes[1]]), cent_first_volume)
                for vol, weigh in nodes_weights[face_nodes[0]]:
                    A[v_ids[first_volume]][v_ids[vol]] += - K_transm * module_face_normal * D_ab * weigh
                    A[v_ids[second_volume]][v_ids[vol]] +=  K_transm * module_face_normal * D_ab * weigh
                    # print("Aval: ", K_transm * module_face_normal * D_ab * weigh, D_ab)

            elif face_nodes[0] in dirichlet_nodes:
                pressure_first_node = mb.tag_get_data(dirichlet_tag, face_nodes[0])
                B[v_ids[first_volume]][0] += K_transm * module_face_normal * D_ab * pressure_first_node[0][0]
                B[v_ids[second_volume]][0] += - K_transm * module_face_normal * D_ab * pressure_first_node[0][0]

            elif face_nodes[0] in neumann_nodes:
                adjacent_blocks = mtu.get_bridge_adjacencies(face_nodes[0], 0, 2)
                block_weight_sum = 0
                for a_block in adjacent_blocks:
                    block_weight = partial_weight(face_nodes[0], a_block)
                    block_weight_sum += block_weight
                neumann_node_factor = neumann_boundary_weight(face_nodes[0], neumann_tag) / block_weight_sum

                B[v_ids[first_volume]][0] += K_transm * module_face_normal * D_ab * (-neumann_node_factor)
                B[v_ids[second_volume]][0] += - K_transm * module_face_normal * D_ab * (-neumann_node_factor)
                for vol, weigh in neumann_nodes_weights[face_nodes[0]]:
                    A[v_ids[first_volume]][v_ids[vol]] += - K_transm * module_face_normal * D_ab * weigh
                    A[v_ids[second_volume]][v_ids[vol]] +=  K_transm * module_face_normal * D_ab * weigh


            if face_nodes[1] in intern_nodes:
                for vol, weigh in nodes_weights[face_nodes[1]]:
                    A[v_ids[first_volume]][v_ids[vol]] += K_transm * module_face_normal * D_ab * weigh
                    A[v_ids[second_volume]][v_ids[vol]] += - K_transm * module_face_normal * D_ab * weigh

            elif face_nodes[1] in dirichlet_nodes:
                pressure_second_node = mb.tag_get_data(dirichlet_tag, face_nodes[1])
                B[v_ids[first_volume]][0] += - K_transm * module_face_normal * D_ab * pressure_second_node[0][0]
                B[v_ids[second_volume]][0] += K_transm * module_face_normal * D_ab * pressure_second_node[0][0]

            elif face_nodes[1] in neumann_nodes:
                adjacent_blocks = mtu.get_bridge_adjacencies(face_nodes[1], 0, 2)
                block_weight_sum = 0
                for a_block in adjacent_blocks:
                    block_weight = partial_weight(face_nodes[1], a_block)
                    block_weight_sum += block_weight
                neumann_node_factor = neumann_boundary_weight(face_nodes[1], neumann_tag) / block_weight_sum
                B[v_ids[first_volume]][0] += - K_transm * module_face_normal * D_ab * (-neumann_node_factor)
                B[v_ids[second_volume]][0] += K_transm * module_face_normal * D_ab * (-neumann_node_factor)
                for vol, weigh in neumann_nodes_weights[face_nodes[1]]:
                    # print("Peso neumann: ", weigh, mb.get_coords([face_nodes[1]]), get_centroid(vol), get_centroid(first_volume), get_centroid(second_volume))
                    A[v_ids[first_volume]][v_ids[vol]] += K_transm * module_face_normal * D_ab * weigh
                    A[v_ids[second_volume]][v_ids[vol]] += - K_transm * module_face_normal * D_ab * weigh

        print("Calculou face ", count, " de ", len(all_faces))
        count = count + 1

    print(A)
    print(B)
    inv_A = np.linalg.inv(A)
    print("Inv: ", inv_A)
    volume_pressures = np.dot(inv_A, B)
    print(volume_pressures)
    mb.tag_set_data(pressure_tag, two_d_entities, volume_pressures.flatten())
    mb.write_file("out_file.vtk")
    return volume_pressures


def grad_trian(entity, p1, p2, K):
    coord_p1 = mb.get_coords([p1])
    coord_p2 = mb.get_coords([p2])
    coord_cent = get_centroid(entity)
    #print coord_p1, coord_p2, coord_cent
    area_tri = area(coord_p1, coord_p2, coord_cent)

    press_1 = node_pressure(p1, pressure_tag, K).flatten()
    press_2 = node_pressure(p2, pressure_tag, K).flatten()
    press_cent = mb.tag_get_data(pressure_tag, entity).flatten()
    #print press_1, press_2, press_cent
    normal_op_p1 = norm_vec(coord_p2, coord_cent, coord_p1)
    normal_op_p2 = norm_vec(coord_p1, coord_cent, coord_p2)
    normal_op_cent = norm_vec(coord_p1, coord_p2, coord_cent)

    grad_tri = (-1/(2*area_tri))*(
        press_1*normal_op_p1 +
        press_2*normal_op_p2 +
        press_cent*normal_op_cent)
    #print press_1, press_2, area_tri
    #print grad_tri, coord_cent
    return grad_tri

def error_indicator(error_tag, K):
    #root_set = mb.get_root_set()
    entities_EI = mb.get_entities_by_dimension(root_set, 2)
    ent_norm_error = np.array([])
    for ent in entities_EI:
        ent_press = mb.tag_get_data(pressure_tag, ent)
        coord_cent = get_centroid(ent)
        edges = mb.get_adjacencies(ent, 1, True)
        grad_vectors = np.array([])
        tri_cent = np.array([])

        for ed in edges:
            nodes = mb.get_adjacencies(ed, 0)
            grad_vec = grad_trian(ent, nodes[0], nodes[1], K)

            grad_vec = grad_vec.flatten()
            grad_vectors = np.append(grad_vectors, grad_vec)

            coord_0 = mb.get_coords([nodes[0]])
            coord_1 = mb.get_coords([nodes[1]])

            u = coord_0 - coord_cent
            v = coord_1 - coord_cent
            t_cent = coord_cent + (u + v)/3
            #print grad_vec, t_cent
            tri_cent = np.append(tri_cent, t_cent)

        tri_cent = np.reshape(tri_cent, (len(edges), 3))
        #print tri_cent
        grad_vectors = np.reshape(grad_vectors, (len(edges), 3))
        error_L = np.array([])

        while len(grad_vectors) != 0:
            for g, L in zip(grad_vectors[1:], tri_cent[1:]):
                delta_grad = g - grad_vectors[0]
                delta_position = L - tri_cent[0]
                grad_var = abs(np.dot(delta_grad, delta_position))
                #print grad_var, delta_grad, delta_position, 'grad_var'
                error_L = np.append(error_L, grad_var)
            grad_vectors = grad_vectors[1:]
            tri_cent = tri_cent[1:]

        avg_error_L = np.average(error_L)
        norm_error = sqrt(avg_error_L)
        #print avg_error_L, norm_error
        ent_norm_error = np.append(ent_norm_error, norm_error)
        mb.tag_set_data(error_tag, ent, norm_error)
    #print 'errors', ent_norm_error
    global_error = sqrt(np.dot(ent_norm_error, ent_norm_error)/len(ent_norm_error))
    #mb.write_file('pressure_error_field.vtk')
    return global_error


# In[123]:

def refine_degree(error_tol, error_tag, ref_degree_tag):
    entities_RG = mb.get_entities_by_dimension(root_set, 2)
    #mb.tag_set_data(ref_degree_tag, entities_RG, np.zeros(1, len(entities_RG)))
    count = len(entities_RG)
    for ent in entities_RG:
        coord_cent = get_centroid(ent)
        faces = mb.get_adjacencies(ent, 1, True)
        min_dist = np.array([])
        for face in faces:
            nodes = mb.get_adjacencies(face, 0)
            half_edge = mid_point([nodes[0]], [nodes[1]])
            dist_edge_cent = sqrt(np.dot(
                half_edge - coord_cent, half_edge - coord_cent))
            min_dist = np.append(min_dist, dist_edge_cent)

        d_init = np.amin(min_dist)
        error_init = mb.tag_get_data(error_tag, ent)

        d_final = d_init*error_tol/error_init

        ref_DEG = log10(d_init/d_final)/log10(2)
        trunc_DEG = trunc(ref_DEG)
        if ref_DEG - trunc_DEG >= 0.25:
            ref_DEG = ceil(ref_DEG)
        else:
            ref_DEG = floor(ref_DEG)

        mb.tag_set_data(ref_degree_tag, ent, np.asarray([ref_DEG], dtype='float64'))
        #print("REF: ", ref_DEG)
        #print 'element', count
    mb.write_file('ref_test.vtk')

def unit_step(ref_degree_tag):
    entities_1_step = mb.get_entities_by_dimension(root_set, 2)
    aux_count = 0
    count = 1
    while count > 0:
        count = 0
        for ent in entities_1_step:

            ent_ref_degree = mb.tag_get_data(ref_degree_tag, ent)
            if ent_ref_degree <= 1:
                continue

            bridge_blocks = mtu.get_bridge_adjacencies(ent, 1, 2)
            for a_bridge_block in bridge_blocks:

                bridge_block_ref_degree = mb.tag_get_data(ref_degree_tag, a_bridge_block)
                ref_degree_diff = ent_ref_degree - bridge_block_ref_degree
                if bridge_block_ref_degree >= ent_ref_degree:
                    continue

                elif bridge_block_ref_degree < ent_ref_degree and ref_degree_diff == 1:
                    continue

                elif bridge_block_ref_degree < ent_ref_degree and ref_degree_diff > 1:
                    new_ref_degree = ent_ref_degree - 1
                    #print new_ref_degree, 'new_ref_degree'
                    #print a_bridge_block, 'a_bridge_block'
                    mb.tag_set_data(ref_degree_tag, a_bridge_block, new_ref_degree)
                    count = count + 1
                    aux_count = aux_count + 1
                    #mb.write_file('ref_test_{0}.vtk'.format(aux_count))
    #Regulariza a malha
    count = 1
    while count > 0:
        count = 0
        for ent in entities_1_step:
            ent_ref_degree = mb.tag_get_data(ref_degree_tag, ent)
            #if ent_ref_degree <= 0:
            if ent_ref_degree <= 1:
                bridge_blocks = mtu.get_bridge_adjacencies(ent, 1, 2)
                count_b = 0
                for a_bridge_block in bridge_blocks:
                    a_bb_ref_degree = mb.tag_get_data(ref_degree_tag, a_bridge_block)
                    if a_bb_ref_degree <= ent_ref_degree:
                        continue
                    elif a_bb_ref_degree > ent_ref_degree:
                        count_b = count_b + 1
                if count_b > 1:
                    #print ent
                    mb.tag_set_data(ref_degree_tag, ent, np.asarray(ent_ref_degree) + 1.0)
                    count = count + 1
                    aux_count = aux_count + 1
                    #mb.write_file('ref_test_{0}.vtk'.format(aux_count))
                    #print 'regularizou bloco', get_centroid([ent])
                else:
                    continue
    mb.write_file('ref_unit_step_reg.vtk')


# In[124]:

count = 0
def nodewise_adaptation(ent, ref_degree):
    global count
    ent = long(ent)
    #print ent, 'ent'

    coord_cent = get_centroid(ent)
    vert_cent = mb.create_vertices(np.array(coord_cent))
    nodes = mb.get_adjacencies(ent, 0)
    coord_nodes = mb.get_coords(nodes)
    #coord_nodes = coord_nodes.flatten()
    coord_nodes = np.reshape(coord_nodes, (len(nodes), 3))
    #print coord_nodes, 'coord_nodes'

    vectors = np.array([crd_node - coord_cent for crd_node in coord_nodes])

    directions = np.zeros(len(vectors))
    for j in range(len(vectors)):
        direction = ang_vectors(vectors[j], [1, 0, 0])
        if vectors[j, 1] <= 0:
            directions[j] = directions[j] + 2.0*pi - direction
        else:
            directions[j] = directions[j] + direction
    indices = np.argsort(directions)
    nodes = np.array(nodes, dtype = 'uint64')
    #print nodes, 'nodes'
    nodes_sorted = nodes[indices]
    triangles_verts = np.array([])
    for i in range(len(nodes_sorted)):
        tri_vertices = np.append([nodes_sorted[i-1], nodes_sorted[i]], vert_cent)
        tri_vertices = np.array(tri_vertices, dtype = 'uint64')
        triangles_verts = np.append(triangles_verts, tri_vertices)

    triangles_verts = np.reshape(triangles_verts, (len(nodes), 3))
    triangles_verts = np.array((triangles_verts), dtype = 'uint64')
    #print triangles_verts
    new_triangles = [
        mb.create_element(types.MBTRI, new_verts) for new_verts in triangles_verts]
    #aentities = mb.get_adjacencies(ent, 1)
    #for aent in aentities:
    #    mb.delete_entities([aent])
    mb.delete_entities([ent])
    triangles_verts = triangles_verts.flatten()
    #mtu.construct_aentities(triangles_verts)
    #print ent, 'ent'
    mb.write_file('output_adpt_{0}.vtk'.format(count))
    count = count + 1
    if ref_degree > 1:
        for tri in new_triangles:
            nodewise_adaptation(long(tri), ref_degree - 1)



# In[125]:

count = 0
def edgewise_adaptation(elem, hanging_nodes_tag, full_edges_tag, ref_degree_tag):
    global count
    global entities_to_adapt

    #print("ref_test: ", ref_degree)
    try:
        hanging_nodes = mb.tag_get_data(hanging_nodes_tag, elem)
        hanging_nodes = set(mb.get_entities_by_handle(hanging_nodes))
    except RuntimeError:
        hanging_nodes = set()
    try:
        full_edges = mb.tag_get_data(full_edges_tag, elem)
        full_edges = set(mb.get_entities_by_handle(full_edges))
    except RuntimeError:
        full_edges = set()

    corner_nodes = set(mb.get_adjacencies(elem, 0))
    corner_nodes = corner_nodes - hanging_nodes

    #Adapta vizinhos com hanging node igual a um corner node
    all_neighbours = mtu.get_bridge_adjacencies(elem, 1, 2)
    for neigh in all_neighbours:
        try:
            neigh_hanging_nodes = mb.tag_get_data(hanging_nodes_tag, neigh)
            neigh_hanging_nodes = set(mb.get_entities_by_handle(neigh_hanging_nodes))

            corners_hanging_on_neighbours = neigh_hanging_nodes & corner_nodes
            corners_hanging_on_neighbours = list(corners_hanging_on_neighbours)

            if len(corners_hanging_on_neighbours) >= 1:
                entities_to_adapt.remove(elem)
                entities_to_adapt.append(elem)
                return True
                #neigh_ref_degree = mb.tag_get_data(ref_degree_tag, neigh)
                #if neigh_ref_degree >= 1:
                    #back_info = edgewise_adaptation(
                        #neigh, hanging_nodes_tag, full_edges_tag, ref_degree_tag)
            else:
                continue

        except RuntimeError:
            continue

    ref_degree = mb.tag_get_data(ref_degree_tag, elem)
    print("count: ", count, mb.tag_get_data(ref_degree_tag, elem), ref_degree, get_centroid(elem), len(entities_to_adapt))
    if ref_degree < 1:
        print("passou")
        return False

    full_edges = list(full_edges)
    #Tratamento do vizinho

    for full_edge in full_edges:
        # divide full_edge no meio
        # trata o vizinho (adicionando esse nó como hanging node pra ele tb)
        # pega o nó do meio dela e coloca em hanging_nodes
        # pega o nó do meio e dá construct_aentities

        nodes = mb.get_adjacencies(full_edge, 0)
        coord_half_node = mid_point([nodes[0]], [nodes[1]])
        half_node = mb.create_vertices(coord_half_node)
        mtu.construct_aentities(half_node)
        hanging_nodes = hanging_nodes | set(np.asarray(half_node, 'uint64'))

        #neighbours_node_0 = set(mtu.get_bridge_adjacencies(nodes[0], 0, 2))
        #neighbours_node_1 = set(mtu.get_bridge_adjacencies(nodes[1], 0, 2))

        #neighbours = neighbours_node_0 & neighbours_node_1
        #neighbours = mtu.get_bridge_adjacencies(full_edge, 1, 2)
        neighbours = mtu.get_bridge_adjacencies(full_edge, 1, 2)
        neighbours = np.asarray(list(neighbours), dtype = 'uint64')

        if len(neighbours) == 1 or len(neighbours) == 0:
            continue

        #elem = np.asarray([elem], dtype = 'uint64')
        neighbour = np.asarray(np.extract(neighbours != np.asarray([elem], dtype = 'uint64'), neighbours), dtype = 'uint64')
        #print("elem está: ", neighbours[0]==elem, neighbours[1]==elem)
        #neighbour = list(neighbour)
        #print("compare: ", neighbour == elem)
        if len(neighbour) != 1:
            #print("elem: ", elem, "neighbours: ", neighbours)
            #print(neighbours[0]==elem, neighbours[1]==elem)
            #print("neighbours: ", len(neighbours), neighbours)
            #print(neighbour, neighbour[0], neighbour[1], 'neighbour')
            #print("centroids: ", get_centroid([neighbour[0]]), get_centroid([neighbour[1]]))
            continue
            try:
                full_missing = mb.tag_get_data(full_edges_tag, neighbour[0])
                if entities_to_adapt.count(neighbour[1]) == 1:
                    entities_to_adapt.remove(neighbour[1])
                ghost_elem_edges = mb.get_adjacencies(neighbour[1], 1)
                for ghost_edge in ghost_elem_edges:
                    mb.delete_entities([ghost_edge])
                mb.delete_entities([neighbour[1]])
                neighbour = np.asarray([neighbour[0]])
            except RuntimeError:
                if entities_to_adapt.count(neighbour[0]) == 1:
                    entities_to_adapt.remove(neighbour[0])
                ghost_elem_edges = mb.get_adjacencies(neighbour[0], 1)
                for ghost_edge in ghost_elem_edges:
                    mb.delete_entities([ghost_edge])
                mb.delete_entities([neighbour[0]])
                neighbour = np.asarray([neighbour[1]])
        #print neighbours, neighbour, elem

        #print('Neigh: ', len(neighbour))
        neighbour_centroid = get_centroid(np.asarray([neighbour]))
        #neighbour = np.asarray(neighbour)
        try:
            neigh_hanging_nodes = mb.tag_get_data(hanging_nodes_tag, neighbour)
            neigh_hanging_nodes = set(mb.get_entities_by_handle(neigh_hanging_nodes))
        except RuntimeError:
            neigh_hanging_nodes = set()


        half_node = set(np.asarray(half_node, 'uint64'))
        neigh_hanging_nodes = neigh_hanging_nodes | half_node


        #half_node = set(np.asarray(half_node, 'uint64'))
        neighbour_nodes = mb.get_adjacencies(neighbour, 0)
        cor_neigh = neighbour_nodes
        neighbour_nodes = set(np.asarray(neighbour_nodes, 'uint64'))
        neighbour_nodes = neighbour_nodes | half_node
        neighbour_nodes = list(neighbour_nodes)
        neighbour_nodes = np.asarray(neighbour_nodes, dtype = 'uint64')
        #print neighbour_nodes, 'neighbour_nodes'
        coord_neighbour_nodes = mb.get_coords(neighbour_nodes)

        coord_neighbour_nodes = np.reshape(
            coord_neighbour_nodes.flatten(), (len(neighbour_nodes), 3))

        vectors = np.array(
            [crd_node - neighbour_centroid for crd_node in coord_neighbour_nodes])

        directions = np.zeros(len(vectors))
        for j in range(len(vectors)):
            direction = ang_vectors(vectors[j], [1, 0, 0])
            if vectors[j, 1] <= 0:
                directions[j] = directions[j] + 2.0*pi - direction
            else:
                directions[j] = directions[j] + direction
        indices = np.argsort(directions)
        neighbour_nodes = np.array(neighbour_nodes, dtype = 'uint64')
        #print nodes, 'nodes'
        nodes_sorted = neighbour_nodes[indices]
        new_neighbour = mb.create_element(types.MBPOLYGON, nodes_sorted)
        mtu.construct_aentities(Range(nodes_sorted))
        neighbour_edges_before = mb.get_adjacencies(neighbour, 1, True)
        #print set(neighbour_edges_before), 'edges before'

        old_neigh_ref_degree = mb.tag_get_data(ref_degree_tag, neighbour)
        mb.tag_set_data(ref_degree_tag, new_neighbour, old_neigh_ref_degree)

        if entities_to_adapt.count(neighbour) == 1:
            #print 'neighbour trade'
            #where = entities_to_adapt.index(neighbour)
            entities_to_adapt.remove(neighbour)
            entities_to_adapt.append(new_neighbour)

        mb.delete_entities(neighbour)

        edges = set(mb.get_adjacencies(new_neighbour, 1, True))
        #print edges, 'edges after'

        neigh_hanging_nodes = list(neigh_hanging_nodes)
        neigh_hanging_nodes = np.asarray(neigh_hanging_nodes, dtype = 'uint64')
        to_remove_edges = set()

        for neigh_hanging_node in neigh_hanging_nodes:
            edges_to_remove = set(mtu.get_bridge_adjacencies(neigh_hanging_node, 0, 1))
            #print("edges_to_remove: ", edges_to_remove)
            to_remove_edges = to_remove_edges | edges_to_remove

        neigh_full_edges = edges - to_remove_edges
        neigh_full_edges = list(neigh_full_edges)
        neigh_full_edges_meshset = mb.create_meshset()
        mb.add_entities(neigh_full_edges_meshset, neigh_full_edges)
        mb.tag_set_data(full_edges_tag, new_neighbour, neigh_full_edges_meshset)

        mtu.construct_aentities(Range(nodes_sorted))

        neigh_hanging_nodes_meshset = mb.create_meshset()
        mb.add_entities(neigh_hanging_nodes_meshset, neigh_hanging_nodes)
        mb.tag_set_data(hanging_nodes_tag, new_neighbour, neigh_hanging_nodes_meshset)

    #Adaptacao do elemento atual
    corner_nodes = set(mb.get_adjacencies(elem, 0))
    corner_nodes = corner_nodes - hanging_nodes

    parcial_nodes = list(corner_nodes | hanging_nodes)
    parcial_nodes = np.asarray(parcial_nodes, dtype = 'uint64')
    coord_parcial_nodes = mb.get_coords(parcial_nodes)
    coord_parcial_nodes = np.reshape(coord_parcial_nodes.flatten(), (len(parcial_nodes), 3))
    coord_elem_cent = get_centroid(elem)

    vectors = np.array(
        [crd_node - coord_elem_cent for crd_node in coord_parcial_nodes])

    directions = np.zeros(len(vectors))
    for j in range(len(vectors)):
        direction = ang_vectors(vectors[j], [1, 0, 0])
        if vectors[j, 1] <= 0:
            directions[j] = directions[j] + 2.0*pi - direction
        else:
            directions[j] = directions[j] + direction
    indices = np.argsort(directions)
    parcial_nodes = np.asarray(parcial_nodes, dtype = 'uint64')
    parcial_nodes = parcial_nodes[indices]



    parcial_elem = mb.create_element(types.MBPOLYGON, parcial_nodes)
    mtu.construct_aentities(Range(parcial_nodes))

    elem_cent = mb.create_vertices(coord_elem_cent)
    elem_cent = np.asarray(elem_cent)
    new_elems = set()
    corner_nodes = list(corner_nodes)

    hanging_nodes_with_tag = []
    count = count + 1

    for corner_node in corner_nodes:
        mtu.construct_aentities(Range(corner_node))
        corner_node_nodes = mtu.get_bridge_adjacencies(corner_node, 1, 0)

        corner_node_nodes = set(np.asarray(corner_node_nodes, dtype = 'uint64'))
        hanging_nodes = set(np.array(list(hanging_nodes), dtype = 'uint64'))

        corner_node_hanging_nodes = corner_node_nodes & hanging_nodes
        #print('corner:', corner_node_nodes, 'hanging: ', hanging_nodes, 'corner_coord: ', mb.get_coords([corner_node]))
        #print('elem', get_centroid(elem), corner_node_nodes, 'corner_node_nodes', hanging_nodes, 'hanging_nodes', corner_node_hanging_nodes)
        corner_node_hanging_nodes = list(corner_node_hanging_nodes)
        hang_0 = corner_node_hanging_nodes[0]
        hang_1 = corner_node_hanging_nodes[1]

        new_corner_nodes = elem_cent
        new_corner_nodes = np.append(elem_cent, hang_0)
        new_corner_nodes = np.append(new_corner_nodes, corner_node)
        new_corner_nodes = np.append(new_corner_nodes, hang_1)
        new_corner_nodes = np.asarray(
            [corner_node, hang_0, elem_cent, hang_1], dtype = 'uint64')

        new_elem = mb.create_element(types.MBPOLYGON, new_corner_nodes)
        new_elems.add(new_elem)
        #print("Novos elementos", len(mb.get_entities_by_dimension(0, 2)))
        mtu.construct_aentities(Range(new_corner_nodes))
        new_elem_full_edges = mb.get_adjacencies(new_elem, 1, True)
        new_elem_full_edges_meshset = mb.create_meshset()
        mb.add_entities(new_elem_full_edges_meshset, new_elem_full_edges)
        mb.tag_set_data(full_edges_tag, new_elem, new_elem_full_edges_meshset)

        mb.tag_set_data(ref_degree_tag, new_elem, np.asarray(ref_degree) - 1)
        #print("ref_test2", np.asarray(ref_degree) - 1)

        bound_hang = np.asarray([], dtype='uint64')
        for hang in np.asarray([hang_0, hang_1]):
            adj_hang = mtu.get_bridge_adjacencies(hang, 0, 2)
            #print("adj_hang: ", len(np.asarray(adj_hang)))
            if len(adj_hang) <= 2:
                bound_hang = np.append(bound_hang, hang)
                #print("Boundary hangs: ", bound_hang)
        if len(bound_hang) == 1: #and bound_hang not in hanging_nodes_with_tag:
            try:
                new_node_tag_neumann = mb.tag_get_data(neumann_tag, corner_node)
                mb.tag_set_data(neumann_tag, bound_hang, new_node_tag_neumann)
                adj_bound_aents_hang = mtu.get_bridge_adjacencies(bound_hang, 0, 1)
                for aent in adj_bound_aents_hang:
                    if len(mtu.get_bridge_adjacencies(aent, 1, 2)) == 1:
                        mb.tag_set_data(neumann_tag, aent, new_node_tag_neumann)
                print("New Neumann Bounds!")
            except RuntimeError:
                print("Coords: ", mb.get_coords([corner_node]), corner_node)
                new_node_tag_dirichlet = mb.tag_get_data(dirichlet_tag, corner_node)
                mb.tag_set_data(dirichlet_tag, bound_hang, new_node_tag_dirichlet)
                print("New Dirichlet Node!")
            hanging_nodes_with_tag = np.append(hanging_nodes_with_tag, bound_hang)



    #aentities_to_remove_parcial_elem = mb.get_adjacencies(parcial_elem, 1)
    #for aent in aentities_to_remove_parcial_elem:
        #mb.delete_entities([aent])

    #aentities_to_remove_elem = mb.get_adjacencies(elem, 1)
    #for aent in aentities_to_remove_elem:
        #mb.delete_entities([aent])

    mb.delete_entities([elem])
    mb.delete_entities([parcial_elem])

    #for new_elem in new_elems:
        #mtu.construct_aentities(Range(new_elem))



    #print 'End of adaptation of ', coord_elem_cent
    new_elems = list(new_elems)
    #print new_elems, 'new_elems'
    #for new_elem in new_elems:


    if entities_to_adapt.count(elem) == 1:
        entities_to_adapt.remove(elem)

    mb.write_file('edgewise_adpt_{0}.vtk'.format(count))

    for new_elem in new_elems:
        new_ref_degree = mb.tag_get_data(ref_degree_tag, new_elem)
        if new_ref_degree >= 1:
            #print("ref_deg: ", new_ref_degree)
            entities_to_adapt.append(new_elem)
            #back_info = edgewise_adaptation(
                #new_elem, hanging_nodes_tag, full_edges_tag, ref_degree_tag)



        # liga corner_node a corner_node_hanging_nodes[0]
        # liga corner_node_hanging_nodes[0] a C
        # liga C a orner_node_hanging_nodes[1]
        # liga orner_node_hanging_nodes[1] a corner_node


# In[126]:

pressures = MPFA_D(dirichlet_nodes, neumann_nodes, intern_nodes)
mb.write_file("out_file.vtk")
print ('------------------------------------------------------------------')
print ("Campo de pressão calculado!")
print ('------------------------------------------------------------------')
global_error = error_indicator(error_tag, K)
print ('------------------------------------------------------------------')
print ('Erro calculado: global_error---> ', global_error)
print ('------------------------------------------------------------------')

refine_degree(0.08, error_tag, ref_degree_tag)
unit_step(ref_degree_tag)
mb.write_file('mpfa_d_teste.vtk')
print ('------------------------------------------------------------------')
print ('Grau de refinamento calculado!')
print ('------------------------------------------------------------------')
print ('Adaptando...')
entities_to_adapt = mb.get_entities_by_dimension(0, 2)
entities_to_adapt = list(entities_to_adapt)
adaptation = 0

while len(entities_to_adapt) > 0:
    #print (len(entities_to_adapt)), ('loop'), ('Adaptando...')
    #mb.write_file('adaptation_{0}.vtk'.format(adaptation))
    #import pdb; pdb.set_trace()
    #print("Volumes Update: ", len(mb.get_entities_by_dimension(root_set,2)))
    ent = entities_to_adapt[0]

    ref_degree = mb.tag_get_data(ref_degree_tag, ent)

    #volumes = mb.get_entities_by_dimension(root_set, 2)
    #ms = mb.create_meshset()
    #mb.add_entities(ms, volumes)
    #import pdb; pdb.set_trace()
    if ref_degree >= 1:
        adaptation = adaptation + 1
        #mb.write_file('adaptation_{0}.vtk'.format(adaptation))
        back_info = edgewise_adaptation(ent, hanging_nodes_tag, full_edges_tag, ref_degree_tag)
    else:
        entities_to_adapt.remove(ent)

mb.write_file('adaptation_{0}.vtk'.format(adaptation + 1))

pressures = MPFA_D(dirichlet_nodes, neumann_nodes, intern_nodes)
print("PRESSURES:", pressures)
all_verts = mb.get_entities_by_dimension(root_set, 0)
for vert in all_verts:
    coords = mb.get_coords([vert])
    print("Verts coords: ", coords)

all_volumes = mb.get_entities_by_dimension(root_set, 2)
for i in range(len(all_volumes)):
    coord_x = get_centroid(all_volumes[i])[0]
    # print("test: ", 1.0 - coord_x == pressures[i])
    print("Val: ", 1.0 - coord_x, pressures[i], (
            1.0 - coord_x) - pressures[i])#, get_centroid(all_volumes[i]))

print("-------------------------------------")
print("Campo de pressoes calculado!")
print("-------------------------------------")
print(pressures)

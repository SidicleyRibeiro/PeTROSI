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

def node_pressure(node, pressure_tag, K):
    pass
    #root_set= mb.get_root_set()
    try:
        node_press = mb.tag_get_data(dirichlet_tag, node)

    except RuntimeError:

        entities_NP = mb.get_entities_by_dimension(root_set, 2)
        around_blocks = mb.get_adjacencies(node, 2)
        sum_psi = 0.0
        wgtd_press = 0.0
        for a_block in around_blocks:
            block_press = mb.tag_get_data(pressure_tag, a_block)
            psi = psi_LPEW2(node, a_block, K)
            wgtd_press = wgtd_press + psi*block_press
            sum_psi = sum_psi + psi

        node_press = wgtd_press/sum_psi

        try:
            neumann_flow = mb.tag_get_data(neumann_tag, node)
            neu_add = Neumann_treat_Bk(node, neumann_tag, K)
            node_press = node_press - neu_add/sum_psi
        except RuntimeError:
            pass

    return node_press

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

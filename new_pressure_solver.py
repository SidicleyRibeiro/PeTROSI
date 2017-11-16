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
mb.load_file('teste_recombine.msh')
root_set = mb.get_root_set()


# In[115]:


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


# In[116]:

#para problema sem poço

physical_tag = mb.tag_get_handle("MATERIAL_SET")

physical_sets = mb.get_entities_by_type_and_tag(
    0, types.MBENTITYSET, np.array((physical_tag,)), np.array((None,)))

for tag in physical_sets:
    tag_id = mb.tag_get_data(physical_tag, np.array([tag]), flat=True)

    if tag_id == 201:
        entity_set_201 = mb.get_entities_by_handle(tag, True)
        for ent_201 in entity_set_201:
            mb.tag_set_data(neumann_tag, ent_201, [0.0,])
            bound_nodes_201 = mtu.get_bridge_adjacencies(ent_201, 0, 0)
            mb.tag_set_data(neumann_tag, bound_nodes_201, np.repeat([0.0], len(bound_nodes_201)))

for tag in physical_sets:
    tag_id = mb.tag_get_data(physical_tag, np.array([tag]), flat=True)

    if tag_id == 101:
        entity_set_101 = mb.get_entities_by_handle(tag, True)
        for ent_101 in entity_set_101:
            mb.tag_set_data(dirichlet_tag, ent_101, [0.0,])
            bound_nodes_101 = mtu.get_bridge_adjacencies(ent_101, 0, 0)
            mb.tag_set_data(dirichlet_tag, bound_nodes_101, np.repeat([0.0], len(bound_nodes_101)))
    if tag_id == 102:
        entity_set_102 = mb.get_entities_by_handle(tag, True)
        for ent_102 in entity_set_102:
            mb.tag_set_data(dirichlet_tag, ent_102, [1.0,])
            bound_nodes_102 = mtu.get_bridge_adjacencies(ent_102, 0, 0)
            mb.tag_set_data(dirichlet_tag, bound_nodes_102, np.repeat([1.0], len(bound_nodes_102)))

#print("Coords: ", mb.get_coords([3]))
#new_node_tag_dirichlet = mb.tag_get_data(dirichlet_tag, [3])

entities = mb.get_entities_by_dimension(root_set, 2)
for ent in entities:
    full_edges = mb.get_adjacencies(ent, 1, True)
    full_edge_meshset = mb.create_meshset()
    mb.add_entities(full_edge_meshset, full_edges)
    mb.tag_set_data(full_edges_tag, ent, full_edge_meshset)

#all_verts = mb.get_entities_by_dimension(root_set, 0)
#mtu.construct_aentities(all_verts)

# Permeability tensor
perm_tensor = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#K = np.dot(np.dot(rot_matrix, K), np.linalg.inv(rot_matrix))


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

def KN_dot_area(A, B, C, K):
    normal = norm_vec(A, B, C)
    square_norm = (norma(B - A))**2.0
    dot_product = np.dot(B - A, C - A)
    int_area = area(A, B, C)
    KN_dot = np.dot(np.dot(normal, K), normal)
    #print A,
    #print KN_dot == square_norm
    KN_d_a = (KN_dot/square_norm) * (dot_product/(2.0*int_area))
    #print KN_d_a, 2*int_area, dot_product, A, B, C

    return KN_d_a

def KT_dot(A, B, C, K):
    normal = norm_vec(A, B, C)
    tangente = B - A
    square_norm = (norma(B - A))**2.0
    KT_d = np.dot(np.dot(normal, K), tangente)/square_norm
    #print KT_d
    return KT_d

def phi_LPEW1(p1, px, Tk_mid, K):
    coords_p1 = mb.get_coords([p1])
    coords_px = mb.get_coords([px])

    adj_p1 = set(mb.get_adjacencies(p1, 2))
    adj_px = set(mb.get_adjacencies(px, 2))
    adj_blocks = adj_p1 & adj_px
    adj_blocks = list(adj_blocks)
    num_phi = 0
    den_phi = 0
    for a_block in adj_blocks:
        cent_block = get_centroid(a_block)
        #print coords_p1, cent_block
        num_kn_dot_area = KN_dot_area(cent_block, Tk_mid, coords_p1, K)
        #print num_kn_dot_area, cent_block
        #print num_kn_dot_area, cent_block, coords_p1
        num_kt_dot = KT_dot(Tk_mid, cent_block, coords_p1, K)
        num_phi = num_phi + num_kn_dot_area - num_kt_dot

        den_kn_dot_area = KN_dot_area(coords_p1, Tk_mid, cent_block, K)
        den_kt_dot = KT_dot(coords_p1, Tk_mid, cent_block, K)
        den_phi = den_phi + den_kn_dot_area + den_kt_dot
    phi = num_phi/den_phi
    return phi

def lambda_LPEW1(p1, entity, K):
    coords_p1 = mb.get_coords([p1])
    cent_entity = get_centroid(entity)
    adj_p1 = set(mtu.get_bridge_adjacencies(p1, 1, 0))
    adj_entity = set(mb.get_adjacencies(entity, 0))
    pxs = adj_p1 & adj_entity
    pxs = list(pxs)
    lambda_W = 0
    for px in pxs:
        coords_px = mb.get_coords([px])
        Tk_mid = mid_point([p1], [px])
        normal_ITk = norm_vec(coords_p1, Tk_mid, cent_entity)
        dot_product = np.dot(np.dot(normal_ITk, K), normal_ITk)
        modulus = norma(coords_p1 - coords_px)
        db_area = 2.0*area(coords_p1, coords_px, cent_entity)
        phi = phi_LPEW1(p1, px, Tk_mid, K)
        #print phi, Tk_mid
        parcela_1 = dot_product*modulus*phi/db_area
        #print parcela_1, cent_entity, Tk_mid
        parcela_2 = KN_dot_area(Tk_mid, cent_entity, coords_p1, K)

        parcela_3 = KT_dot(Tk_mid, cent_entity, coords_p1, K)
        #print parcela_2, cent_entity
        lambda_W = lambda_W + parcela_1 + parcela_2 + parcela_3
    return lambda_W

def csi_LPEW2(p1, px, Tk_mid, K):
    coords_p1 = mb.get_coords([p1])
    coords_px = mb.get_coords(px)

    adj_p1 = set(mb.get_adjacencies(p1, 2))
    adj_px = set(mb.get_adjacencies(px, 2))
    adj_blocks = adj_p1 & adj_px
    adj_blocks = list(adj_blocks)

    num = 0.0
    den = 0.0
    for a_block in adj_blocks:
        a_block_verts = mb.get_adjacencies(a_block, 0)
        mtu.construct_aentities(a_block_verts)
        pts_adj_p1 = set(mtu.get_bridge_adjacencies(p1, 1, 0))

        cent_block = get_centroid(a_block)
        pts_adj_block = set(mb.get_adjacencies(a_block, 0))

        pys = pts_adj_p1 & pts_adj_block
        pys = list(pys)
        py = np.extract(pys != np.array([px]), pys)
        #print p1, px, py, pys, mb.get_coords(pys), 'sets', pts_adj_block, pts_adj_p1

        py = np.asarray(py, dtype='uint64')
        coord_py = mb.get_coords(py)
        Tk_py = mid_point(py, [p1])

        normal_py_p1 = norm_vec(coords_p1, coord_py, cent_block)
        normal_Tks = - norm_vec(Tk_py, Tk_mid, cent_block)
        area_Tks_p1 = area(Tk_py, Tk_mid, coords_p1)
        #print area_Tks_p1, 'area num'
        dot_product_num = np.dot(np.dot(K, normal_Tks), normal_py_p1)
        parcela_num = dot_product_num/area_Tks_p1
        num = num + parcela_num

        normal_p1_Tk = norm_vec(coords_p1, Tk_mid, cent_block)
        normal_cent_p1 = norm_vec(cent_block, coords_p1, Tk_mid)
        area_c_p1_Tk = area(cent_block, coords_p1, Tk_mid)
        dot_product_den = np.dot(np.dot(K, normal_p1_Tk), normal_cent_p1)
        #print area_c_p1_Tk, 'area den'
        parcela_den = dot_product_den/area_c_p1_Tk
        #print dot_product_num > 0, normal_py_p1, coords_p1, coord_py, cent_block
        #print dot_product_den > 0, normal_cent_p1, cent_block, coords_p1, Tk_mid
        #print parcela_den, len(adj_blocks), dot_product, area_c_p1_Tk
        den = den + parcela_den
        #print p1, px, py, pys, num, den
    #print num, 'num'
    #print den, 'den'
    csi = num/(2.0*den)
    #print csi, 'csi', num, den
    return csi

def psi_LPEW2(p1, entity, K):
    coords_p1 = mb.get_coords([p1])
    cent_entity = get_centroid(entity)
    adj_p1 = set(mtu.get_bridge_adjacencies(p1, 1, 0))
    adj_entity = set(mb.get_adjacencies(entity, 0))
    pxs = adj_p1 & adj_entity
    pxs = np.asarray(list(pxs), dtype='uint64')
    coords_pxs_0 = mb.get_coords([pxs[0]])
    Tk_mid_0 = mid_point([p1],[pxs[0]])
    normal_p1_Tk_0 = norm_vec(coords_p1, Tk_mid_0, cent_entity)
    normal_p1_px_0 = norm_vec(coords_p1, coords_pxs_0, cent_entity)
    area_c_p1_Tk_0 = area(coords_p1, Tk_mid_0, cent_entity)
    dot_product_0 = np.dot(np.dot(K, normal_p1_Tk_0), normal_p1_px_0)
    psi_factor_0 = dot_product_0/(4.0*area_c_p1_Tk_0)
    csi_0 = csi_LPEW2(p1, [pxs[0]], Tk_mid_0, K)
    parcela_0 = csi_0*psi_factor_0

    coords_pxs_1 = mb.get_coords([pxs[1]])
    Tk_mid_1 = mid_point([p1],[pxs[1]])
    normal_p1_Tk_1 = norm_vec(coords_p1, Tk_mid_1, cent_entity)
    normal_p1_px_1 = norm_vec(coords_p1, coords_pxs_1, cent_entity)
    area_c_p1_Tk_1 = area(coords_p1, Tk_mid_1, cent_entity)
    dot_product_1 = np.dot(np.dot(K, normal_p1_Tk_1), normal_p1_px_1)
    psi_factor_1 = dot_product_1/(4.0*area_c_p1_Tk_1)
    csi_1 = csi_LPEW2(p1, [pxs[1]], Tk_mid_1, K)
    parcela_1 = csi_1*psi_factor_1

    psi = csi_0*psi_factor_0 + csi_1*psi_factor_1

    #print p1 == long(43), entity == long(4611686018427395571), len(pxs), pxs
    #print p1 == long(43), entity == long(4611686018427395571), psi_factor_0, csi_0, parcela_0, 'A'
    #print p1 == long(43), entity == long(4611686018427395571), psi_factor_1, csi_1, parcela_1, 'B'
    #print p1 == long(43), entity == long(4611686018427395571), psi, 'psi'

    return psi

def Neumann_treat_Bk(pt_ni, neumann_tag, K):
    face_adj = mtu.get_bridge_adjacencies(pt_ni, 0, 1)
    #print len(face_adj)
    coords_ni = mb.get_coords([pt_ni])
    neu_Bk = 0
    for fac in face_adj:
        try:
            neu_flow_rate = mb.tag_get_data(neumann_tag, fac)
            pts_adj_face = mtu.get_bridge_adjacencies(fac, 0, 0)
            pt_nx = np.extract(pts_adj_face != np.array([pt_ni]), pts_adj_face)
            #print pts_adj_face[0], pts_adj_face[1], pt_ni, pt_nx
            pt_nx = np.asarray(pt_nx, dtype='uint64')
            coords_nx = mb.get_coords(pt_nx)
            #print pt_nx, coords_nx, len(face_adj)
            Tk_mid = mid_point([pt_ni], pt_nx)

            csi_neu = csi_LPEW2(pt_ni, pt_nx, Tk_mid, K)
            #print csi_neu, Tk_mid, coords_ni
            #print csi_neu, coords_ni, coords_nx, Tk_mid, len(face_adj)
            norma_ITk = norma(Tk_mid - coords_ni)
            parcela_i = (1 + csi_neu)*norma_ITk*neu_flow_rate

            neu_Bk = neu_Bk + parcela_i
            #print neu_Bk
        except RuntimeError:
            continue

    return neu_Bk

def KN_ABG(A, B, G, K):
    normal_AB = norm_vec(A, B, G)
    dot_product = np.dot(np.dot(normal_AB, K), normal_AB)
    area_ABG = area(A, B, G)
    square_norm = np.dot(A - B, A - B)
    KN_abg = dot_product/(2.0*area_ABG*square_norm)
    return KN_abg

def KT_ABG(A, B, G, K):
    normal_AB = norm_vec(A, B, G)
    tan_dir_AB = B - A
    dot_product = np.dot(np.dot(normal_AB, K), tan_dir_AB)
    square_norm = np.dot(tan_dir_AB, tan_dir_AB)
    KT_abg = dot_product/square_norm
    return KT_abg

def MPFA_D:
    all_faces = mb.get_entities_by_dimension(root_set, 1)
    for face in all_faces:
        adjacent_entitites = mtu.get_bridge_adjacencies(face, 1, 2)
        if len(adjacent_entitites) == 1:
            try:
                neumann_flux = mb.tag_get_data(face, neumann_tag)

            except RuntimeError:
                face_nodes = mb.get_adjacencies(face, 0)

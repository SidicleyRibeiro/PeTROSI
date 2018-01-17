
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

#para problema com poço
'''
physical_tag = mb.tag_get_handle("MATERIAL_SET")

physical_sets = mb.get_entities_by_type_and_tag(
    0, types.MBENTITYSET, np.array((physical_tag,)), np.array((None,)))

for tag in physical_sets:
    tag_id = mb.tag_get_data(physical_tag, np.array([tag]), flat=True)

    if tag_id == 202:
        null_flow_rate_edges = mb.get_entities_by_handle(tag, True)
        #mb.tag_set_data(neumann_tag, null_flow_rate_edges, np.zeros([1, len(null_flow_rate_edges)]))
        for null_flow_rate_edge in null_flow_rate_edges:
            mb.tag_set_data(neumann_tag, null_flow_rate_edge, [0.0,])

    if tag_id == 101:
        productor = mb.get_entities_by_handle(tag, True)
        mb.tag_set_data(dirichlet_tag, productor, [0.0,])

for tag in physical_sets:
    tag_id = mb.tag_get_data(physical_tag, np.array([tag]), flat=True)

    if tag_id == 201:
        injector = mb.get_entities_by_handle(tag, True)
        print len(injector), np.asarray(injector)
        mb.tag_set_data(neumann_tag, injector, [1.0,])
        well_edges = mtu.get_bridge_adjacencies(injector, 1, 1)
        print len(well_edges), np.array(well_edges)
        mb.tag_set_data(neumann_tag, well_edges, [1.0, 1.0])

well_edges_tag = mb.tag_get_data(neumann_tag, well_edges)

print well_edges_tag, 'well_edges_tag'
'''


# In[118]:

entities = mb.get_entities_by_dimension(root_set, 2)
for ent in entities:
    full_edges = mb.get_adjacencies(ent, 1, True)
    full_edge_meshset = mb.create_meshset()
    mb.add_entities(full_edge_meshset, full_edges)
    mb.tag_set_data(full_edges_tag, ent, full_edge_meshset)

#all_verts = mb.get_entities_by_dimension(root_set, 0)
#mtu.construct_aentities(all_verts)

# Permeability tensor
K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#K = np.dot(np.dot(rot_matrix, K), np.linalg.inv(rot_matrix))



# In[120]:


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
'''
def psi_LPEW2(p1, entity, K):
    coords_p1 = mb.get_coords([p1])
    cent_entity = get_centroid(entity)
    adj_p1 = set(mtu.get_bridge_adjacencies(p1, 1, 0))
    adj_entity = set(mb.get_adjacencies(entity, 0))
    pxs = adj_p1 & adj_entity
    pxs = list(pxs)
    psi = 0
    for px in pxs:
        coords_px = mb.get_coords([px])
        Tk_mid = mid_point([p1],[px])
        normal_p1_Tk = norm_vec(coords_p1, Tk_mid, cent_entity)
        normal_p1_px = norm_vec(coords_p1, coords_px, cent_entity)
        area_c_p1_Tk = area(coords_p1, Tk_mid, cent_entity)
        dot_product = np.dot(np.dot(K, normal_p1_Tk), normal_p1_px)

        psi_factor = dot_product/(4.0*area_c_p1_Tk)
        #print dot_product, 'dot', area_c_p1_Tk, 'area'
        csi = csi_LPEW2(p1, px, Tk_mid, K)

        psi = psi + csi*psi_factor
        print p1 == long(43), entity == long(4611686018427395571), len(pxs), pxs
        #print p1 == long(43), entity == long(4611686018427395571), px == long(42), csi, psi_factor, csi*psi_factor, 'A'
        #print p1 == long(43), entity == long(4611686018427395571), px == long(44), csi, psi_factor, csi*psi_factor, 'B'
    #print p1 == long(43), entity == long(4611686018427395571), psi
    #if psi < 0:
        #psi = -psi
    return psi
'''
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




# In[121]:

def MPFA_D(pressure_tag, K):
    #root_set = mb.get_root_set()
    entities_D = mb.get_entities_by_dimension(root_set, 2)
    #print root_set, 'root_set'
    #print entities_D, 'first entities', len(entities_D)
    A = np.zeros([len(entities_D), len(entities_D)])
    B = np.zeros([len(entities_D), 1])
    for ent in entities_D:  #Itera para cada elemento do dominio da malha
        centroid = get_centroid(ent)
        ent_list = entities_D
        ent_list = list(ent_list)
        k = ent_list.index(ent)
        faces = mb.get_adjacencies(ent, 1, True)
        #print len(faces), 'faces', get_centroid(ent)
        for fac in faces: #  Itera para cada face de cada elemento
            adj_cel = np.array(mb.get_adjacencies(fac, 2))
            #print len(adj_cel), 'cells'
            pts = mb.get_adjacencies(fac, 0)

            cds = mb.get_coords(pts)
            cds = np.reshape(cds, (2, 3))
            normal_fac = norm_vec(cds[0], cds[1], centroid)
            normal_fac_op = -normal_fac
            norm_fac = norma(cds[0]-cds[1])
            orien_fac = count_wise(cds[0], cds[1], centroid)

            if len(adj_cel) == 1:  # Edge belongs to the boundary.
                square_norm = np.dot(cds[0] - cds[1], cds[0] - cds[1])
                kn_abg = KN_ABG(cds[0], cds[1], centroid, K)
                parcela_Akk = square_norm*kn_abg

                A[k,k] = A[k,k] + parcela_Akk

                for i in range(len(pts)):
                    dot_product = np.dot(centroid - cds[i-1], cds[i] - cds[i-1])
                    kn_pts = KN_ABG(cds[i], cds[i-1], centroid, K)
                    kt_pts = KT_ABG(cds[i], cds[i-1], centroid, K)
                    factor_Bk = (kn_pts*dot_product + kt_pts)

                    try: #Node belongs to a Dirichlet Boundary
                        pressure_i = mb.tag_get_data(dirichlet_tag, pts[i])
                        parcela_Bk = pressure_i*factor_Bk
                        B[k] = B[k] + parcela_Bk

                    except RuntimeError: #Node belongs to a Neumann Boundary
                        around_blocks = mb.get_adjacencies(pts[i], 2)
                        #print cds[i]
                        LPEW2 = np.zeros([len(entities_D), len(entities_D)])
                        sum_psi = 0
                        for a_block in around_blocks: #Para cada elemento ao redor do vertice

                            psi = psi_LPEW2(pts[i], a_block, K)
                            j = ent_list.index(np.asarray([a_block], dtype='uint64'))
                            LPEW2[k,j] = LPEW2[k,j] + psi
                            sum_psi = sum_psi + psi

                        LPEW2 = LPEW2*(-factor_Bk)/sum_psi
                        A = A + LPEW2
                        neu_Bk = Neumann_treat_Bk(pts[i], neumann_tag, K)
                        parcela_Bk = neu_Bk*factor_Bk/sum_psi
                        B[k] = B[k] - parcela_Bk

            else: # Edge is shared by two cells
                #print 'shared by two cells', len(adj_cel)
                num3 = np.dot(np.dot(K, normal_fac), normal_fac)
                den3 = 4.0*area(cds[0], cds[1], centroid)
                A[k,k] = A[k,k] + num3/den3

                cel_adj = np.asarray(np.extract(adj_cel != ent, adj_cel), dtype='uint64')
                j = ent_list.index(cel_adj)
                centroid2 = get_centroid(cel_adj)
                num4 = np.dot(np.dot(K, normal_fac_op), normal_fac_op)
                den4 = 4.0*area(cds[0], cds[1], centroid2)
                A[k,j] = A[k,j] - num4/den4  #Calcula termo implicito do elemento oposto em relacao a face atual na iteracao

                for i in range(len(pts)): # Para cada vertice da face
                    normal_ptop_cen = norm_vec(cds[i-1], centroid, cds[i])
                    area_cen = area(cds[i-1], centroid, cds[i])
                    normal_ptop_cenop = norm_vec(cds[i-1], centroid2, cds[i])
                    area_cenop = area(cds[i-1], centroid2, cds[i])

                    num5 = np.dot(np.dot(K, normal_fac), normal_ptop_cen)
                    num6 = np.dot(np.dot(K, normal_fac_op), normal_ptop_cenop)
                    den5 = 4.0*area_cen
                    den6 = 4.0*area_cenop
                    node_coef = (num5/den5) - (num6/den6)

                    try: #  Node belongs to Dirichlet boundary
                        dirich_press = mb.tag_get_data(dirichlet_tag, pts[i])
                        B[k] = B[k] - node_coef*dirich_press

                    except RuntimeError:
                        around_blocks = np.asarray(mtu.get_bridge_adjacencies(pts[i], 2, 2), dtype='uint64')
                        adj_pts_pts = set(mtu.get_bridge_adjacencies(pts[i], 1, 0))

                        try: #Node belongs to Neumann boundary
                            neumann_flow_rate = mb.tag_get_data(neumann_tag, pts[i])
                            neu_Bk = Neumann_treat_Bk(pts[i], neumann_tag, K)

                            LPEW2 = np.zeros([len(entities_D), len(entities_D)])
                            sum_psi = 0.0
                            for a_block in around_blocks: #Para cada elemento ao redor do vertice
                                psi = psi_LPEW2(pts[i], a_block, K)
                                j = ent_list.index(a_block)
                                LPEW2[k,j] = LPEW2[k,j] + psi
                                sum_psi = sum_psi + psi

                            LPEW2 = LPEW2*node_coef/sum_psi
                            A = A + LPEW2
                            parcela_Bk = neu_Bk*node_coef/sum_psi
                            B[k] = B[k] + parcela_Bk

                        except RuntimeError:
                            METHOD = 'LPEW2'
                            #print cds[i]
                            if METHOD == 'LPEW1':
                                LPEW1 = np.zeros([len(entities_D), len(entities_D)])
                                sum_lambda = 0
                                for a_block in around_blocks:
                                    lambda_W = lambda_LPEW1(pts[i], a_block, K)
                                    j = ent_list.index(long(a_block))
                                    LPEW1[k,j] = LPEW1[k,j] + lambda_W
                                    sum_lambda = sum_lambda + lambda_W

                                LPEW1 = LPEW1*node_coef/sum_lambda
                                A = A = LPEW1

                            if METHOD == 'LPEW2':
                                #print cds[i]
                                LPEW2 = np.zeros([len(entities_D), len(entities_D)])
                                sum_psi = 0.0
                                for a_block in around_blocks: #Para cada elemento ao redor do vertice
                                    psi = psi_LPEW2(pts[i], a_block, K)
                                    j = ent_list.index(np.asarray([a_block], dtype='uint64'))
                                    LPEW2[k,j] = LPEW2[k,j] + psi*node_coef
                                    sum_psi = sum_psi + psi

                                for a_block in around_blocks:
                                    j = ent_list.index(np.asarray([a_block], dtype='uint64'))
                                    change = LPEW2[k,j]/sum_psi
                                    LPEW2[k,j] = change
                                    A[k,j] = A[k,j] + LPEW2[k,j]

    ele_press = np.linalg.solve(A, B)
    #print ele_press, 'ele_press'
    #print entities_D, 'entities_D'
    #ele_press = np.array(ele_press, dtype = 'float64')
    mb.tag_set_data(pressure_tag, entities_D, ele_press.flatten())
    for entest in entities_D:
        coord_x = get_centroid(entest)[0]
        presss = mb.tag_get_data(pressure_tag, entest)
        print("verif: ", 1 - coord_x, presss, 1 - coord_x - presss)


    #mb.write_file('pressure_field_00001.vtk')

    return ele_press

'''
# In[122]:

def node_pressure(node, pressure_tag, K):
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

pressure_field = MPFA_D(pressure_tag, K)
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

#Takes all boundary vertices and set boundary data to those in between


'''

# In[ ]:

pressure_field = MPFA_D(pressure_tag, K)
# vertices = mb.get_entities_by_dimension(root_set, 0)
# for vertice in vertices:
#     coord_vert = mb.get_coords([vertice])
#     node_p = node_pressure(vertice, pressure_tag, K)
#     #print node_p, coord_vert
#     mb.tag_set_data(node_pressure_tag, vertice, node_p)
mb.write_file('pressure_previous_code.vtk')
'''
'''

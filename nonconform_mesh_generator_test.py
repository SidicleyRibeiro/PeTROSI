
# coding: utf-8

# In[490]:


import random

import numpy as np
from pymoab import core
from pymoab import types
from pymoab import topo_util



def norma(vector):
    vector = np.array(vector)
    dot_product = np.dot(vector, vector)
    mag = np.sqrt(dot_product)
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

def counterclock_sort(coords):
    inner_coord = sum(coords)/(len(coords))
    vectors = np.array(
        [crd_node - inner_coord for crd_node in coords])

    directions = np.zeros(len(vectors))
    for j in range(len(vectors)):
        direction = ang_vectors(vectors[j], [1, 0, 0])
        if vectors[j, 1] <= 0:
            directions[j] = directions[j] + 2.0*np.pi - direction
        else:
            directions[j] = directions[j] + direction
    indices = np.argsort(directions)
    return indices

def crazy_mesh():

    mb = core.Core()
    mtu = topo_util.MeshTopoUtil(mb)
    root_set = mb.get_root_set()


    # In[491]:

    # Parametros

    num_layers = 15

    random_weight_x = 0.0
    random_weight_y = 0.3

    max_step = 1

    num_points_y = 15

    left_dirichlet_value = 1.0
    right_dirichlet_value = 0.0


    # In[492]:

    random.uniform(-1, 1)


    # In[493]:

    # Create coords

    layers = []

    for layer_id in range(num_layers):
        layers.append(
            np.array([
                    [float(layer_id + random.uniform(-1, 1)*random_weight_x) / (num_layers - 1),
                     float(y_val + random.uniform(-1, 1)*random_weight_y) / (num_points_y - 1),
                     0.0] for y_val in range(num_points_y)], dtype='float64'))


    # Make it so that the boundary coords are within the bounding box (0,0) - (1,1)
    for layer in layers:
        layer[0][1] = 0.0
        layer[-1][1] = 1.0

    layers = np.array(layers, dtype='float64')
    # print(layers)


    # In[494]:

    # Create verts
    verts = []

    for layer_id in range(num_layers):
        coords = layers[layer_id]
        verts.append(mb.create_vertices(coords.flatten()))

    verts = np.array(verts, dtype='uint64')
    # print(verts)


    # In[495]:

    dirichlet_tag = mb.tag_get_handle(
        "Dirichlet", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

    neumann_tag = mb.tag_get_handle(
        "Neumann", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

    pressure_tag = mb.tag_get_handle(
        "Pressure", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

    perm_tag = mb.tag_get_handle(
        "Permeability", 9, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

    neumann_entities_tag = mb.tag_get_handle("Neumann_entities", 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)

    dirichlet_entities_tag = mb.tag_get_handle("Dirichlet_entities", 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)


    # In[496]:

    # Create elems

    for layer_id in range(num_layers-1):
        y1_id = 0
        y2_id = 0

        layer1 = verts[layer_id]
        layer2 = verts[layer_id + 1]

        while (y1_id < num_points_y-1) and (y2_id < num_points_y-1):
            y1_inc = y1_id + random.randint(1,max_step)
            y2_inc = y2_id + random.randint(1,max_step)
            if y1_inc >= num_points_y:
                y1_inc = num_points_y - 1
            if y2_inc >= num_points_y:
                y2_inc = num_points_y - 1

            poly = [layer1[y1_id]]

            for y2_id_aux in range(y2_id, y2_inc + 1):
                poly.append(layer2[y2_id_aux])
            for y1_id_aux in range(y1_inc, y1_id, -1):
                poly.append(layer1[y1_id_aux])

            elem = mb.create_element(types.MBPOLYGON, poly)
            mb.tag_set_data(pressure_tag, elem, random.random())

            y1_id = y1_inc
            y2_id = y2_inc

        if y1_id != num_points_y - 1:
            poly = [layer1[y1_id], layer2[y2_id]]
            for y1_id_aux in range(num_points_y - 1, y1_id, -1):
                poly.append(layer1[y1_id_aux])
            mb.create_element(types.MBPOLYGON, poly)
        elif y2_id != num_points_y - 1:
            poly = [layer1[y1_id], layer2[y2_id]]
            for y1_id_aux in range(y2_id + 1, num_points_y, 1):
                poly.append(layer2[y1_id_aux])
            mb.create_element(types.MBPOLYGON, poly)


    # In[497]:

    all_verts = mb.get_entities_by_dimension(root_set, 0)
    mtu.construct_aentities(all_verts)


    # In[498]:


    print("VOL ANTES", len(mb.get_entities_by_dimension(root_set, 2)))
    print("VERT ANTES", len(mb.get_entities_by_dimension(root_set, 0)))


    # In[499]:

    for a_vert in all_verts:
        vols_adj = mtu.get_bridge_adjacencies(a_vert, 0, 2)
        faces_adj = mtu.get_bridge_adjacencies(a_vert, 0, 1)

        if len(vols_adj) == 2 and len(faces_adj) == 2:
            # print(len(vols_adj), len(faces_adj))
            for a_vol_adj in vols_adj:
                nodes = set(mb.get_adjacencies(a_vol_adj, 0))

                nodes.remove(a_vert)
                nodes = list(nodes)
                coords = mb.get_coords(nodes).reshape([len(nodes), 3])
                nodes_indices = counterclock_sort(coords)
                coords = coords[nodes_indices]
                nodes = np.asarray(nodes, dtype='uint64')
                # print(nodes)
                nodes = nodes[nodes_indices]
                elem = mb.create_element(types.MBPOLYGON, nodes)
                mb.delete_entities([a_vol_adj])
                # print ('vert', [a_vert])
                # mb.delete_entities([a_vert])

    print("VOL DEPOIS", len(mb.get_entities_by_dimension(root_set, 2)))
    print("VERT DEPOIS", len(mb.get_entities_by_dimension(root_set, 0)))



    # In[455]:

    # Dirichlet
    # Nodes

    mb.tag_set_data(dirichlet_tag, verts[0], np.repeat(left_dirichlet_value, num_points_y))
    mb.tag_set_data(dirichlet_tag, verts[num_layers-1], np.repeat(right_dirichlet_value, num_points_y))

    dirichlet_set = mb.create_meshset()
    left_right_nodes = set(verts[0]) | set(verts[num_layers-1])
    mb.add_entities(dirichlet_set, left_right_nodes)

    # Faces

    faces = mb.get_entities_by_dimension(root_set, 1)

    left_nodes = set(verts[0])
    right_nodes = set(verts[num_layers-1])

    for face in faces:
        adj_vols = mtu.get_bridge_adjacencies(face, 1, 2)

        if len(adj_vols) == 1:
            adj_nodes = mtu.get_bridge_adjacencies(face, 0, 0)
            if set(adj_nodes).intersection(left_nodes):
                mb.tag_set_data(dirichlet_tag, face, left_dirichlet_value)
                mb.add_entities(dirichlet_set, [face])

            if set(adj_nodes).intersection(right_nodes):
                mb.tag_set_data(dirichlet_tag, face, right_dirichlet_value)
                mb.add_entities(dirichlet_set, [face])

    # dirichlet_set = np.asarray([dirichlet_set], dtype='uint64')
    # print(dirichlet_set, face)
    mb.tag_set_data(dirichlet_entities_tag, root_set, dirichlet_set)


    # In[456]:

    # Neumann
    neumann_set = mb.create_meshset()
    # Nodes
    for layer_id in range(1, num_layers-1):
        mb.tag_set_data(neumann_tag, [verts[layer_id, 0], verts[layer_id, -1]], [0.0, 0.0])
        mb.add_entities(neumann_set, [verts[layer_id, 0]])
        mb.add_entities(neumann_set, [verts[layer_id, -1]])

    # Faces
    faces = mb.get_entities_by_dimension(root_set, 1)

    top_bottom_nodes = set(verts[1:-1,0]) | set(verts[1:-1,-1])

    for face in faces:
        adj_vols = mtu.get_bridge_adjacencies(face, 1, 2)

        if len(adj_vols) == 1:
            adj_nodes = mtu.get_bridge_adjacencies(face, 0, 0)
            if set(adj_nodes).intersection(top_bottom_nodes):
                mb.tag_set_data(neumann_tag, face, 0.0)
                mb.add_entities(neumann_set, [face])

    mb.tag_set_data(neumann_entities_tag, root_set, neumann_set)


    # In[457]:

    # Permeability

    K_1 = np.array([1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0])

    K_2 = np.array([2.0, 0.0, 0.0,
                    0.0, 2.0, 0.0,
                    0.0, 0.0, 2.0])

    all_volumes = mb.get_entities_by_dimension(0, 2)

    for vol in all_volumes:
        cent = mtu.get_average_position([vol])
        if cent[0] < 0.5:
            mb.tag_set_data(perm_tag, vol, K_2)
        else:
            mb.tag_set_data(perm_tag, vol, K_1)

    mb.write_file('mesh_nonconform_test.vtk')
    print("FUNCIONOU")
    return mb
mb = crazy_mesh()

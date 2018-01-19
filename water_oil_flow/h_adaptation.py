
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




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

import pymoab
import numpy as np

class Preprocessor(pymoab):


    mb = pymoab.core.Core()
    root_set = mb.get_root_set()
    types = pymoab.types
    mtu = pymoab.topo_util.MeshTopoUtil(mb)

    pressure_tag = mb.tag_get_handle(
        "pressure", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

    dirichlet_tag = mb.tag_get_handle(
        "dirichlet", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

    neumann_tag = mb.tag_get_handle(
        "neumann", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

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

    all_b_conditions = []
    def __init__(self, mesh_file, b_condition):
        self.__class__.mb.load_file(mesh_file)
        self.b_conditions = b_condition
        self.__class__.all_b_conditions.append(b_condition)

    def material_set_init(self):
        self.physical_tag = self.__class__.mb.tag_get_handle("MATERIAL_SET")
        self.physical_sets = self.__class__.mb.get_entities_by_type_and_tag(
            0, self.__class__.types.MBENTITYSET, np.array(
            (self.physical_tag,)), np.array((None,)))

    def b_condition_values(self, b_condition_type):
        ids_values = self.b_conditions[b_condition_type]
        ids = list(ids_values.keys())

        bound_nodes = set()
        for id_ in ids:
            for tag in self.physical_sets:
                tag_id = self.__class__.mb.tag_get_data(
                        self.physical_tag, np.array([tag]), flat=True)
                entity_set = self.__class__.mb.get_entities_by_handle(tag, True)

                if tag_id == id_:
                    for ent in entity_set:
                        nodes = self.__class__.mtu.get_bridge_adjacencies(ent, 0, 0)
                        bound_nodes = bound_nodes | set(nodes)

                        if b_condition_type == "dirichlet":
                            self.__class__.mb.tag_set_data(dirichlet_tag, ent, [ids_values[id_]])
                            self.__class__.mb.tag_set_data(
                                    self.__class__.dirichlet_tag, nodes, np.repeat([ids_values[id_]], len(nodes)))

                        if b_condition_type == "neumann":
                            self.__class__.mb.tag_set_data(neumann_tag, ent, [ids_values[id_]])
                            self.__class__.mb.tag_set_data(
                                    self.__class__.neumann_tag, nodes, np.repeat([ids_values[id_]], len(nodes)))

        return list(bound_nodes)

    def well_condition(self, coords, radius):
        pass

    def get_centroid(self, entity):

        # verts = mb.get_adjacencies(entity, 0)
        # coords = mb.get_coords(verts).reshape(len(verts), 3)
        # # print("Coords test: ", coords)
        # centroide = sum(coords)/(len(verts))
        verts = self.__class__.mb.get_adjacencies(entity, 0)
        coords = np.array([self.__class__.mb.get_coords([vert]) for vert in verts])

        qtd_pts = len(verts)
        #print qtd_pts, 'qtd_pts'
        coords = np.reshape(coords, (qtd_pts, 3))
        pseudo_cent = sum(coords)/qtd_pts

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

        self.centroide = wgtd_cent/total_area
        return self.centroide

    def permeability(self, block_coords):
        self.perm_tensor = [1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0]
        return self.perm_tensor

    def all_hanging_nodes_full_edges(self):
        entities = self.__class__.mb.get_entities_by_dimension(self.__class__.root_set, 2)
        for ent in entities:

            full_edges = self.__class__.mb.get_adjacencies(ent, 1, True)
            full_edge_meshset = self.__class__.mb.create_meshset()
            mb.add_entities(self.__class__.full_edge_meshset, full_edges)
            mb.tag_set_data(self.__class__.full_edges_tag, ent, full_edge_meshset)
            mb.tag_set_data(self.__class__.perm_tag, ent, self.permeability(self.get_centroid(ent)))

    def mesh_data(self):
        return self.__class__.mb

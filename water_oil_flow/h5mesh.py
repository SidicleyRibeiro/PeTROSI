import h5py
import meshio


class MeshHandler:

    def __init__(self, mesh_file = None):
        points, cells, point_data, cell_data, field_data = \
            meshio.read(mesh_file)
        self.mesh_data = h5py.File("new_mesh.hdf5", 'w')

        self.all_entities = self.mesh_data.create_group('all_entities')
        node_data = self.all_entities.create_group('node_data')
        line_data = self.all_entities.create_group('line_data')
        cell_2D_data = self.all_entities.create_group('cell_2D_data')
        cell_3D_data = self.all_entities.create_group('cell_3D_data')

        field_h5dir = self.mesh_data.create_group('fields')

    def store_line_connectivities(self):
        lines_connectivities = self.cells['line']



    def get_entities_by_dimension(self, dimension):
        if dimension == 0:
            return range(len(self.points))
        if dimension == 1:

            pass
        if dimension == 2:
            pass
        if dimension == 3:
            pass

    def get_adjacencies(self, entity):






    def get_entities_by_dimension(self, )

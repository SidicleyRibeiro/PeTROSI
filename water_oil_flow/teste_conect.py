from pymoab import core
from pymoab import types
from pymoab import topo_util
import numpy as np

# Inicializa instancia de core (que contem maioria das funcoes ex. ler malha,
# criar elementos, pegar adjacencias, etc)
mb = core.Core()

# Alguma coisa importante, sei que usa quando for pegar todas as entidades de
# uma dada dimensao, ex. todos_quadrados = mb.get_entities_by_dimension(root_set, 2)
# Eh como se esse root_set identificasse a malha total (malha raiz)
root_set = mb.get_root_set()


mtu = topo_util.MeshTopoUtil(mb)

# Cria as coordenadas
p0 = np.array([0.0, 0.0, 0.0])
p1 = np.array([1.0, 0.0, 0.0])
p2 = np.array([2.0, 0.0, 0.0])
p3 = np.array([2.0, 1.0, 0.0])
p4 = np.array([1.0, 1.0, 0.0])
p5 = np.array([0.0, 1.0, 0.0])
p6 = np.array([1.0, 2.0, 0.0])
p7 = np.array([0.0, 2.0, 0.0])
p8 = np.array([2.5, 1.5, 0.0])
p9 = np.array([2.0, 2.5, 0.0])
p10 = np.array([2.5, 3.0, 0.0])
p11 = np.array([3.0, 3.5, 0.0])
p12 = np.array([3.5, 2.5, 0.0])
p13 = np.array([3.25, 1.75, 0.0])
p14 = np.array([2.75, 1.25, 0.0])

# Pega essas coordenadas acima e cria os nos da malha
vt = mb.create_vertices(np.array([p0, p1, p2, p3, p4,
                           p5, p6, p7, p8, p9,
                           p10, p11, p12, p13, p14]))


# Cria os elementos 2D a partir dos nos criados acima, observe que cada um tem
# uma certa quantidade de nos que delimita o poligono e que eles estao ordenados
# no sentido antihorario
quad_1 = mb.create_element(types.MBPOLYGON, np.array([vt[0], vt[1], vt[4], vt[5]], dtype='uint64'))
quad_2 = mb.create_element(types.MBPOLYGON, np.array([vt[1], vt[2], vt[3], vt[4]], dtype='uint64'))

tri_1 = mb.create_element(types.MBPOLYGON, np.array([vt[5], vt[4], vt[6]], dtype='uint64'))
tri_2 = mb.create_element(types.MBPOLYGON, np.array([vt[5], vt[6], vt[7]], dtype='uint64'))

pent = mb.create_element(types.MBPOLYGON, np.array([vt[4], vt[3], vt[8], vt[9], vt[6]], dtype='uint64'))

sept = mb.create_element(types.MBPOLYGON, np.array([vt[8], vt[14], vt[13],
                                                    vt[12], vt[11], vt[10], vt[9]], dtype='uint64'))

# Escreve a malha para o arquivo h5m, o qual a gente le utilizando o h5py
mb.write_file('teste_conect.h5m')

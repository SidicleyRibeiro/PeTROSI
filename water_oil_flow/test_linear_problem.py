import mesh_preprocessor
import pressure_solver
import numpy as np
# import matplotlib.pyplot as plt
# Defines mesh file
mesh_file = "geometry_horizontal_layers.msh"

# Instanciates "Mesh_Manager" with mesh file and boundary conditions

mesh_data = mesh_preprocessor.Mesh_Manager(mesh_file)

# Sets mesh information with correspondant values to tags defined on mesh file

K_1 = np.array([2.0, 0.0, 0.0,
                0.0, 2.0, 0.0,
                0.0, 0.0, 2.0])

K_2 = np.array([1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0])

mesh_info = {"permeability":{1:K_1, 2:K_2}, "dirichlet":{101:0.0, 102:1.0}, "neumann": {201:0.0}}
mesh_data.mesh_problem_info(mesh_info)

# Appends more data, such as hanging nodes and conform edges for adaptation process,
# as well as permeability tensor, to each 2D element on mesh
mesh_data.all_hanging_nodes_full_edges()

# Calculates pressure field with arbitrary method (eg. MPFA-D, TPFA, MPFA-O)
pressure_field = pressure_solver.MPFA_D(mesh_data)

#Calculates node pressures
node_pressures = pressure_solver.get_nodes_pressures(mesh_data)

#Calculates pressure gradient field
pressure_gradient = pressure_solver.pressure_grad(mesh_data)
# print("Pressure gradient: ", pressure_gradient)
#Saves mesh data to a file
mesh_data.mb.write_file("test_linear_problem.vtk")

print("NUMERO ELEMENTOS: ", len(mesh_data.all_volumes))

# Testing node pressures to linear problem
all_nodes = mesh_data.all_nodes
file_coord_node = open('for_node_coord_plot.txt', 'w')
file_pressure_node = open('for_node_pressure_plot.txt', 'w')
for node in all_nodes:
    coord = mesh_data.mb.get_coords([node])
    file_coord_node.write('{0}\n'.format(coord[0]))
    file_pressure_node.write('{0}\n'.format(node_pressures[node][0][0]))
file_coord_node.close()
file_pressure_node.close()

all_volumes = mesh_data.mb.get_entities_by_dimension(mesh_data.root_set, 2)
file_coord_elem = open('for_element_coord_plot.txt', 'w')
file_pressure_elem = open('for_element_pressure_plot.txt', 'w')
for i in range(len(all_volumes)):
    coord_x = mesh_data.get_centroid(all_volumes[i])[0]
    file_coord_elem.write('{0}\n'.format(coord_x))
    file_pressure_elem.write('{0}\n'.format(pressure_field[i][0]))
file_coord_elem.close()
file_pressure_elem.close()

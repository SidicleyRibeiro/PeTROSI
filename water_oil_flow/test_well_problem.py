import mesh_preprocessor
import pressure_solver
import numpy as np
# Defines mesh file
mesh_file = "geometry_well_test.msh"

# Sets boundary conditions with correspondant values to tags defined on mesh file
boundary_conditions = {"neumann": {201:0.0}}

# Sets well(s) conditions
wells_infos = [{"Flow_Rate_Well": np.array([0.0, 0.0, 0.0])},
                {"Pressure_Well": np.array([1.0, 1.0, 0.0])}]
wells_src_terms = [1.0, 1.0]

# Instanciates "Mesh_Manager" with mesh file and boundary conditions
mesh_data = mesh_preprocessor.Mesh_Manager(mesh_file, boundary_conditions)
# Appends boundary condition values to mesh data
mesh_data.bound_condition_values("neumann")
#mesh_data.bound_condition_values("dirichlet")

# Appends well sorce term to mesh volumes
mesh_data.well_condition(wells_infos, wells_src_terms)

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
mesh_data.mb.write_file("test_well_problem.vtk")

# Testing node pressures to linear problem
all_nodes = mesh_data.all_nodes
for node in all_nodes:
    coords_node = mesh_data.mb.get_coords([node])
    print("NODE Val: ", coords_node, node_pressures[node])

# print(len(mesh_data.all_nodes))
# mesh_data.mb.create_vertices(np.array([0.5, 0.5, 0.0]))
# print("Teste quant: ", len(mesh_data.all_nodes), len(mesh_data.mb.get_entities_by_dimension(mesh_data.root_set, 0)))

# Testing centroid pressures to linear problem
all_volumes = mesh_data.mb.get_entities_by_dimension(mesh_data.root_set, 2)
for i in range(len(all_volumes)):
    coords_vol = mesh_data.get_centroid(all_volumes[i])
    print("Val: ", coords_vol, pressure_field[i])


print("NUMERO ELEMENTOS: ", len(mesh_data.all_volumes))

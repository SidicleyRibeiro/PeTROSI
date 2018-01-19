import mesh_preprocessor
import pressure_solver

# Defines mesh file
mesh_file = "geometry_test.msh"

# Sets boundary conditions with correspondant values to tags defined on mesh file
boundary_conditions = {"dirichlet":{101:0.0, 102:1.0}, "neumann": {201:0.0}}

# Instaciates "Preprocessor" with mesh file and boundary conditions
mesh_data = mesh_preprocessor.Preprocessor(mesh_file, boundary_conditions)

# Appends boundary condition values to mesh data
mesh_data.bound_condition_values("neumann")
mesh_data.bound_condition_values("dirichlet")

# Appends more data, such as hanging nodes and conform edges for adaptation process,
# as well as permeability tensor, to each 2D element on mesh
mesh_data.all_hanging_nodes_full_edges()

# Calculates pressure field with arbitrary method (eg. MPFA-D, TPFA, MPFA-O)
pressure_field = pressure_solver.MPFA_D(mesh_data)

all_volumes = mesh_data.mb.get_entities_by_dimension(mesh_data.root_set, 2)
for i in range(len(all_volumes)):
    coord_x = mesh_data.get_centroid(all_volumes[i])[0]
    # print("test: ", 1.0 - coord_x == pressures[i])
    print("Val: ", 1.0 - coord_x, pressure_field[i], (
            1.0 - coord_x) - pressure_field[i])#, get_centroid(all_volumes[i]))

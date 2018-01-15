

# Implementation of squemes to calculate multiphase flow in porous media

This simple code in Python aims to calculate the pressure and saturation fields of multiphase flow in porous media, 
along with a mesh adaptation of type h. 

To do so it uses a given mesh with a given set of boundary conditions (eg. dirichlet, neumann and source-terms from placed
wells).
  
So far, the code only calculates the pressure field of a given problem in 2D, using the formulation derived by Gao and Wu,
hence, it only deals with monophase flow. 

# To-do list:

- Add wells;
- Fix adaptation so it can adapt a mesh, with respect to a given error tolerance, more than once,
until the topology offers the error tolerance needed; 
- Add better error estimator to adaptation process;
- Implement multiphase squemes (explicit and implicit);

After these features, it's made a 1st version of a 2D flow and next...

- Make everything 3D (BIG CHALLENGE!).

The 2nd version will be performed with the @PADMEC/Elliptic package, which is about to get its 1st version.

// Gmsh project created on Tue Feb 27 09:44:49 2018
Point(1) = {0, 0, 0, 1.0};
Point(2) = {1, 0, 0, 1.0};
Point(3) = {1, 1, 0, 1.0};
Point(4) = {0, 1, 0, 1.0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line Loop(5) = {4, 1, 2, 3};
Plane Surface(6) = {5};

Physical Point(102) = {4, 1};
Physical Point(101) = {3, 2};

Physical Line(102) = {4};
Physical Line(101) = {2};
Physical Line(201) = {3, 1};

Physical Surface(1) = {6};

Transfinite Line{1, 2} = 4;
Transfinite Line{3, 4} = 4;
Transfinite Surface{6} = {1, 2, 3, 4};
Recombine Surface {6};

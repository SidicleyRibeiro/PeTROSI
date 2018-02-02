Point(1) = {0, 1, 0, 1};
Point(2) = {0, 0.5, 0, 1};
Point(3) = {0, 0, 0, 1};
Point(4) = {1, 0, 0, 1};
Point(5) = {1, 0.5, 0, 1};
Point(6) = {1, 1, 0, 1};

Line(1) = {6, 1};
Line(2) = {1, 2};
Line(3) = {2, 5};
Line(4) = {5, 6};

Line(5) = {2, 3};
Line(6) = {3, 4};
Line(7) = {4, 5};

Line Loop(8) = {1, 2, 3, 4};
Plane Surface(9) = {8};

Line Loop(10) = {-3, 5, 6, 7};
Plane Surface(11) = {10};

Transfinite Line {1} = 16;
Transfinite Line {2, 4} = 8;
//Transfinite Surface {9} = {1, 2, 5, 6};
Recombine Surface {9};

Transfinite Line {3, 6} = 16;
Transfinite Line {5, 7} = 8;
//Transfinite Surface {11} = {2, 3, 4, 5};
Recombine Surface {11};

Physical Point(101) = {4, 5, 6};
Physical Point(102) = {1, 2, 3};

Physical Line(201) = {1, 6};
Physical Line(101) = {4, 7};
Physical Line(102) = {2, 5};

Physical Surface(1) = {9};
Physical Surface(2) = {11};

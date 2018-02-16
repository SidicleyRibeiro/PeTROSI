Point(1) = {0, 0, 0, 1};
Point(2) = {0.5, 0, 0, 1};
Point(3) = {1, 0, 0, 1};
Point(4) = {1, 1, 0, 1};
Point(5) = {0.5, 1, 0, 1};
Point(6) = {0, 1, 0, 1};

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

Transfinite Line {1} = 4;
Transfinite Line {2, 4} = 3;
//Transfinite Surface {9} = {1, 2, 5, 6};
Recombine Surface {9};

Transfinite Line {3, 6} = 4;
Transfinite Line {5, 7} = 3;
//Transfinite Surface {11} = {2, 3, 4, 5};
Recombine Surface {11};

Physical Point(101) = {3, 4};
Physical Point(102) = {1, 6};

Physical Line(201) = {2, 5, 7, 4};
Physical Line(101) = {6};
Physical Line(102) = {1};

Physical Surface(1) = {9};
Physical Surface(2) = {11};

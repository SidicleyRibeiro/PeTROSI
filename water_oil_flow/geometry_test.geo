Point(1) = {0, 0, 0, 1};
Point(2) = {1, 0, 0, 1};
Point(3) = {1, 1, 0, 1};
Point(4) = {0, 1, 0, 1};
Line(1) = {4, 1};
Line(2) = {1, 2};
Line(3) = {2, 3};
Line(4) = {3, 4};
Line Loop(5) = {4, 1, 2, 3};
Plane Surface(6) = {5};

Transfinite Line {1, 2, 3} = 4;
Transfinite Line {4} = 3;
//Transfinite Surface {6} = {1, 2, 3, 4};
Recombine Surface {6};

Physical Point(101) = {2, 3};
Physical Point(102) = {1, 4};

Physical Line(201) = {4, 2};
Physical Line(101) = {3};
Physical Line(102) = {1};


Physical Surface(1) = {6};
//Transfinite Line {4} = 10 Using Progression 1;
//Transfinite Line {1, 2, 3} = 10 Using Progression 1;
Coherence;

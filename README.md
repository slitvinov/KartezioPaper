# Intro

```
Each individual is represented as a 2D array, where i0, i1, etc., are
possible values rather than dimensions. I guess, this structure is
from Cartesian Genetic Programming (CGP):

gen[i0, i1, i2, n0 ... n30, o0, o1][node_type, arity0, arity1, p0, p1]

The indices i0=0, i1=1, and i2=2 refer to the inputs, which in this
case are the three channels of an RGB image. The nodes, n0 to n30,
represent different operations (like sqrt, fill_holes, intrange;
hidden in package nodes.py), with 42 possible node types, and 30 is
the maximum number of nodes used. Each node has a type, and can take
up to two inputs (arity0, arity1), produces one output, and can have
up to two parameters (p0, p1). The parameters are discrete values
between 0 and 255. This table defines a graph, allowing the
calculation of o0 and o1 starting from i0, i1, and i2. The cost
function computes the segmentation score based on o0, o1, and the
known reference segmentation.
```

# Run

```
python diff.py
for i in diff.*.gv; do dot $i -T png -o ${i/.gv/.png}; done
```

```
dot main.00000160.gv -o main.png -T png
```

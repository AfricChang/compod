# -*- coding: utf-8 -*-
"""
Feasibility test for the sage.all shim on COMPOD's BSP hot path.

Run (native Windows, Python 3.10-3.13)::

    pip install pycddlib gmpy2
    set PYTHONPATH=<repo>\\windows_compat
    python windows_compat/test_compat.py

No numpy / scipy / SageMath / compiler needed for this test.
It exercises exactly the API sequence in pycompod/complex.py:
  _init_bounding_box -> _inequalities -> Polyhedron(ieqs) ->
  hspace.intersection(cell) -> dim() -> volume()/bounding_box()/center() ->
  face.adjacency_matrix() -> render_solid().obj_repr()
"""

import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sage.all import QQ, Polyhedron, vector, Matrix, arctan2  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print("  [%s] %s%s" % ("OK" if cond else "!!", name,
                           ("  -> " + str(detail)) if detail else ""))


def box(lo=(0, 0, 0), hi=(2, 2, 2)):
    """8 corners, exactly like complex.py:_init_bounding_box."""
    a, b = lo, hi
    return [[a[0], a[1], a[2]], [a[0], b[1], a[2]],
            [a[0], a[1], b[2]], [a[0], b[1], b[2]],
            [b[0], b[1], b[2]], [b[0], a[1], b[2]],
            [b[0], b[1], a[2]], [b[0], a[1], a[2]]]


def halfspaces(plane):
    """Reproduce complex.py:183 _inequalities(): [d,a,b,c] and its negation."""
    p = [QQ(plane[-1]), QQ(plane[0]), QQ(plane[1]), QQ(plane[2])]
    return p, [QQ(-t) for t in p]


def sorted_vertex_indices(adj):
    """Copy of complex.py:418 _sorted_vertex_indices (polygon ring walk)."""
    pointer, out = 0, [0]
    for _ in range(len(adj[0]) - 1):
        connected = [i for i, v in enumerate(adj[pointer]) if v]
        if connected[0] not in out:
            pointer = connected[0]
        else:
            pointer = connected[1]
        out.append(pointer)
    return out


print("=" * 72)
print("1. QQ exactness and vector/matrix helpers")
check("QQ(0.1) is exact binary rational",
      QQ(0.1) == Fraction(3602879701896397, 36028797018963968), QQ(0.1))
v = vector([QQ(1), QQ(2), QQ(3)])
check("vector arithmetic", list(v + vector([1, 1, 1])) == [2, 3, 4])
check("vector.cross_product", list(v.cross_product([1, 0, 0])) == [0, 3, -2])
m = Matrix(QQ, [[1, 2], [3, 4]])
check("Matrix.transpose()", list(m.transpose()[0]) == [1, 3])
check("arctan2", abs(float(arctan2(1, 1)) - 0.7853981633974483) < 1e-12)

print("=" * 72)
print("2. Polyhedron(vertices=..., base_ring=QQ) -> V/H conversion")
bb = Polyhedron(vertices=box(), base_ring=QQ)
check("dim() == 3", bb.dim() == 3, bb.dim())
check("vertices_list() has 8 exact corners", len(bb.vertices_list()) == 8,
      bb.vertices_list()[0])
check("bounding_box()", bb.bounding_box() == ((0, 0, 0), (2, 2, 2)),
      bb.bounding_box())
check("center() == (1,1,1)", all(t == 1 for t in bb.center()), bb.center())
check("volume() == 8 (exact Fraction)", bb.volume() == 8, bb.volume())
check("inequalities_list() gives 6 facets", len(bb.inequalities_list()) == 6,
      bb.inequalities_list())
adj = bb.adjacency_matrix()
check("adjacency_matrix() 8x8, degree 3", len(adj) == 8 and
      all(sum(r) == 3 for r in adj), [sum(r) for r in adj])

print("=" * 72)
print("3. BSP split: Polyhedron(ieqs) + intersection() + dim()/volume()")
# x = 1  ->  plane params (a,b,c,d) = (1,0,0,-1)
pos, neg = halfspaces((1, 0, 0, -1))
check("_inequalities format [d,a,b,c]", list(pos) == [-1, 1, 0, 0], pos)
hp = Polyhedron(ieqs=[pos])
hn = Polyhedron(ieqs=[neg])
c_left = hp.intersection(bb)    # x <= 1
c_right = hn.intersection(bb)   # x >= 1
check("left child dim() == 3", c_left.dim() == 3, c_left.dim())
check("right child dim() == 3", c_right.dim() == 3, c_right.dim())
check("left volume == 4", c_left.volume() == 4, c_left.volume())
check("right volume == 4", c_right.volume() == 4, c_right.volume())
check("volume conserved after split", c_left.volume() + c_right.volume() == bb.volume())
check("children meet exactly at the split plane x=1",
      c_left.bounding_box()[0][0] == c_right.bounding_box()[1][0] == 1,
      (c_left.bounding_box(), c_right.bounding_box()))

print("=" * 72)
print("4. Interface facet (2D polyhedron) -> adjacency / ordered ring")
face = c_left.intersection(c_right)
check("facet dim() == 2", face.dim() == 2, face.dim())
check("facet volume() == 0", face.volume() == 0, face.volume())
check("facet has 4 vertices", len(face.vertices_list()) == 4,
      face.vertices_list())
fadj = face.adjacency_matrix()
check("facet adjacency degree 2", all(sum(r) == 2 for r in fadj),
      [sum(r) for r in fadj])
ring = sorted_vertex_indices(fadj)
check("ring walk visits all 4 vertices once", sorted(ring) == [0, 1, 2, 3], ring)
check("ring is a closed cycle", fadj[ring[-1]][ring[0]] == 1)

print("=" * 72)
print("5. Degenerate cut (plane grazing the box) must NOT create a 3D child")
gpos, gneg = halfspaces((1, 0, 0, -2))  # x = 2 grazes the box boundary
g_graze = Polyhedron(ieqs=[gpos]).intersection(bb)  # x >= 2 -> only a facet
g_full = Polyhedron(ieqs=[gneg]).intersection(bb)   # x <= 2 -> the whole box
check("grazing side dim() == 2", g_graze.dim() == 2, g_graze.dim())
check("grazing side volume() == 0", g_graze.volume() == 0, g_graze.volume())
check("full side still dim 3 / volume 8",
      g_full.dim() == 3 and g_full.volume() == 8, (g_full.dim(), g_full.volume()))
# this is exactly the guard at complex.py:3336: only split when BOTH dim()==3
check("BSP guard rejects the degenerate split",
      not (g_graze.dim() == 3 and g_full.dim() == 3))

print("=" * 72)
print("6. Oblique cut keeps exact volume conservation")
opos, oneg = halfspaces((1, 1, 1, -3))  # x+y+z = 3
o1 = Polyhedron(ieqs=[opos]).intersection(bb)
o2 = Polyhedron(ieqs=[oneg]).intersection(bb)
check("oblique children dim 3", o1.dim() == 3 and o2.dim() == 3,
      (o1.dim(), o2.dim()))
check("oblique volumes sum to 8 (exact)", o1.volume() + o2.volume() == 8,
      (o1.volume(), o2.volume()))

print("=" * 72)
print("7. Incremental BSP (complex.py:3316 loop) over several planes")
cells = [bb]
planes = [(1, 0, 0, -Fraction(1, 2)), (0, 1, 0, -1),
          (0, 0, 1, -Fraction(3, 2)), (1, 1, 0, -2)]
for pl in planes:
    p, n = halfspaces(pl)
    hp2, hn2 = Polyhedron(ieqs=[p]), Polyhedron(ieqs=[n])
    nxt = []
    for c in cells:
        a, b = hp2.intersection(c), hn2.intersection(c)
        if a.dim() == 3 and b.dim() == 3:
            nxt.extend([a, b])
        else:
            nxt.append(c)
    cells = nxt
total = sum(c.volume() for c in cells)
check("all leaves are 3D", all(c.dim() == 3 for c in cells))
check("BSP partition conserves volume exactly", total == 8, total)
check("BSP produced >1 cells", len(cells) > 1, len(cells))

print("=" * 72)
print("8. render_solid().obj_repr() used by save_surface / write_cell")
ss = bb.render_solid().obj_repr(bb.render_solid().default_render_params())
check("ss[2] has 8 'v x y z' lines", len(ss[2]) == 8 and
      all(t.startswith("v ") for t in ss[2]), ss[2][:2])
check("ss[3] has 6 'f i j k' lines", len(ss[3]) == 6 and
      all(t.startswith("f ") and len(t[2:].split(" ")) >= 3 for t in ss[3]),
      ss[3][:2])
check("OBJ face indices are 1-based ints",
      all(int(i) >= 1 for t in ss[3] for i in t[2:].split(" ")))

print("=" * 72)
print("RESULT: %d passed, %d failed" % (len(PASS), len(FAIL)))
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("sage.all shim reproduces the COMPOD Sage kernel on native Windows.")

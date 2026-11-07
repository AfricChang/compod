# -*- coding: utf-8 -*-
"""
Minimal drop-in replacement for the subset of ``sage.all`` used by COMPOD.

Motivation
----------
SageMath has no native Windows distribution.  Sage's ``Polyhedron`` over QQ
uses cddlib (double description method) as one of its backends, and the Sage
``ieqs`` convention ``[b, a1, a2, a3] -> b + a.x >= 0`` is *identical* to the
cddlib H-matrix layout.  This shim therefore reimplements the API surface used
by ``pycompod`` directly on top of ``cdd.gmp`` (pycddlib's exact rational
backend, based on ``fractions.Fraction``).

Install (native Windows, no WSL, no compiler)::

    pip install pycddlib gmpy2

Then put the directory containing this ``sage`` package on PYTHONPATH, or
copy this package next to ``pycompod``.  ``from sage.all import ...`` in
pycompod will then resolve to this module.

Implemented API (verified against pycompod's hot paths):
    QQ, vector, Matrix, arctan2, polytopes, Polyhedron
    Polyhedron: ieqs / vertices / base_ring, intersection, dim,
                vertices_list, bounding_box, volume, center,
                inequalities_list, adjacency_matrix, is_empty,
                render_solid().obj_repr(), scale

Known limitations (see test_compat.py for what IS covered):
  * ``center()`` returns a tuple rather than a Sage vector.  pycompod only
    consumes it as an iterable / np.array input, so this is compatible.
  * ``vector.norm()`` returns a float sqrt, not Sage's symbolic sqrt.
    Only ``SagePlane`` uses it, and that class is never instantiated.
  * ``parent()`` / ``element_class()`` / ``affine_hull_projection()`` are
    Sage internals used by ``_polyhedron_intersection`` in complex.py, which
    is currently dead code (commented out at all call sites).
  * Exactness is preserved end-to-end (Fraction), but Fraction arithmetic is
    slower than Sage's QQ (GMP).  Swap QQ for gmpy2.mpq and convert at the cdd
    boundary if profiling shows this matters.
"""

from __future__ import annotations

import math
from fractions import Fraction

try:
    from cdd import gmp as _g
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "sage/all.py shim needs pycddlib >= 3.0 (cdd.gmp backend). "
        "Install with: pip install pycddlib"
    ) from exc


# --------------------------------------------------------------------------
# Exact rational field.  Sage's QQ(float) produces the exact binary rational
# behind the IEEE double; Fraction(float) does exactly the same.
# --------------------------------------------------------------------------
class QQ(Fraction):
    """Exact rational, like Sage's QQ, but also usable in float format specs.

    pycompod writes PLY/OBJ headers with "{:6f}".format(v) on object arrays
    holding exact rationals (export_complex.py:329).  Sage's Rational supports
    that; plain fractions.Fraction raises TypeError.
    """

    def __format__(self, format_spec):
        if format_spec and format_spec[-1] in "eEfFgG%":
            return format(float(self), format_spec)
        return super().__format__(format_spec)


def _fraction_format(self, format_spec):
    """Sage's Rational accepts float format specs; Fraction does not.

    Arithmetic on a QQ instance returns a plain Fraction, so patching the base
    class (only adding support for specs that previously raised TypeError) is
    the only way to cover every value pycompod writes to PLY/OBJ.
    """
    if format_spec and format_spec[-1] in "eEfFgG%":
        return format(float(self), format_spec)
    return object.__format__(self, format_spec)


if not getattr(Fraction, "_compod_float_format", False):
    Fraction.__format__ = _fraction_format
    Fraction._compod_float_format = True


def _q(x):
    """Coerce anything (float, numpy scalar, mpq, int, str) to exact QQ."""
    if isinstance(x, QQ):
        return x
    if isinstance(x, Fraction):
        return QQ(x.numerator, x.denominator)
    try:
        return QQ(x)
    except (TypeError, ValueError):
        return QQ(float(x))


def arctan2(y, x):
    # pycompod wraps the result in float() at complex.py:411
    return math.atan2(float(y), float(x))


class polytopes:  # noqa: N801 - imported by vg.py but never used
    pass


# --------------------------------------------------------------------------
# Lightweight vector / matrix (only the operations pycompod actually calls)
# --------------------------------------------------------------------------
class vector:  # noqa: N801 - mirrors sage.all.vector spelling
    def __init__(self, arg=None, ring=None):
        if ring is not None:  # Sage also allows vector(QQ, [..])
            arg, ring = ring, arg
        if arg is None:
            self._v = []
        elif isinstance(arg, vector):
            self._v = list(arg._v)
        else:
            self._v = [_q(t) if ring is not None else t for t in arg]
            if ring is not None:
                self._v = [_q(t) for t in self._v]

    def __len__(self):
        return len(self._v)

    def __iter__(self):
        return iter(self._v)

    def __getitem__(self, i):
        return self._v[i]

    def __repr__(self):
        return "(" + ", ".join(map(str, self._v)) + ")"

    def __add__(self, other):
        return vector([a + b for a, b in zip(self._v, other)])

    def __sub__(self, other):
        return vector([a - b for a, b in zip(self._v, other)])

    def __neg__(self):
        return vector([-a for a in self._v])

    def __rmul__(self, s):
        return vector([s * a for a in self._v])

    def __mul__(self, other):
        if isinstance(other, (vector, list, tuple)):
            return sum(a * b for a, b in zip(self._v, other))
        if isinstance(other, Matrix):
            return Matrix([[sum(a * b for a, b in zip(self._v, col))
                            for col in zip(*other._m)]])
        return vector([a * other for a in self._v])

    def __truediv__(self, s):
        return vector([a / s if not isinstance(a, float) else a / s
                       for a in self._v])

    def dot_product(self, other):
        return sum(a * b for a, b in zip(self._v, other))

    def cross_product(self, other):
        a, b = self._v, list(other)
        return vector([a[1] * b[2] - a[2] * b[1],
                       a[2] * b[0] - a[0] * b[2],
                       a[0] * b[1] - a[1] * b[0]])

    def norm(self):
        return math.sqrt(sum(float(a) * float(a) for a in self._v))

    def numpy(self, dtype=float):
        import numpy as np
        return np.array([float(a) for a in self._v], dtype=dtype)

    def __array__(self, dtype=None):
        import numpy as np
        return np.array(self._v, dtype=dtype or object)


class Matrix:  # noqa: N801 - mirrors sage.all.Matrix spelling
    def __init__(self, *args):
        # Matrix(QQ, rows) / Matrix(rows) / Matrix(vector)
        if len(args) == 2:
            ring, rows = args
            rows = [[_q(t) for t in row] for row in rows]
        elif len(args) == 1:
            a = args[0]
            if isinstance(a, vector):
                rows = [list(a)]
            elif isinstance(a, Matrix):
                rows = [list(r) for r in a._m]
            else:
                rows = [list(r) for r in a]
        else:
            rows = []
        self._m = rows

    def transpose(self):
        return Matrix([list(col) for col in zip(*self._m)])

    def __mul__(self, other):
        if isinstance(other, Matrix):
            cols = list(zip(*other._m))
            return Matrix([[sum(a * b for a, b in zip(row, col)) for col in cols]
                           for row in self._m])
        return Matrix([[a * other for a in row] for row in self._m])

    def __add__(self, other):
        return Matrix([[a + b for a, b in zip(r1, r2)]
                       for r1, r2 in zip(self._m, other._m)])

    def __iter__(self):
        return iter(self._m)

    def __getitem__(self, i):
        return self._m[i]

    def __repr__(self):
        return "Matrix(" + repr(self._m) + ")"


# --------------------------------------------------------------------------
# Exact rational polyhedron
# --------------------------------------------------------------------------
def _rank(rows):
    """Exact rank of a list of row vectors over QQ."""
    m = [list(r) for r in rows]
    if not m:
        return 0
    ncol = len(m[0])
    rank = 0
    for c in range(ncol):
        piv = next((i for i in range(rank, len(m)) if m[i][c] != 0), None)
        if piv is None:
            continue
        m[rank], m[piv] = m[piv], m[rank]
        pv = m[rank][c]
        for i in range(rank + 1, len(m)):
            if m[i][c] != 0:
                f = m[i][c] / pv
                for j in range(c, ncol):
                    m[i][j] -= f * m[rank][j]
        rank += 1
        if rank == len(m):
            break
    return rank


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def _det3(a, b, c):
    return (a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))


class Polyhedron:
    """
    Exact convex polyhedron with an H-representation.

    ``ieqs`` rows are ``[b, a1, a2, a3]`` meaning ``b + a.x >= 0`` -- byte for
    byte the same convention Sage uses (complex.py:183, vg.py:631).
    """

    def __init__(self, ieqs=None, vertices=None, base_ring=None, eqns=None,
                 _h=None):
        if _h is not None:
            self._h = [tuple(_q(t) for t in row) for row in _h]
        elif ieqs is not None:
            self._h = [tuple(_q(t) for t in row) for row in ieqs]
        elif vertices is not None:
            self._h = self._v_to_h(vertices)
        else:
            raise ValueError("Polyhedron needs ieqs= or vertices=")
        self._eqns = [tuple(_q(t) for t in row) for row in (eqns or [])]
        self._reset_cache()

    def _reset_cache(self):
        self._poly = None
        self._gen_rows = None
        self._gen_lin = None
        self._adj = None
        self._verts = None
        self._vindex = None

    # complex.py:3144 pickles self.cells / graph edge "intersection", which
    # contain Polyhedron instances. cdd's C-level Polyhedron is not picklable,
    # so persist only the exact H-representation and rebuild lazily.
    def __getstate__(self):
        return {"_h": self._h, "_eqns": self._eqns}

    def __setstate__(self, state):
        self._h = state["_h"]
        self._eqns = state.get("_eqns", [])
        self._reset_cache()

    @staticmethod
    def _v_to_h(vertices):
        # cdd V-representation: each vertex is [1, x, y, z]
        rows = [[QQ(1)] + [_q(t) for t in pt] for pt in vertices]
        mat = _g.matrix_from_array(rows, rep_type=_g.RepType.GENERATOR)
        poly = _g.polyhedron_from_matrix(mat)
        return [tuple(_q(t) for t in row) for row in _g.copy_inequalities(poly).array]

    # -- lazy double description ------------------------------------------
    def _run(self):
        if self._poly is not None:
            return
        rows = [list(r) for r in self._h] + [list(r) for r in self._eqns]
        if not rows:
            rows = [[Fraction(0), Fraction(0), Fraction(0), Fraction(0)]]
        mat = _g.matrix_from_array(rows, rep_type=_g.RepType.INEQUALITY)
        if self._eqns:
            mat.lin_set = set(range(len(self._h), len(rows)))
        self._poly = _g.polyhedron_from_matrix(mat)
        gen = _g.copy_generators(self._poly)
        self._gen_rows = [list(_q(t) for t in row) for row in gen.array]
        self._gen_lin = set(gen.lin_set)
        try:
            self._adj = _g.copy_adjacency(self._poly)
        except Exception:
            self._adj = []

    # -- Sage-compatible API ----------------------------------------------
    def intersection(self, other):
        """Halfspace intersection == concatenating H-representations.

        This is exactly what Sage's cdd backend does internally; keeping it
        lazy avoids a V->H round trip on every BSP split.
        """
        return Polyhedron(_h=list(self._h) + list(other._h),
                          eqns=list(self._eqns) + list(other._eqns))

    def vertices_list(self):
        self._run()
        if self._verts is None:
            self._verts = []
            self._vindex = []
            for i, row in enumerate(self._gen_rows):
                if row[0] != 0:  # generators with leading 1 are vertices
                    self._verts.append(tuple(row[1:4]))
                    self._vindex.append(i)
        return self._verts

    # -- Sage aliases used across pycompod --------------------------------
    def vertices(self):
        """Alias of vertices_list(); Sage returns Vertex objects, tuples work
        identically for the np.array(...) / indexing calls in pycompod."""
        return list(self.vertices_list())

    def vertex_generator(self):
        return iter(self.vertices_list())

    def n_vertices(self):
        return len(self.vertices_list())

    def n_inequalities(self):
        return len(self._h)

    def n_equations(self):
        return len(self._eqns)

    def is_full_dimensional(self):
        return self.dim() == 3

    def Vrepresentation(self):
        return tuple(self.vertices_list())

    def Hrepresentation(self):
        return tuple(self._h) + tuple(self._eqns)

    def contains(self, point):
        p = [_q(t) for t in point]
        return all(h[0] + sum(h[1 + k] * p[k] for k in range(3)) >= 0
                   for h in self._h)

    def parent(self):
        return _PolyhedronParent()

    def dim(self):
        self._run()
        if not self._gen_rows:
            return -1
        verts = self.vertices_list()
        dirs = []
        if verts:
            for v in verts[1:]:
                dirs.append([v[k] - verts[0][k] for k in range(3)])
        for row in self._gen_rows:
            if row[0] == 0:  # ray / line directions
                dirs.append([row[1], row[2], row[3]])
        return _rank(dirs) if dirs else 0

    def is_empty(self):
        self._run()
        return len(self._gen_rows) == 0

    def bounding_box(self):
        verts = self.vertices_list()
        if not verts:
            return None
        lo = tuple(min(v[k] for v in verts) for k in range(3))
        hi = tuple(max(v[k] for v in verts) for k in range(3))
        return (lo, hi)

    def center(self):
        """Average of vertices (Sage semantics).  Returned as a tuple."""
        verts = self.vertices_list()
        n = len(verts)
        if n == 0:
            return None
        return tuple(_q(sum(v[k] for v in verts) / n) for k in range(3))

    def inequalities_list(self):
        return [list(h) for h in self._h]

    def inequalities(self):
        return tuple(self._h)

    def equations(self):
        return tuple(self._eqns)

    def adjacency_matrix(self):
        """Vertex adjacency as a 0/1 matrix, indexed like vertices_list()."""
        self._run()
        verts = self.vertices_list()
        pos = {gi: k for k, gi in enumerate(self._vindex)}
        n = len(verts)
        mat = [[0] * n for _ in range(n)]
        for gi, neighbours in enumerate(self._adj):
            if gi not in pos:
                continue
            for gj in neighbours:
                if gj in pos:
                    mat[pos[gi]][pos[gj]] = 1
        return mat

    # -- faces / volume / OBJ export --------------------------------------
    def _face_rings(self):
        """Return each facet as an ordered ring of vertex indices."""
        verts = self.vertices_list()
        if len(verts) < 3:
            return []
        rings, seen = [], set()
        for h in self._h:
            b, a = h[0], h[1:4]
            if all(t == 0 for t in a):
                continue
            on = [i for i, v in enumerate(verts)
                  if b + sum(a[k] * v[k] for k in range(3)) == 0]
            if len(on) < 3:
                continue
            key = frozenset(on)
            if key in seen:  # redundant inequalities describe the same facet
                continue
            seen.add(key)
            # build an orthonormal basis inside the facet plane (float is fine
            # here: it only determines the cyclic ordering, not the geometry)
            af = [float(t) for t in a]
            ln = math.sqrt(sum(t * t for t in af))
            n3 = [t / ln for t in af]
            axis = min(range(3), key=lambda k: abs(n3[k]))
            e = [0.0, 0.0, 0.0]
            e[axis] = 1.0
            u = _cross(n3, e)
            lu = math.sqrt(sum(t * t for t in u))
            u = [t / lu for t in u]
            w = _cross(n3, u)
            cf = [sum(float(verts[i][k]) for i in on) / len(on) for k in range(3)]

            def angle(i):
                d = [float(verts[i][k]) - cf[k] for k in range(3)]
                return math.atan2(sum(d[k] * w[k] for k in range(3)),
                                  sum(d[k] * u[k] for k in range(3)))

            on = sorted(on, key=angle)
            # orient outward
            v0, v1, v2 = (verts[on[j]] for j in range(3))
            d1 = [float(v1[k] - v0[k]) for k in range(3)]
            d2 = [float(v2[k] - v0[k]) for k in range(3)]
            if sum(_cross(d1, d2)[k] * n3[k] for k in range(3)) < 0:
                on.reverse()
            rings.append(on)
        return rings

    def volume(self):
        """Exact Euclidean volume (Fraction), via divergence-free tetrahedra."""
        if self.dim() < 3:
            return Fraction(0)
        verts = self.vertices_list()
        o = self.center()
        total = QQ(0)
        for ring in self._face_rings():
            v0 = [verts[ring[0]][k] - o[k] for k in range(3)]
            for j in range(1, len(ring) - 1):
                va = [verts[ring[j]][k] - o[k] for k in range(3)]
                vb = [verts[ring[j + 1]][k] - o[k] for k in range(3)]
                total += abs(_det3(v0, va, vb))
        return _q(total / 6)

    def scale(self, factor, center=None):
        """Uniform scaling; H-representation is rewritten analytically."""
        f = _q(factor)
        if f <= 0:
            raise ValueError("scale factor must be positive")
        c = [_q(t) for t in (center if center is not None else (0, 0, 0))]
        new_h = []
        for h in self._h:
            # x' = c + f*(x - c)  =>  x = c + (x' - c)/f
            # b + a.x >= 0  <=>  [f*(b + a.c) - a.c] + a.x' >= 0
            b, a = h[0], h[1:4]
            ac = sum(a[k] * c[k] for k in range(3))
            new_h.append([f * (b + ac) - ac, a[0], a[1], a[2]])
        return Polyhedron(_h=new_h)

    def render_solid(self):
        return _SolidRenderer(self)

    def __repr__(self):
        return "Polyhedron(ieqs=%d, dim=%s)" % (len(self._h), self.dim())


class _PolyhedronParent:
    """Supports the Sage-internal idiom in complex.py:_polyhedron_intersection
    (currently dead code, but cheap to keep compatible)."""

    def element_class(self, parent, Vrep, Hrep):
        return Polyhedron(_h=Hrep[0], eqns=Hrep[1])


class _SolidRenderer:
    """Mimics sage.plot.plot3d.solid, whose obj_repr() pycompod consumes."""

    def __init__(self, poly):
        self.poly = poly

    def default_render_params(self):
        return {}

    def obj_repr(self, params=None):
        # complex.py:1076 and export_complex.py:109 use ss[2] (vertex lines)
        # and ss[3] (face lines); indices in faces are 1-based.
        verts = ["v %s %s %s" % tuple(str(float(t)) for t in v)
                 for v in self.poly.vertices_list()]
        faces = ["f " + " ".join(str(i + 1) for i in ring)
                 for ring in self.poly._face_rings()]
        return (None, None, verts, faces)

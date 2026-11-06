"""
The two representations a surface's signed distance field is evaluated in, both over the knot samples one bake fits
(`lodos.col.sdf.implicit.create_implicit_surface_from_mesh`):

    c0   trilinear interpolation of the knot samples, 8 a query: C0 field, normals discontinuous across cell faces
    c2   the tricubic B-spline through them, 64 coefficients a query: C2 field, C1 normals

A surface picks one with its `spline_type` ("linear" for `c0`, "cubic" for `c2`; `lodos.col.sdf.SPLINE_TYPES`). The
packed collision tables use the same names, beside `vox`, the voxel-centre lookup.
"""

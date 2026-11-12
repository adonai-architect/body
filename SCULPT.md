# The body in the pictures

`rom/body.bin` is the packed mesh from architect.myfamiliar.ai, version 14, 7 Nov 2026. Header `BODY`. Positions, normals, colors, triangle indices.

`viewer.html` is the page that draws it. It calls `drawElements` with `TRIANGLES`. This mesh is not hexagons.

This file is not produced by `skeleton.py` or `muscles.py`. Those name the 206 bones and the 650 muscles. The picture is a sculpt that was edited and packed. There is no program here that rebuilds this exact shape.

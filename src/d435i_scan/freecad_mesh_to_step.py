"""Executed by FreeCADCmd, not by the application's Python interpreter."""

from __future__ import annotations

import json
import sys


def main() -> int:
    if len(sys.argv) < 4:
        print("D435I_STEP_RESULT=" + json.dumps({"ok": False, "reason": "参数不足"}))
        return 2
    mesh_path, step_path, tolerance_text = sys.argv[-3:]
    tolerance_mm = float(tolerance_text)
    try:
        import Mesh
        import Part

        mesh = Mesh.Mesh(mesh_path)
        shape = Part.Shape()
        shape.makeShapeFromMesh(mesh.Topology, tolerance_mm)
        try:
            shape = shape.removeSplitter()
        except Exception:
            pass
        was_closed = bool(shape.isClosed())
        if was_closed and shape.ShapeType == "Shell":
            shape = Part.makeSolid(shape)
        shape.exportStep(step_path)
        payload = {
            "ok": True,
            "closed_input": was_closed,
            "shape_type": shape.ShapeType,
            "faces": len(shape.Faces),
            "kind": "faceted_brep",
        }
        print("D435I_STEP_RESULT=" + json.dumps(payload))
        return 0
    except Exception as exc:
        print(
            "D435I_STEP_RESULT=" + json.dumps({"ok": False, "reason": str(exc)}, ensure_ascii=False)
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

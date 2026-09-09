from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def find_freecad_cmd(configured: str = "") -> Path | None:
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if candidate.is_file():
            return candidate
    for executable in ("FreeCADCmd.exe", "FreeCADCmd", "freecadcmd"):
        found = shutil.which(executable)
        if found:
            return Path(found).resolve()
    roots = [Path("C:/Program Files"), Path("C:/Program Files (x86)")]
    for root in roots:
        if not root.exists():
            continue
        candidates = sorted(root.glob("FreeCAD*/bin/FreeCADCmd.exe"), reverse=True)
        if candidates:
            return candidates[0].resolve()
    return None


def export_faceted_step(
    mesh_path: str | Path,
    step_path: str | Path,
    configured_freecad_cmd: str = "",
    tolerance_mm: float = 0.05,
) -> dict[str, Any]:
    freecad_cmd = find_freecad_cmd(configured_freecad_cmd)
    if freecad_cmd is None:
        return {
            "ok": False,
            "reason": "未找到 FreeCADCmd。安装免费 FreeCAD 后重试 STEP 导出。",
            "kind": "faceted_brep",
        }
    script = Path(__file__).resolve().with_name("freecad_mesh_to_step.py")
    process = subprocess.run(
        [
            str(freecad_cmd),
            str(script),
            str(Path(mesh_path).resolve()),
            str(Path(step_path).resolve()),
            str(float(tolerance_mm)),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
        check=False,
    )
    result: dict[str, Any] = {
        "ok": process.returncode == 0 and Path(step_path).is_file(),
        "returncode": process.returncode,
        "stdout": process.stdout[-4000:],
        "stderr": process.stderr[-4000:],
        "kind": "faceted_brep",
        "freecad_cmd": str(freecad_cmd),
    }
    for line in reversed(process.stdout.splitlines()):
        if line.startswith("D435I_STEP_RESULT="):
            try:
                result.update(json.loads(line.split("=", 1)[1]))
            except json.JSONDecodeError:
                pass
            break
    if not result["ok"] and "reason" not in result:
        result["reason"] = "FreeCAD 转换失败；详情见 report.json。"
    return result

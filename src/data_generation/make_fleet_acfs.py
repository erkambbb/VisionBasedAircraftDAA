"""Builds X-Plane-loadable .acf files for the 166 UAV models.

The generated .acf files in plane_models/ are minimal (no Vne, weights, ...) and X-Plane
refuses them ("You entered a Vne of zero!"). Here every model gets the complete
shadow_twin_boom.acf as template with only the visible object replaced by the model's OBJ.
The template's CG is 3.3 ft aft / 0.3 ft above the model origin (see CG_OFFSET_M in fleet_core).

Usage: python3 make_fleet_acfs.py [--models-dir DIR] [--xp-dir DIR]
"""
import argparse
import os
import shutil

XP_DIR = os.path.expanduser("~/.local/share/Steam/steamapps/common/X-Plane 11")
TEMPLATE_OBJ_LINE = "P _obja/28/_v10_att_file_stl shadow_twin_boom_xp.obj"


def fix_obj(text):
    """OBJ8 'IDX10' lines must hold exactly 10 indices; the generator leaves a short last line
    (X-Plane: "IDX10 index is out of range - the OBJ file is bad"). Short lines become 'IDX' lines."""
    out = []
    for line in text.splitlines():
        if line.startswith("IDX10 "):
            vals = line.split()[1:]
            if len(vals) != 10:
                out.extend(f"IDX {v}" for v in vals)
                continue
        out.append(line)
    return "\n".join(out) + "\n"


def build(models_dir, xp_dir):
    extra = os.path.join(xp_dir, "Aircraft", "Extra Aircraft")
    tpl_dir = os.path.join(extra, "shadow_twin_boom")
    tpl = open(os.path.join(tpl_dir, "shadow_twin_boom.acf"), encoding="utf-8").read()
    if TEMPLATE_OBJ_LINE not in tpl:
        raise SystemExit("şablon acf beklenen nesne satırını içermiyor")
    out_root = os.path.join(extra, "uav_fleet")
    shutil.rmtree(out_root, ignore_errors=True)
    names = sorted(d for d in os.listdir(models_dir) if d.startswith("uav_") and os.path.isdir(os.path.join(models_dir, d)))
    for name in names:
        dst = os.path.join(out_root, name)
        os.makedirs(os.path.join(dst, "objects"), exist_ok=True)
        for f in os.listdir(os.path.join(models_dir, name, "objects")):
            src = os.path.join(models_dir, name, "objects", f)
            if f.endswith(".obj"):
                open(os.path.join(dst, "objects", f), "w").write(fix_obj(open(src).read()))
            else:
                shutil.copy2(src, os.path.join(dst, "objects", f))
        shutil.copytree(os.path.join(tpl_dir, "airfoils"), os.path.join(dst, "airfoils"))
        shutil.copy2(os.path.join(tpl_dir, "objects", "empty.obj"), os.path.join(dst, "objects", "empty.obj"))
        acf = tpl.replace(TEMPLATE_OBJ_LINE, f"P _obja/28/_v10_att_file_stl {name}.obj")
        # X-Plane çizdiği pervane diski şablondaki konumda kalıyor, uçakla alakasız -> gizle
        acf = acf.replace("P acf/_num_prop 1\n", "P acf/_num_prop 0\n").replace("P _prop/0/_num_blades 2.0\n", "P _prop/0/_num_blades 0.0\n")
        open(os.path.join(dst, f"{name}.acf"), "w", encoding="utf-8").write(acf)
    print(f"{len(names)} model yazıldı: {out_root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=os.path.expanduser("~/Belgeler/ARIKOVANI/plane_models"))
    ap.add_argument("--xp-dir", default=XP_DIR)
    a = ap.parse_args()
    build(a.models_dir, a.xp_dir)

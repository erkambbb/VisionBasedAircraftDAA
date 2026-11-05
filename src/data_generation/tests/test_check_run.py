import json
import os

from data_generation import check_run as cr


def _dataset(tmp_path, done, frames):
    for sub in ("images/train", "images/valid"):
        (tmp_path / sub).mkdir(parents=True)
    for idx, n in frames.items():
        for i in range(n):
            (tmp_path / "images/train" / f"u{idx:03d}_{i:03d}.jpg").write_bytes(b"x")
    (tmp_path / "progress.json").write_text(json.dumps({"done": done, "failed": []}))
    (tmp_path / "fleet_run.log").write_text("[001] tamam\n  X-Plane cevap vermedi (x)\nKONTROL: 120 kare, sorun yok\n")
    return str(tmp_path)


def test_healthy_run_has_no_problems(tmp_path):
    root = _dataset(tmp_path, ["uav_001_a"], {1: 120, 2: 7})
    lines, problems = cr.report(root, True, True)
    assert problems == []
    assert any("kare: 127 / 20800" in l for l in lines) and any("KONTROL satırı: 1, sorunlu: 0" in l for l in lines)
    assert not any("cevap vermedi" in l for l in lines)


def test_reports_dead_processes_wrong_count_and_stale_frames(tmp_path):
    root = _dataset(tmp_path, ["uav_001_a"], {1: 119})
    assert set(cr.report(root, False, False)[1]) == {"kare sayısı yanlış: ['uav_001_a']", "üretim çalışmıyor", "X-Plane kapalı"}
    old = os.path.join(root, "images/train/u001_000.jpg")
    for f in os.listdir(os.path.dirname(old)):
        os.utime(os.path.join(os.path.dirname(old), f), (1, 1))
    assert any("yeni kare yok" in p for p in cr.report(root, True, True)[1])


def test_big_model_expects_1000():
    assert cr.expected("uav_166_tactical_talon") == 1000 and cr.expected("uav_002_x") == 120

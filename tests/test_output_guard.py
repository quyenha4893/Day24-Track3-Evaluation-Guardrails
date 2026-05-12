"""Tests for OutputGuard."""

import sys, importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

spec = importlib.util.spec_from_file_location("og", Path(__file__).parent.parent / "phase-c" / "output_guard.py")
og = importlib.util.module_from_spec(spec)
spec.loader.exec_module(og)
OutputGuard = og.OutputGuard


def test_safe_response_marked_safe(monkeypatch):
    guard = OutputGuard()
    monkeypatch.setattr(guard, "_call_gemini", lambda u, a: "safe")
    ok, verdict, ms = guard.check("Hỏi gì?", "Đây là câu trả lời bình thường.")
    assert ok is True
    assert "safe" in verdict.lower()


def test_unsafe_response_marked_unsafe(monkeypatch):
    guard = OutputGuard()
    monkeypatch.setattr(guard, "_call_gemini", lambda u, a: "unsafe\nS1")
    ok, verdict, _ = guard.check("Hỏi?", "Nội dung bạo lực ...")
    assert ok is False


def test_exception_fails_closed(monkeypatch):
    guard = OutputGuard()

    def boom(u, a):
        raise RuntimeError("Gemini down")

    monkeypatch.setattr(guard, "_call_gemini", boom)
    ok, verdict, _ = guard.check("q", "a")
    assert ok is False
    assert "fail_closed" in verdict or "error" in verdict.lower()

"""Tests for InputGuard PII redaction."""

import sys
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

spec = importlib.util.spec_from_file_location("ig", Path(__file__).parent.parent / "phase-c" / "input_guard.py")
ig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ig)
InputGuard = ig.InputGuard


def test_vn_cccd_redacted():
    guard = InputGuard()
    out, _ = guard.sanitize("Số CCCD của tôi là 012345678901")
    assert "012345678901" not in out
    assert "[CCCD]" in out


def test_phone_redacted():
    guard = InputGuard()
    out, _ = guard.sanitize("Liên hệ 0987654321")
    assert "0987654321" not in out


def test_empty_input():
    guard = InputGuard()
    out, _ = guard.sanitize("")
    assert out == ""


def test_latency_under_50ms():
    guard = InputGuard()
    _, ms = guard.sanitize("Số CCCD 012345678901, phone 0987654321, email a@b.com")
    assert ms < 50, f"latency {ms}ms exceeds 50ms"

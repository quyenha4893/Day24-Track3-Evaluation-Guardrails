"""Bonus — NeMo Guardrails topic validator, offline comparison vs custom TopicGuard."""

import os, time, builtins
from pathlib import Path

# Load .env before NeMo initializes (NeMo reads OPENAI_API_KEY at LLMRails() time)
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(str(_env_file))
    except ImportError:
        # Fallback: parse .env manually
        for line in _env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# NeMo reads YAML files without specifying encoding; on Windows the default
# charmap codec cannot decode UTF-8 Vietnamese characters.  Monkey-patch
# builtins.open so that .yml/.yaml files are always opened as UTF-8.
_orig_open = builtins.open


def _utf8_open(file, mode="r", **kwargs):
    if isinstance(file, str) and file.endswith((".yml", ".yaml")) and "b" not in mode:
        kwargs.setdefault("encoding", "utf-8")
    return _orig_open(file, mode, **kwargs)


builtins.open = _utf8_open

# NeMo block message (default for self check input flow)
_NEMO_BLOCK_MSG = "I'm sorry, I can't respond to that."


class NemoTopicGuard:
    """Wraps NeMo Guardrails self_check_input flow for topic filtering.

    NeMo semantics:
      - self_check_input prompt asks "Should this be blocked? Yes/No"
      - is_content_safe parser: Yes -> blocked (off-topic), No -> allowed (on-topic)
      - When blocked, generate() returns the default refuse-to-respond message.
    """

    def __init__(self):
        from nemoguardrails import LLMRails, RailsConfig
        config = RailsConfig.from_path(str(Path(__file__).parent / "nemo_config.yml"))
        self.rails = LLMRails(config)

    def check(self, text: str) -> tuple[bool, str, float]:
        """Return (on_topic: bool, reason: str, latency_ms: float).

        on_topic=True  -> NeMo allowed the message (not blocked)
        on_topic=False -> NeMo blocked the message (off-topic)
        """
        start = time.perf_counter()
        try:
            resp = self.rails.generate(messages=[{"role": "user", "content": text}])
            content = resp.get("content", "") if isinstance(resp, dict) else str(resp)
            # NeMo returns the refusal string when input is blocked
            blocked = _NEMO_BLOCK_MSG.lower() in content.lower()
            on_topic = not blocked
            reason = f"NeMo: {'allowed' if on_topic else 'blocked'} | response={content[:60]!r}"
        except Exception as e:
            on_topic = True  # fail-open
            reason = f"nemo_error: {e}"
        ms = (time.perf_counter() - start) * 1000
        return on_topic, reason, ms


def main():
    import pandas as pd
    from pathlib import Path

    print("Initializing NeMo Guardrails...")
    nemo = NemoTopicGuard()
    df = pd.read_csv(Path(__file__).parent / "topic_test_results.csv")
    rows = []
    print(f"Running NeMo on {len(df)} inputs...")
    for i, r in enumerate(df.itertuples(), 1):
        on_topic, reason, ms = nemo.check(r.input)
        rows.append({
            "input": r.input,
            "custom_predicted": r.on_topic_predicted,
            "nemo_predicted": on_topic,
            "expected": r.on_topic_expected,
            "nemo_latency_ms": round(ms, 1),
            "nemo_reason": reason,
        })
        safe_input = r.input[:40].encode("ascii", errors="replace").decode("ascii")
        print(f"  [{i:2d}/20] expected={r.on_topic_expected} nemo={on_topic} {ms:.0f}ms | {safe_input!r}")
        time.sleep(0.5)  # mild rate limit

    out = pd.DataFrame(rows)
    out.to_csv(Path(__file__).parent / "topic_comparison_nemo.csv", index=False)

    custom_acc = (df["on_topic_expected"] == df["on_topic_predicted"]).mean()
    nemo_acc = (out["expected"] == out["nemo_predicted"]).mean()
    avg_lat = out["nemo_latency_ms"].mean()
    print(f"\nCustom TopicGuard acc: {custom_acc:.1%}")
    print(f"NeMo Guardrails acc:   {nemo_acc:.1%}")
    print(f"NeMo avg latency:      {avg_lat:.0f}ms")


if __name__ == "__main__":
    main()

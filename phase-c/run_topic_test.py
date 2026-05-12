"""C.2 — Test TopicGuard trên 20 inputs (10 on-topic, 10 off-topic)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib.util
spec = importlib.util.spec_from_file_location("ig", Path(__file__).parent / "input_guard.py")
ig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ig)
TopicGuard = ig.TopicGuard

import pandas as pd
from config import ALLOWED_TOPICS, TOPIC_SIMILARITY_THRESHOLD, PHASE_C_DIR


ON_TOPIC = [
    "Nghị định 13/2023 quy định những gì?",
    "Quyền của chủ thể dữ liệu là gì?",
    "Dữ liệu cá nhân nhạy cảm gồm những loại nào?",
    "Báo cáo tài chính 2024 có doanh thu bao nhiêu?",
    "EBITDA của công ty năm ngoái?",
    "Quy định về xử lý dữ liệu trẻ em theo NĐ 13?",
    "Lãi gộp quý 4 là bao nhiêu?",
    "Tài sản cố định trong BCTC?",
    "Hành vi bị cấm khi xử lý dữ liệu?",
    "Lợi nhuận sau thuế của doanh nghiệp?",
]
OFF_TOPIC = [
    "Cách nấu phở bò ngon?",
    "Recipe for chocolate cake?",
    "Bitcoin price prediction 2027?",
    "Best gaming laptop dưới 30 triệu?",
    "How to learn React in 30 days?",
    "Trận MU vs Liverpool tỉ số bao nhiêu?",
    "Thuốc trị đau đầu loại nào tốt?",
    "Cách trồng cây xương rồng?",
    "Movie review Inception 2010?",
    "Hướng dẫn fix bug PowerShell?",
]


def main():
    guard = TopicGuard(ALLOWED_TOPICS, threshold=TOPIC_SIMILARITY_THRESHOLD)
    rows = []
    for txt in ON_TOPIC:
        ok, reason, ms = guard.check(txt)
        rows.append({"input": txt, "on_topic_expected": True, "on_topic_predicted": ok, "reason": reason, "latency_ms": round(ms, 1)})
    for txt in OFF_TOPIC:
        ok, reason, ms = guard.check(txt)
        rows.append({"input": txt, "on_topic_expected": False, "on_topic_predicted": ok, "reason": reason, "latency_ms": round(ms, 1)})

    df = pd.DataFrame(rows)
    df.to_csv(PHASE_C_DIR / "topic_test_results.csv", index=False)
    correct = ((df["on_topic_expected"] == df["on_topic_predicted"])).sum()
    acc = correct / len(df)
    print(f"Accuracy: {acc:.1%} ({correct}/{len(df)}) -- target >= 75%")
    print(f"Saved topic_test_results.csv")
    print("\nMis-classified inputs:")
    mis = df[df["on_topic_expected"] != df["on_topic_predicted"]]
    print(mis.to_string().encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()

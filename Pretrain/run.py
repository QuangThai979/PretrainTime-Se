"""
run.py - Entry point CLI.
Ví dụ:
  python run.py data/input/sensor_data.csv --time_col time --targets temp humidity --freq 1min
"""
from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

import pandas as pd


def _block_network() -> None:
    """Chốt chặn offline: mọi kết nối mạng đều bị chặn -> mô hình chỉ chạy local."""
    def _deny(*_a, **_k):
        raise RuntimeError("Network bị chặn: pipeline này chỉ chạy local, không gọi API ngoài.")
    socket.socket.connect = _deny  # type: ignore[assignment]
    socket.create_connection = _deny  # type: ignore[assignment]


_block_network()

from pipeline import TimeSeriesPipeline
from preprocess import preprocess_timeseries


def read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext in (".csv", ".txt"):
        return pd.read_csv(path)
    raise ValueError(f"Định dạng không hỗ trợ: {ext} (dùng CSV/Parquet)")


def main() -> int:
    # Windows console thường là cp1252 -> ép UTF-8 để in được tiếng Việt
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="Time-series cleaning & anomaly pipeline")
    ap.add_argument("input", help="Đường dẫn file, vd data/input/sensor_data.csv")
    ap.add_argument("--time_col", required=True, help="Tên cột thời gian")
    ap.add_argument("--targets", nargs="*", default=None, help="Các cột số cần xử lý")
    ap.add_argument("--freq", default=None, help="Tần suất pandas, vd 1min, 1H, 1D")
    ap.add_argument("--contamination", type=float, default=0.02, help="Tỉ lệ outlier ước tính")
    ap.add_argument("--top", type=int, default=10, help="Số dòng nhiễu in ra terminal")
    ap.add_argument("--out_dir", default="outputs", help="Thư mục gốc đầu ra")
    args = ap.parse_args()

    t0 = time.perf_counter()
    path = Path(args.input)
    if not path.is_file():
        print(f"[LỖI] Không tìm thấy file: {path}", file=sys.stderr)
        return 1
    try:
        name = path.stem  # sensor_data.csv -> sensor_data
        out = Path(args.out_dir) / name
        out.mkdir(parents=True, exist_ok=True)

        df = preprocess_timeseries(read_table(path), args.time_col, args.targets)
        pipe = TimeSeriesPipeline(freq=args.freq, contamination=args.contamination)
        clean, report = pipe.run(df, list(df.columns))

        clean.to_csv(out / "data_clean.csv")
        report.to_csv(out / "data_report.csv")
        pipe.plot(clean, report, str(out / "ts_health_report.png"))

        # outliers.csv: chỉ các dòng nhiễu, xếp theo anomaly_score giảm dần
        top = report[report["is_outlier"]].sort_values("anomaly_score", ascending=False)
        top.insert(0, "row_number", report.index.get_indexer(top.index) + 1)
        top.to_csv(out / "outliers.csv")
    except Exception as exc:
        print(f"[LỖI] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    s = pipe.summary_
    print("=" * 52)
    print(f" File           : {path.name}")
    print(f" Engine         : {s['engine']}")
    print(f" Tần suất       : {s['freq']}")
    print(f" Số dòng        : {s['rows_in']} -> {s['rows_out']} (khôi phục {s['restored_rows']})")
    print(f" NaN đã nội suy : {s['nan_filled']} (còn lại: {s['remaining_nan']})")
    print(f" Mốc trùng gộp  : {s['duplicate_timestamps']}")
    print(f" Outlier        : {s['outliers']}")
    print(f" Đầu ra         : {out}")
    if len(top):
        cols = ["row_number", "anomaly_score"] + [
            f"{c}_{k}" for c in pipe._targets[:3] for k in ("raw", "clean")]
        print(f" Top {min(args.top, len(top))} dòng nhiễu (row_number = số thứ tự trong data_clean.csv):")
        print(top[cols].head(args.top).round(3).to_string())
    print(f" Thời gian chạy : {time.perf_counter() - t0:.2f}s")
    print("=" * 52)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
pipeline.py - Lõi xử lý chuỗi thời gian (cố định, không cần sửa theo dataset).

Phối hợp 4 thư viện theo kiểu "dùng nếu có, có fallback nếu không":
  * PyOD  : ECOD (Empirical-Cumulative-distribution Outlier Detection) -> anomaly_score
  * Darts / sktime : ý tưởng resample + fill missing values (cài đặt bằng Pandas để nhanh < 2s)
  * Merlion: phong cách vẽ biểu đồ (overlay + vùng đỏ nhạt + anomaly score)

Lưu ý: import darts/merlion/sktime mất vài giây, nên lõi tính toán dùng NumPy/Pandas
và chỉ import PyOD (nhẹ). Nếu PyOD lỗi/không có, tự động rơi về ECOD bản NumPy thuần.
"""
from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# ECOD fallback (NumPy) - dùng khi PyOD không import được (vd. thiếu numba)
# --------------------------------------------------------------------------- #
def _ecod_numpy(X: np.ndarray) -> np.ndarray:
    """ECOD rút gọn: score = tổng -log(xác suất đuôi) theo từng chiều (đuôi trái/phải/skew)."""
    n, d = X.shape
    # Hạng (rank) -> ECDF trái và phải
    ranks = np.argsort(np.argsort(X, axis=0), axis=0) + 1.0
    left = -np.log(ranks / n)                 # đuôi trái
    right = -np.log((n - ranks + 1.0) / n)    # đuôi phải
    # Độ lệch (skewness) quyết định dùng đuôi nào cho từng chiều
    mu, sd = X.mean(0), X.std(0) + 1e-12
    skew = (((X - mu) / sd) ** 3).mean(0)
    skewed = np.where(skew < 0, left, right)
    # Lấy max giữa 3 loại điểm số rồi cộng qua các chiều
    return np.maximum.reduce([left, right, skewed]).sum(axis=1)


def _score_ecod(X: np.ndarray, contamination: float) -> Tuple[np.ndarray, np.ndarray, str]:
    """Trả về (scores, labels, tên_engine). Ưu tiên PyOD ECOD."""
    try:
        from pyod.models.ecod import ECOD  # type: ignore

        model = ECOD(contamination=contamination)
        model.fit(X)
        return model.decision_scores_, model.labels_.astype(bool), "pyod.ECOD"
    except Exception as exc:  # ImportError, thiếu numba, v.v.
        warnings.warn(f"PyOD không khả dụng ({type(exc).__name__}); dùng ECOD NumPy.")
        scores = _ecod_numpy(X)
        thr = np.quantile(scores, 1.0 - contamination)
        return scores, scores > thr, "numpy.ECOD"


# --------------------------------------------------------------------------- #
class TimeSeriesPipeline:
    """Làm sạch + phát hiện bất thường cho chuỗi thời gian đa biến, KHÔNG xóa dòng."""

    def __init__(
        self,
        freq: Optional[str] = None,
        contamination: float = 0.02,
        lower_q: float = 0.01,
        upper_q: float = 0.99,
        smooth_window: int = 3,
    ):
        if not 0 < contamination < 0.5:
            raise ValueError("contamination phải nằm trong (0, 0.5)")
        if not 0 <= lower_q < upper_q <= 1:
            raise ValueError("Cần 0 <= lower_q < upper_q <= 1")
        self.freq = freq
        self.contamination = contamination
        self.lower_q, self.upper_q = lower_q, upper_q
        self.smooth_window = max(1, int(smooth_window))
        self.engine_ = "n/a"
        self.summary_: dict = {}

    # ---- 1. Resample & căn mốc thời gian ---------------------------------- #
    def _align(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame phải có DatetimeIndex (preprocess.py đã đặt sẵn).")
        df = df[~df.index.isna()].sort_index()  # sort_index theo thời gian
        n_dup = int(df.index.duplicated().sum())
        if n_dup:  # mốc trùng -> gộp bằng trung bình để trục thời gian duy nhất
            df = df.groupby(level=0).mean()
        self.summary_["duplicate_timestamps"] = n_dup

        freq = self.freq
        if freq is None and len(df) >= 3:
            freq = pd.infer_freq(df.index)
            if freq is None:  # không suy ra được -> dùng khoảng cách phổ biến nhất
                step = df.index.to_series().diff().mode()
                freq = step.iloc[0] if len(step) else None
        self.summary_["freq"] = str(freq)
        if freq is None:
            return df
        full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
        # Giữ cả mốc gốc không khớp lưới (nếu có) để không mất dữ liệu
        return df.reindex(df.index.union(full_idx)) if not df.index.isin(full_idx).all() \
            else df.reindex(full_idx)

    # ---- 2. Nội suy theo thời gian ---------------------------------------- #
    @staticmethod
    def _impute(df: pd.DataFrame) -> pd.DataFrame:
        out = df.interpolate(method="time", limit_direction="both")
        return out.ffill().bfill()  # đảm bảo 0 NaN (kể cả đầu/cuối chuỗi)

    # ---- 4. Làm mịn + Winsorization -------------------------------------- #
    def _smooth_clip(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Trả về (bản sạch, mặt nạ các ô thực sự bị Winsorize cắt đỉnh)."""
        out = df
        if self.smooth_window > 1:  # median lăn: khử spike nhỏ mà giữ biên độ tín hiệu
            out = out.rolling(self.smooth_window, center=True, min_periods=1).median()
        lo = out.quantile(self.lower_q)
        hi = out.quantile(self.upper_q)
        clipped = out.clip(lower=lo, upper=hi, axis=1)  # ép về [P1, P99], không drop dòng
        return clipped, (clipped.to_numpy() != out.to_numpy())

    # ---- API chính -------------------------------------------------------- #
    def run(self, df: pd.DataFrame, targets: Optional[List[str]] = None
            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Trả về (data_clean, data_report). Cùng index, cùng số dòng."""
        if df is None or df.empty:
            raise ValueError("DataFrame đầu vào rỗng.")
        cols = targets or df.select_dtypes("number").columns.tolist()
        if not cols:
            raise ValueError("Không có cột số nào để xử lý.")
        raw = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

        aligned = self._align(raw)
        all_nan = [c for c in cols if aligned[c].isna().all()]
        if all_nan:
            raise ValueError(f"Cột toàn NaN, không thể nội suy: {all_nan}")
        was_missing = aligned.isna().any(axis=1)
        filled = self._impute(aligned)

        # Phát hiện bất thường trên dữ liệu đã bù (chuẩn hóa theo cột)
        X = filled.to_numpy(dtype=float)
        X = (X - X.mean(0)) / (X.std(0) + 1e-12)
        scores, labels, self.engine_ = _score_ecod(X, self.contamination)

        clean, winsorized = self._smooth_clip(filled)

        # Cờ từng cột:
        #   _winsorized : ô bị cắt đỉnh về [P1, P99]
        #   _corrected  : ô bị cắt đỉnh HOẶC (dòng outlier và giá trị bị thay đổi);
        #                 làm mịn median ở dòng bình thường không tính là "sửa lỗi"
        changed = ~np.isclose(filled.to_numpy(), clean.to_numpy(), rtol=1e-9, atol=1e-12)
        corrected = winsorized | (changed & labels[:, None])
        report = pd.DataFrame(index=filled.index)
        report["anomaly_score"] = scores
        report["is_outlier"] = labels
        report["was_imputed"] = was_missing.to_numpy()
        for j, c in enumerate(cols):
            report[f"{c}_raw"] = aligned[c].to_numpy()
            report[f"{c}_clean"] = clean[c].to_numpy()
            report[f"{c}_winsorized"] = winsorized[:, j]
            report[f"{c}_corrected"] = corrected[:, j]
        self._targets = cols

        self.summary_.update(
            rows_in=len(raw), rows_out=len(clean), restored_rows=len(clean) - len(raw),
            nan_filled=int(aligned.isna().sum().sum()),
            outliers=int(labels.sum()), engine=self.engine_,
            remaining_nan=int(clean.isna().sum().sum()),
        )
        clean.index.name = report.index.name = raw.index.name or "timestamp"
        self._raw_aligned = aligned
        return clean, report

    # ---- 5. Trực quan hóa (phong cách Merlion) --------------------------- #
    def plot(self, clean: pd.DataFrame, report: pd.DataFrame, path: str) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cols = list(clean.columns)
        fig, axes = plt.subplots(len(cols) + 1, 1, figsize=(13, 2.6 * (len(cols) + 1)),
                                 sharex=True)
        axes = np.atleast_1d(axes)
        idx = clean.index
        out = report["is_outlier"].to_numpy()
        # Gom các điểm outlier liên tiếp thành vùng để tô cho gọn
        spans, start = [], None
        for i, flag in enumerate(out):
            if flag and start is None:
                start = i
            if (not flag or i == len(out) - 1) and start is not None:
                end = i if flag else i - 1
                spans.append((idx[start], idx[end])); start = None
        step = (idx[1] - idx[0]) if len(idx) > 1 else pd.Timedelta(0)

        # Vùng đã nội suy (NaN / mốc mất) -> tô màu cam nhạt, tách khỏi vùng outlier (đỏ)
        imp = report["was_imputed"].to_numpy()
        imp_spans, start = [], None
        for i, flag in enumerate(imp):
            if flag and start is None:
                start = i
            if (not flag or i == len(imp) - 1) and start is not None:
                end = i if flag else i - 1
                imp_spans.append((idx[start], idx[end])); start = None

        for ax, c in zip(axes[:-1], cols):
            ax.plot(idx, self._raw_aligned[c], "--", color="gray", lw=1, label="Gốc")
            ax.plot(idx, clean[c], color="tab:blue", lw=1.3, label="Sạch")
            for s, e in imp_spans:
                ax.axvspan(s - step / 2, e + step / 2, color="orange", alpha=0.25, lw=0)
            for s, e in spans:
                ax.axvspan(s - step / 2, e + step / 2, color="red", alpha=0.18, lw=0)
            # Đánh dấu chấm đỏ tại điểm outlier trên giá trị gốc
            ax.scatter(idx[out], self._raw_aligned[c].to_numpy()[out], s=14, color="red",
                       zorder=3)
            ax.set_ylabel(c)
            ax.legend(handles=ax.get_lines()[:2] + [
                plt.Rectangle((0, 0), 1, 1, color="red", alpha=0.3, label="Outlier"),
                plt.Rectangle((0, 0), 1, 1, color="orange", alpha=0.4, label="Đã nội suy")],
                loc="upper right", fontsize=8)
        axes[-1].plot(idx, report["anomaly_score"], color="darkred", lw=1)
        axes[-1].set_ylabel("anomaly_score")
        axes[-1].set_xlabel("Thời gian")
        axes[0].set_title(f"Time-series health report  (engine: {self.engine_})")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)

"""
preprocess.py - File TÙY BIẾN theo từng dataset chuỗi thời gian.

HƯỚNG DẪN SỬA CHO TỪNG LOẠI DỮ LIỆU
-----------------------------------
Chỉ sửa file này; pipeline.py và run.py giữ nguyên. Hàm phải trả về DataFrame có
DatetimeIndex + các cột số. Các tình huống thường gặp:

1. Định dạng thời gian lạ  (vd "15/03/2024 14:05", "20240315", epoch)
   -> Sửa dòng pd.to_datetime ở BƯỚC 1:
        pd.to_datetime(df[c], format="%d/%m/%Y %H:%M", errors="coerce")   # chuỗi
        pd.to_datetime(df[c], unit="s",  errors="coerce")                 # epoch giây
        pd.to_datetime(df[c], unit="ms", errors="coerce")                 # epoch mili-giây
   -> Ngày tháng kiểu Việt/Âu (ngày trước tháng): thêm dayfirst=True.

2. Ngày và giờ nằm ở HAI cột riêng (Date, Time)
   -> Ghép trước BƯỚC 1:
        df["ts"] = df["Date"].astype(str) + " " + df["Time"].astype(str); time_col = "ts"

3. Có nhiều thiết bị / sensor / mã cổ phiếu trong cùng một file (cột id)
   -> Pipeline xử lý MỘT chuỗi mỗi lần. Lọc một id, hoặc pivot thành nhiều cột:
        df = df[df["sensor_id"] == "A1"]
        df = df.pivot_table(index=time_col, columns="sensor_id", values="value")

4. Số dùng dấu phẩy thập phân ("12,5") hoặc có đơn vị ("23.4°C", "1,200")
   -> Làm sạch chuỗi trước BƯỚC 3:
        df[c] = df[c].astype(str).str.replace(",", ".", regex=False).str.extract(r"(-?\d+\.?\d*)")[0]

5. Giá trị đặc biệt thay cho dữ liệu thiếu (-999, 9999, "N/A", "--", 0 do mất tín hiệu)
   -> Đổi sang NaN để pipeline tự nội suy:
        df[cols] = df[cols].replace([-999, 9999], np.nan)    # nhớ import numpy as np
        # với 0 chỉ làm khi 0 chắc chắn là lỗi: df[c] = df[c].replace(0, np.nan)

6. Múi giờ (UTC vs giờ địa phương)
   -> BƯỚC 1 thêm: pd.to_datetime(..., utc=True).dt.tz_convert("Asia/Ho_Chi_Minh").dt.tz_localize(None)

7. Dữ liệu tài chính (OHLC) / giá chỉ có ngày làm việc
   -> Dùng --freq B (business day); cột target nên là Close/Volume. Cuối tuần là "mốc
      mất" hợp lệ, nên cân nhắc --freq B thay vì D để không nội suy cả cuối tuần.

8. Dữ liệu IoT/sensor tần số cao, rất lớn
   -> Resample thô trước khi vào pipeline (BƯỚC 4):
        out = out.resample("1min").mean()

9. Có cột phân loại (trạng thái, nhãn)
   -> Không đưa vào targets. Pipeline chỉ nhận cột số; cột chữ sẽ bị bỏ ở bước tự chọn.

10. File Excel / JSON / định dạng khác
   -> Đọc ở run.py (hàm read_table) thêm nhánh mới, ví dụ pd.read_excel(path).
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd


def preprocess_timeseries(df: pd.DataFrame, time_col: str,
                          target_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Chuẩn hóa df: cột thời gian -> DatetimeIndex, chọn các cột số cần lọc."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]  # bỏ khoảng trắng thừa trong tên cột
    time_col = time_col.strip()
    if time_col not in df.columns:
        raise KeyError(f"Không thấy cột thời gian '{time_col}'. Cột hiện có: {list(df.columns)}")

    # ===== BƯỚC 1: chuyển cột thời gian sang datetime (xem mục 1, 2, 6 ở docstring) =====
    # Giá trị không parse được -> NaT rồi loại (dòng không có mốc giờ không thể đặt lên
    # trục thời gian). Đây là bước duy nhất bỏ dòng.
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    bad = int(df[time_col].isna().sum())
    if bad:
        print(f"[preprocess] Bỏ {bad} dòng có timestamp không hợp lệ.")
        df = df.dropna(subset=[time_col])
    if df.empty:
        raise ValueError("Không còn dòng nào có timestamp hợp lệ.")

    # ===== BƯỚC 2: chọn cột chỉ số (xem mục 3, 9) =====
    if target_cols:
        target_cols = [c.strip() for c in target_cols]
        missing = [c for c in target_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Cột target không tồn tại: {missing}")
        cols = target_cols
    else:  # tự chọn: mọi cột có thể ép sang số (trừ cột thời gian)
        cols = []
        for c in df.columns.drop(time_col):
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().mean() > 0.5:
                cols.append(c)
        if not cols:
            raise ValueError("Không tìm thấy cột số nào; hãy truyền --targets.")

    # ===== BƯỚC 3: ép kiểu số, làm sạch giá trị lạ (xem mục 4, 5) =====
    out = df[[time_col] + cols].set_index(time_col)
    out[cols] = out[cols].apply(pd.to_numeric, errors="coerce")

    # ===== BƯỚC 4: tuỳ chọn resample thô (xem mục 8) =====
    return out.sort_index()

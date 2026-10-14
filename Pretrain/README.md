# Pretrain – Time-Series Cleaning & Anomaly Pipeline

Tiền xử lý, sửa lỗi và trực quan hóa **dữ liệu chuỗi thời gian** (đơn/đa biến): khôi phục mốc
thời gian bị mất, nội suy NaN, phát hiện bất thường, làm mịn/cắt đỉnh nhiễu, xuất báo cáo.
**Chạy 100% local, không gọi API ngoài, không tải trọng số pretrained.**

## 1. Cấu trúc

```
D:\PretrainLiner\              # thư mục gốc workspace
├── requirements.txt           # danh sách thư viện (cài từ thư mục này)
├── .venv\                     # môi trường ảo (tự tạo ở mục 2)
└── Pretrain\                  # thư mục dự án - CHẠY LỆNH TỪ ĐÂY
    ├── data\input\            # đặt file CSV/Parquet vào đây
    ├── preprocess.py          # SỬA file này cho từng dataset (đọc timestamp, chọn cột)
    ├── pipeline.py            # lõi xử lý (cố định)
    ├── run.py                 # CLI chạy chính
    └── outputs\<tên_file>\
        ├── data_clean.csv          # chuỗi sạch, đủ mốc, 0 NaN
        ├── data_report.csv         # mọi dòng: anomaly_score, is_outlier, was_imputed,
        │                           #   <cột>_raw, <cột>_clean, <cột>_winsorized, <cột>_corrected
        ├── outliers.csv            # chỉ các dòng nhiễu, xếp theo anomaly_score giảm dần (có row_number)
        └── ts_health_report.png    # Gốc (nét đứt) vs Sạch; đỏ = outlier (+chấm đỏ), cam = vùng nội suy;
                                    #   panel cuối = anomaly score
```

## 2. Cài môi trường

Yêu cầu Python 3.9 – 3.12 (đã kiểm thử trên 3.9.13). Các thư viện được khoá đúng phiên bản
đã chạy thành công trong `requirements.txt`.

Chạy từ thư mục gốc `D:\PretrainLiner` (nơi có `requirements.txt`):

```bash
cd D:\PretrainLiner
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

`requirements.txt` (ở thư mục gốc) gồm: `numpy`, `pandas`, `matplotlib`, `pyarrow` (Parquet),
`pyod` + `numba` + `scipy` + `scikit-learn` + `joblib` (ECOD). `darts`, `sktime`,
`salesforce-merlion` **không bắt buộc**, đã để sẵn dạng comment trong file (xem mục 4).

Kiểm tra nhanh môi trường: `python -c "from pyod.models.ecod import ECOD; print('OK')"`.

## 3. Cách chạy

`run.py` nằm trong `Pretrain\`, nên **phải đứng trong thư mục `Pretrain`** (các đường dẫn
`data/input/...` và `outputs/` tính theo thư mục đang đứng):

```bash
cd D:\PretrainLiner\Pretrain
D:\PretrainLiner\.venv\Scripts\activate          # nếu đã cài môi trường ảo ở mục 2
python run.py data/input/sensor_data.csv --time_col time --targets temp humidity --freq 1min
```

Kết quả nằm ở `Pretrain\outputs\sensor_data\`.

Muốn chạy từ thư mục gốc `D:\PretrainLiner` thì trỏ đường dẫn đầy đủ:

```bash
python Pretrain\run.py Pretrain\data\input\sensor_data.csv --time_col time --targets temp humidity --freq 1min --out_dir Pretrain\outputs
```

Lỗi `can't open file '...\run.py'` nghĩa là bạn đang đứng sai thư mục, hãy `cd` vào `Pretrain`.

| Tham số | Ý nghĩa |
|---|---|
| `input` | file CSV/Parquet |
| `--time_col` | tên cột thời gian (bắt buộc) |
| `--targets` | các cột số cần xử lý; bỏ trống = tự chọn mọi cột số |
| `--freq` | tần suất pandas: `1min`, `1H`, `1D`, `B`…; bỏ trống = tự suy ra |
| `--contamination` | tỉ lệ outlier ước tính (mặc định 0.02) |
| `--top` | số dòng nhiễu in ra terminal (mặc định 10) |
| `--out_dir` | thư mục gốc đầu ra (mặc định `outputs`) |

Cuối lệnh in tổng kết (số dòng khôi phục, NaN đã nội suy, số outlier, thời gian chạy).

## 4. Kiến trúc & luồng xử lý

```
CSV/Parquet ─► run.py ─► preprocess.py ─► pipeline.py ─► outputs/<tên_file>/
                         (datetime index,   1 sort_index + gộp mốc trùng
                          chọn cột số)      2 reindex theo freq (khôi phục mốc mất)
                                            3 nội suy method="time" + ffill/bfill
                                            4 ECOD (PyOD) -> anomaly_score, is_outlier
                                            5 rolling median + Winsorize [P1,P99]
                                            6 vẽ biểu đồ
```

- **Không bao giờ `drop()` dòng** trong pipeline – trục thời gian luôn liên tục
  (chỉ `preprocess.py` bỏ dòng có timestamp không parse được).
- Anomaly score tính trên dữ liệu **đã nội suy, chuẩn hóa theo cột**, nên các điểm NaN
  không gây nhiễu.
- Vai trò các thư viện: **PyOD** = bộ phát hiện (ECOD) thật sự dùng; **darts / sktime**
  = ý tưởng resample + fill missing, được cài bằng Pandas để chạy < 2 giây (import các
  thư viện này mất vài giây); **Merlion** = phong cách biểu đồ. Cần dùng trực tiếp các
  thư viện đó thì thay vào `_align` / `_impute` / `_score_ecod` trong `pipeline.py`.

## 5. Mô hình chỉ dùng cho luồng dữ liệu nào

| Thành phần | Dùng cho | Không phù hợp |
|---|---|---|
| ECOD (PyOD, không giám sát, không tham số) | Chuỗi **số liên tục** đơn/đa biến: cảm biến IoT, log hệ thống, giá, lưu lượng… phát hiện spike/điểm ngoại lai về **biên độ** | Dữ liệu phân loại/chữ; bất thường kiểu **hình dạng/chu kỳ** (đổi pha, mất mùa vụ) vì ECOD không xét thứ tự thời gian; chuỗi quá ngắn (< ~50 điểm) |
| Nội suy `time` + ffill/bfill | Dữ liệu lấy mẫu đều có khoảng mất vừa phải | Khoảng mất rất dài (nội suy sẽ ra đường thẳng giả) |
| Winsorize P1/P99 + median 3 | Khử nhiễu xung/spike đơn lẻ | Dữ liệu mà đỉnh lớn là tín hiệu thật (sự kiện hiếm cần giữ) → tăng `lower_q/upper_q` hoặc đặt `smooth_window=1` |

Dữ liệu nhiều thiết bị trong một file: xử lý **từng thiết bị một** hoặc pivot thành cột
(xem docstring `preprocess.py`, mục 3).

## 6. Offline / trọng số

Đã kiểm tra: pipeline **không có trọng số pretrained, không gọi API/HTTP, không tải model**.
ECOD là thuật toán thống kê không giám sát, fit trực tiếp trên dữ liệu mỗi lần chạy, nên
không có gì để lưu. Ngoài ra `run.py` gọi `_block_network()` để **chặn mọi kết nối mạng**;
nếu thư viện nào cố gọi ra ngoài sẽ báo lỗi `Network bị chặn` thay vì âm thầm kết nối.
Nếu sau này thêm mô hình có trọng số, hãy tải về một lần, lưu vào `Pretrain/weights/`,
và load từ đường dẫn local.

## 7. Cách sửa lỗi thường gặp

| Triệu chứng | Nguyên nhân / cách fix |
|---|---|
| `Không thấy cột thời gian 'x'` | Sai tên `--time_col` (phân biệt hoa/thường); xem danh sách cột trong thông báo |
| `Bỏ N dòng có timestamp không hợp lệ` quá nhiều | Sai định dạng ngày → chỉnh `pd.to_datetime(..., format=...)` trong `preprocess.py` (mục 1) |
| Số dòng tăng vọt sau khi chạy | `--freq` nhỏ hơn tần suất thật (vd dữ liệu theo giờ nhưng `--freq 1min`) → sửa `--freq` |
| `Cột toàn NaN` | Cột chứa chữ/đơn vị bị ép thành NaN → làm sạch chuỗi (mục 4 trong `preprocess.py`) |
| Quá nhiều/ít outlier | Chỉnh `--contamination` |
| Đỉnh thật bị cắt phẳng | Tăng `upper_q`/giảm `lower_q` hoặc `smooth_window=1` trong `TimeSeriesPipeline(...)` |
| Warning `PyOD không khả dụng` | Thiếu `numba`: `pip install numba`. Pipeline vẫn chạy bằng ECOD NumPy tương đương |
| `UnicodeEncodeError` trên Windows | `run.py` đã ép UTF-8; nếu vẫn lỗi: `set PYTHONUTF8=1` |
| `Network bị chặn` | Thư viện đang cố gọi mạng; xác định dòng gây ra ở traceback, bỏ/cấu hình offline |
| Chạy > 2 giây | Dữ liệu lớn (> vài trăm nghìn dòng) → resample thô ở `preprocess.py` (mục 8) |

## 8. Thông số có thể chỉnh trong `TimeSeriesPipeline(...)`

`freq`, `contamination` (0.02), `lower_q` (0.01), `upper_q` (0.99), `smooth_window` (3; 1 = tắt làm mịn).

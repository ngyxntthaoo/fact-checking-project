# Hướng Dẫn Sử Dụng Vast.ai để Huấn Luyện Concept-HGN

> **Dự án:** Concept-HGN — Mô hình kiểm tra thực tế dựa trên đồ thị dị biệt (Heterogeneous Graph Network) trên tập dữ liệu FEVER  
> **Mục tiêu:** Huấn luyện đầy đủ 10 epochs để đạt kết quả trong bài báo (~76–78% Label Accuracy)

---

## Mục Lục

1. [Yêu cầu trước khi bắt đầu](#1-yêu-cầu-trước-khi-bắt-đầu)
2. [Đăng ký và cài đặt Vast.ai](#2-đăng-ký-và-cài-đặt-vastai)
3. [Chọn GPU phù hợp](#3-chọn-gpu-phù-hợp)
4. [Tạo và cấu hình Instance](#4-tạo-và-cấu-hình-instance)
5. [Kết nối SSH vào Instance](#5-kết-nối-ssh-vào-instance)
6. [Clone code từ GitHub](#6-clone-code-từ-github)
7. [Upload dữ liệu lên Instance](#7-upload-dữ-liệu-lên-instance)
8. [Cài đặt môi trường](#8-cài-đặt-môi-trường)
9. [Chạy huấn luyện đầy đủ](#9-chạy-huấn-luyện-đầy-đủ)
10. [Theo dõi tiến trình](#10-theo-dõi-tiến-trình)
11. [Tải checkpoint về máy](#11-tải-checkpoint-về-máy)
12. [Lưu ý tiết kiệm chi phí](#12-lưu-ý-tiết-kiệm-chi-phí)
13. [Xử lý lỗi thường gặp](#13-xử-lý-lỗi-thường-gặp)

---

## 1. Yêu Cầu Trước Khi Bắt Đầu

### Trên máy tính cá nhân (MacBook/PC)

- [ ] Code đã được push lên GitHub: `https://github.com/ngyxntthaoo/fact-checking-project`
- [ ] Có sẵn file dữ liệu FEVER:
  - `KernelGAT/data/KernelGAT/data/all_train.json` (~145K mẫu)
  - `KernelGAT/data/KernelGAT/data/all_dev.json` (~20K mẫu)
- [ ] Có sẵn file `HeterFC/concept-hgn/concept_cache.pkl`
- [ ] Có tài khoản Vast.ai và đã nạp tối thiểu **$10** (đủ cho ~1 lần huấn luyện đầy đủ)

### Ước tính chi phí

| GPU | Giá/giờ | Thời gian 10 epochs | **Tổng chi phí** |
|-----|---------|---------------------|-----------------|
| RTX 3090 (24 GB) | ~$0.10–0.20 | ~12–18 giờ | **~$2–4** ✅ |
| RTX 4090 (24 GB) | ~$0.15–0.60 | ~8–12 giờ | **~$3–6** ✅ |
| A100 (80 GB) | ~$1–4 | ~4–6 giờ | **~$6–20** ❌ |

> **Khuyến nghị:** Chọn **RTX 3090** — rẻ nhất, đủ VRAM (24 GB >> 8 GB cần thiết).

---

## 2. Đăng Ký và Cài Đặt Vast.ai

### Bước 2.1 — Tạo tài khoản

1. Truy cập [https://vast.ai](https://vast.ai) → Click **"Sign Up"**
2. Đăng ký bằng email hoặc Google
3. Vào **Billing** → **Add Credit** → Nạp tối thiểu $10 bằng thẻ tín dụng

### Bước 2.2 — Thêm SSH Public Key

Vast.ai dùng SSH key để kết nối — **không dùng mật khẩu**.

```bash
# Trên máy tính cá nhân, kiểm tra xem đã có SSH key chưa:
cat ~/.ssh/id_rsa.pub

# Nếu chưa có, tạo mới:
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
# Nhấn Enter 3 lần (để mặc định, không đặt passphrase)

# Copy nội dung public key:
cat ~/.ssh/id_rsa.pub
```

Sau đó:
1. Vào **Vast.ai** → **Account** → **SSH Keys**
2. Click **"Add SSH Key"** → Dán nội dung vừa copy → **Save**

---

## 3. Chọn GPU Phù Hợp

### Truy cập trang tìm kiếm GPU

1. Vào [https://vast.ai/console/create/](https://vast.ai/console/create/)
2. Cấu hình bộ lọc như sau:

```
GPU Type:   RTX 3090  (hoặc RTX 4090 nếu muốn nhanh hơn)
Min VRAM:   24 GB
Disk Space: ≥ 30 GB
GPU Count:  1
```

3. Sắp xếp theo **"Price $/hr"** từ thấp đến cao
4. Chọn instance có:
   - ✅ VRAM ≥ 24 GB
   - ✅ Disk ≥ 30 GB  
   - ✅ Upload speed ≥ 200 Mbps (để upload data nhanh)
   - ✅ **"On-demand"** (ổn định hơn "Interruptible" cho training dài)

> ⚠️ **Lưu ý:** Chọn **"On-demand"** thay vì **"Interruptible/Spot"** vì training mất 12–18 giờ. Instance Spot có thể bị dừng giữa chừng.

---

## 4. Tạo và Cấu Hình Instance

### Bước 4.1 — Chọn Docker Image

Khi tạo instance, ở phần **"Select Image"**, chọn:

```
pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
```

Hoặc tìm kiếm: `pytorch cuda`

> Đây là image chuẩn có sẵn CUDA + PyTorch, không cần cài thêm nhiều.

### Bước 4.2 — Cấu hình Storage

- **Disk**: Đặt tối thiểu **30 GB** (data ~5 GB + model ~1.4 GB + cache + checkpoints)

### Bước 4.3 — Tạo Instance

Click **"Rent"** → Xác nhận → Chờ instance khởi động (~1–3 phút).

---

## 5. Kết Nối SSH vào Instance

### Bước 5.1 — Lấy thông tin SSH

Sau khi instance chạy, vào **"Instances"** → Click instance vừa tạo → Xem:

```
SSH Command: ssh -p <PORT> root@<IP_ADDRESS>
```

Ví dụ:
```
ssh -p 12345 root@123.456.789.0
```

### Bước 5.2 — Kết nối

```bash
# Trên máy tính cá nhân:
ssh -p 12345 root@123.456.789.0

# Lần đầu sẽ hỏi "Are you sure you want to continue connecting?" → gõ: yes
```

Nếu kết nối thành công, bạn sẽ thấy terminal của instance vast.ai:
```
root@container:~#
```

---

## 6. Clone Code từ GitHub

```bash
# Trong terminal SSH của vast.ai:

# Clone repo
git clone https://github.com/ngyxntthaoo/fact-checking-project.git
cd fact-checking-project

# Kiểm tra cấu trúc
ls -la
# Kết quả mong đợi:
# HeterFC/   KernelGAT/   setup_vast.sh   run_train.sh   .gitignore
```

---

## 7. Upload Dữ Liệu Lên Instance

> ⚠️ **Quan trọng:** Các file dữ liệu được gitignore — không có trong repo. Phải upload thủ công từ máy local.

### Mở terminal MỚI trên máy tính cá nhân (không đóng SSH)

```bash
# Terminal mới trên MacBook — KHÔNG phải terminal SSH

# Di chuyển đến thư mục project
cd "/Users/thnhthao/Master 2025/Thesis/fact-checking-project"

# Thay thế PORT và IP bằng thông tin thực của instance:
PORT=12345
IP=123.456.789.0

# 1. Upload thư mục dữ liệu FEVER (~2–3 GB, mất vài phút)
scp -P $PORT -r \
  "KernelGAT/data/" \
  root@$IP:/root/fact-checking-project/KernelGAT/

# 2. Upload concept cache (~120 KB)
scp -P $PORT \
  "HeterFC/concept-hgn/concept_cache.pkl" \
  root@$IP:/root/fact-checking-project/HeterFC/concept-hgn/

echo "Upload hoàn tất!"
```

### Xác nhận dữ liệu đã upload (trong terminal SSH)

```bash
# Quay lại terminal SSH:
ls KernelGAT/data/KernelGAT/data/
# Kết quả mong đợi: all_dev.json  all_train.json  ...

ls HeterFC/concept-hgn/concept_cache.pkl
# Kết quả mong đợi: HeterFC/concept-hgn/concept_cache.pkl
```

---

## 8. Cài Đặt Môi Trường

```bash
# Trong terminal SSH:
cd /root/fact-checking-project

# Cấp quyền thực thi cho scripts
chmod +x setup_vast.sh run_train.sh

# Chạy script cài đặt tự động
./setup_vast.sh
```

Script này sẽ tự động:
1. Cài `torch`, `torch-geometric`, `torch-scatter`, `torch-sparse` (phiên bản CUDA)
2. Cài các thư viện trong `requirements.txt`
3. Tải mô hình spaCy `en_core_web_lg`
4. Kiểm tra CUDA và các file dữ liệu

### Kết quả mong đợi cuối script:

```
[6/6] Setup complete!
  ✓ all_train.json found
  ✓ all_dev.json found
  ✓ concept_cache.pkl found
  CUDA available: True
  GPU: NVIDIA GeForce RTX 3090
```

> ⚠️ Nếu CUDA = False: Instance chưa nhận GPU — thử khởi động lại instance trên Vast.ai.

---

## 9. Chạy Huấn Luyện Đầy Đủ

### Cách 1 — Dùng script (khuyến nghị)

```bash
cd /root/fact-checking-project
./run_train.sh
```

### Cách 2 — Chạy thủ công với tùy chỉnh

```bash
cd /root/fact-checking-project/HeterFC/concept-hgn

# Full training (tắt smoke test)
SMOKE_TEST=0 python train_fever.py --no-smoke
```

### Cách 3 — Chạy trong background (không bị ngắt khi đóng SSH)

```bash
# Dùng tmux để giữ session khi đóng SSH
tmux new-session -s training

# Trong session tmux:
cd /root/fact-checking-project
./run_train.sh

# Để detach (thoát tmux mà không dừng training):
# Nhấn: Ctrl+B, rồi nhấn: D

# Để attach lại sau:
tmux attach-session -t training
```

> ✅ **Luôn dùng `tmux`** khi training dài. Nếu SSH bị ngắt do mạng, training vẫn tiếp tục chạy.

### Output mong đợi khi bắt đầu:

```
Using device: cuda:0
Loading tokenizer and PLM...
Loading FEVER train...
Loading FEVER dev...
Preprocessing train...
Tokenizing: 100%|████| 145449/145449 [02:10<00:00]
Building graphs: 100%|████| 145449/145449 [00:45<00:00]
...
------ Start Training! ------
Epoch 1/10: 100%|████| 145449/145449 [01:23<00:00]
[dev] Label Accuracy: 0.6234 | Macro-F1: 0.5891
✓ Checkpoint saved: checkpoints/concept_hgn_fever.pt
```

---

## 10. Theo Dõi Tiến Trình

### Xem log real-time (trong tmux)

```bash
# Attach lại tmux session:
tmux attach-session -t training
```

### Xem GPU usage

```bash
# Mở tab SSH mới:
watch -n 2 nvidia-smi
```

Kết quả mong đợi:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI ...   Driver Version: ...   CUDA Version: 12.1     |
|-------------------------------|
| GPU  0  RTX 3090              | 60°C  |  95% GPU-Util |
|                               | 18000MiB / 24576MiB           |
```

> GPU Util ~90–100% = đang train tốt. Nếu ~0% = có vấn đề.

### Xem checkpoint đã lưu

```bash
ls -lh /root/fact-checking-project/checkpoints/
# concept_hgn_fever.pt   ~1.4GB
```

---

## 11. Tải Checkpoint Về Máy

Sau khi training hoàn tất (hoặc muốn lưu giữa chừng):

```bash
# Terminal MỚI trên MacBook:
PORT=12345
IP=123.456.789.0

# Tạo thư mục lưu kết quả
mkdir -p "/Users/thnhthao/Master 2025/Thesis/fact-checking-project/checkpoints"

# Download checkpoint
scp -P $PORT \
  root@$IP:/root/fact-checking-project/checkpoints/concept_hgn_fever.pt \
  "/Users/thnhthao/Master 2025/Thesis/fact-checking-project/checkpoints/"

echo "Đã tải về!"
```

---

## 12. Lưu Ý Tiết Kiệm Chi Phí

### ✅ Nên làm

- **Dừng instance ngay** khi training xong — tính tiền theo giờ dù không dùng
- Dùng **tmux** để không phải giữ kết nối SSH liên tục
- Tải checkpoint về **ngay** sau khi training — instance có thể bị xóa
- Kiểm tra **smoke test** trước (`./run_train.sh --smoke`) để chắc code chạy đúng

### ❌ Tránh

- Để instance chạy idle sau khi training xong
- Dùng A100 cho project này (quá đắt, không cần thiết)
- Chọn disk quá nhỏ (< 30 GB) gây lỗi hết dung lượng giữa chừng

### Kiểm tra smoke test trước khi train đầy đủ

```bash
# Chạy 200 mẫu để xác nhận mọi thứ hoạt động đúng (~5 phút):
cd /root/fact-checking-project/HeterFC/concept-hgn
python train_fever.py --smoke

# Nếu thấy "Start Training!" và không có lỗi → an toàn để train đầy đủ
```

---

## 13. Xử Lý Lỗi Thường Gặp

### ❌ `CUDA available: False`

```bash
# Kiểm tra driver GPU:
nvidia-smi

# Nếu lệnh không tìm thấy → Restart instance trên Vast.ai dashboard
# Nếu vẫn không được → Thuê instance khác (host bị lỗi)
```

### ❌ `FileNotFoundError: all_train.json`

```bash
# Kiểm tra đường dẫn:
ls KernelGAT/data/KernelGAT/data/

# Nếu thiếu → Upload lại từ máy local (xem Bước 7)

# Hoặc chỉ định đường dẫn thủ công:
TRAIN_PATH=/root/fact-checking-project/KernelGAT/data/KernelGAT/data/all_train.json \
DEV_PATH=/root/fact-checking-project/KernelGAT/data/KernelGAT/data/all_dev.json \
SMOKE_TEST=0 python train_fever.py --no-smoke
```

### ❌ `CUDA out of memory`

```bash
# Giảm batch size về 1 (đã là mặc định):
TRAIN_BATCH=1 SMOKE_TEST=0 python train_fever.py --no-smoke

# Hoặc xóa preprocessed cache cũ và chạy lại:
rm -rf /root/fact-checking-project/HeterFC/concept-hgn/preprocessed/
```

### ❌ `OSError: Can't find model 'en_core_web_lg'`

```bash
python -m spacy download en_core_web_lg
```

### ❌ `ModuleNotFoundError: No module named 'torch_geometric'`

```bash
pip install torch-geometric torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

### ❌ SSH bị ngắt kết nối giữa training

```bash
# Attach lại tmux (training vẫn đang chạy):
ssh -p PORT root@IP
tmux attach-session -t training
```

---

## Tóm Tắt Quy Trình (Quick Reference)

```bash
# ── Trên MacBook ────────────────────────────────────────────────────
# 1. Upload data (chỉ cần làm 1 lần)
scp -P PORT -r "KernelGAT/data/" root@IP:/root/fact-checking-project/KernelGAT/
scp -P PORT "HeterFC/concept-hgn/concept_cache.pkl" root@IP:/root/fact-checking-project/HeterFC/concept-hgn/

# ── Trong SSH (vast.ai instance) ────────────────────────────────────
# 2. Clone + setup
git clone https://github.com/ngyxntthaoo/fact-checking-project.git
cd fact-checking-project
chmod +x setup_vast.sh run_train.sh
./setup_vast.sh

# 3. Smoke test (kiểm tra nhanh ~5 phút)
cd HeterFC/concept-hgn && python train_fever.py --smoke && cd ../..

# 4. Full training trong tmux
tmux new-session -s training
./run_train.sh
# Ctrl+B, D  ← detach tmux

# 5. Tải checkpoint về (trên MacBook)
scp -P PORT root@IP:/root/fact-checking-project/checkpoints/concept_hgn_fever.pt ./checkpoints/

# 6. Dừng instance trên Vast.ai dashboard để ngừng tính tiền
```

---

## Kết Quả Mục Tiêu (Theo Bài Báo)

| Metric | Giá trị mục tiêu |
|--------|-----------------|
| Label Accuracy | **~76–78%** |
| Macro-F1 | **~72–74%** |
| FEVER Score | **~72–74%** |

> Các kết quả này đạt được sau **10 epochs** huấn luyện đầy đủ trên toàn bộ tập FEVER train (~145K mẫu) với GPU RTX 3090.

---

*Tài liệu này được tạo cho dự án Luận văn Thạc sĩ — Concept-HGN Fact Checking, 2025.*

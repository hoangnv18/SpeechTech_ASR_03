# ASR_Vietnam

## Dataset folder structure
<pre>
|- dataset
        |- Train_set
            |- file1.wav
            |- file1.txt
            |- file2.wav
            |- file2.txt
            |- .......
        |- Validation_set
            |- file10.wav
            |- file10.txt
            |- file20.wav
            |- file20.txt
            |- .......
</pre>

## Setup environemt
```
conda create -n ASR_env python==3.11
```

```
pip install -r requirements.txt
```

```
conda activate ASR_env
```
## Training
```
cd code
python train.py
```

## Inference public_test
```
cd code
python inference_publictest.py
```

## Inference private_test
```
cd code
python inference_privatetest.py
```

## Phân chia công việc
| Thành viên       | Công việc                                               |
|:-----------------|---------------------------------------------------------|
| Đỗ Gia Phúc      | Thu thập dữ liệu âm thanh. Viết code inference          |
| Nguyễn Đức Nhật  | Đào tạo mô hình. Làm pptx, docx                         |
| Nguyễn Văn Hoàng | Viết code pipeline training mô hình. Tổng hợp thông tin |

## Kết quả chạy các model sử dụng

| Mô hình           | Kích thước     | Public test    | Private test   |
|:------------------|----------------|----------------|----------------|
| Whisper large v3  | 1.55B          | 17.66          | 54.03          |
| Pho Whisper small | 244M           | 9.08           | 40.7           |

## Báo cáo word, pptx trong thư mục report


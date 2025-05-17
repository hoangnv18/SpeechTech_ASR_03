# ASR_Vietnam

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

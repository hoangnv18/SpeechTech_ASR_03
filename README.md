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

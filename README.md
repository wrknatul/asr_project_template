# ASR project barebones

## Installation guide

1) Install all required units by shell command:
```shell
pip install -r ./requirements.txt
```
2)To check validation and scores you need to download my checkpoint: https://disk.yandex.ru/d/Oe_5eImX_xkQkQ

3) to run model you should use coommand: 

```shell
python3 test.py -c hw_asr/configs/tran_v2.json -r PATH_TO_DOWNLOADED_CHECKPOINT/model_best.pth
```
change test dataset in config on which you wanna to use

4) If you want to train model use:
```shell
python3 train.py -c hw_asr/configs/tran_v2.json
```
You can also use your own config
5) Thanks for attention!
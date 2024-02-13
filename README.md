# Udop

## `infer.py`
To perform Inference using the Udop model
## `prepare_data.py`
Code to prepare the data for Udop (uses the docvqa_dataset.json)
## `FineTune_udop.py`
code to FineTune the Udop model on the given data.
## `udop_baseline_compare.py`
code to get the udop inference and original answer for the given questions for the image.

## Note:
- docvqa_udop_temp.json : temperory data for udop finetuning. Consists of (95/1500) data.
- docvqa_udop.json : full dataset for udop finetuning, for first image dataset.


## Train
```
python FineTune_udop.py -i '/data/circulars/DATA/udop/docvqa_udop.json' -d '/data/circulars/DATA/split_circulars/SplitCircularsv2/first_page' -e 1 -m 'nielsr/udop-large' -o '/data/circulars/DATA/udop/model_output' -b 1
```
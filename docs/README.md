#Start train
PYTHONPATH=. python ner/train.py --config configs/train.yaml


#Make submition
PYTHONPATH=. python scripts/make_submission_from_val.py \
  --val_csv data/submission.csv \
  --out submission_val.csv \
  --model artifacts/ner-checkpoint \
  --debug

#
PYTHONPATH=. python scripts/eval_preproc_val.py \
  --gold data/val.csv \
  --pred submission_val.csv \
  --types BRAND,TYPE \
  --out diff_report.csv
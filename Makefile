.PHONY: train clean
train:
	python train.py
clean:
	rm -f submission.csv results/model_report.md models/hgb_min.pkl

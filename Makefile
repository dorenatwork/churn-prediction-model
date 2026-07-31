.PHONY: setup data eda train explain test api mlflow-ui

setup:
	pip install -r requirements.txt

data:
	python -m src.data_prep

eda:
	python -m src.eda

train:
	python -m src.train

explain:
	python -m src.explain

test:
	pytest tests/ -v --cov=src

api:
	uvicorn src.serve:app --reload

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

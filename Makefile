ifdef model
MODEL_NAME=$(model)
else
MODEL_NAME=bert
endif

NUM_LABELS=$(num_labels)

export IMAGE_NAME=$(MODEL_NAME)-clf-image
export CONTAINER_NAME=$(MODEL_NAME)-clf-container
export PWD=$(shell pwd)
export PLATFORM=linux/arm64/v8 # if you use M1 Mac, set linux/arm64/v8

# BERT: make docker-build
# RoBERTa: make model=roberta docker-build
# DeBERTa: make model=deberta docker-build
docker-build:
	docker build --target $(MODEL_NAME) -t $(IMAGE_NAME) -f Dockerfile . --platform $(PLATFORM)

# BERT: make num_labels=xxx docker-run
# RoBERTa: make model=roberta num_labels=xxx docker-run
# DeBERTa: make model=deberta num_labels=xxx docker-run
docker-run:
	docker run --rm -it -v $(PWD):/work --platform $(PLATFORM) --name $(CONTAINER_NAME) $(IMAGE_NAME)

train:
	poetry run python3 scripts/train.py --model_name $(MODEL_NAME) --num_labels $(NUM_LABELS)

pytest:
	poetry run pytest

pysen-lint:
	poetry run pysen run lint

pysen-format:
	poetry run pysen run format

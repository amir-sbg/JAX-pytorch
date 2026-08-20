.PHONY: install test train clean

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

train:
	python -m dual_backend.cli train --backend both --epochs 80 --batch-size 32

clean:
	rm -rf artifacts reports .pytest_cache

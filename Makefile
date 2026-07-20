.PHONY: serve build pdf all clean translations figures

serve:
	@bash scripts/serve.sh

build:
	@bash scripts/build.sh html

pdf:
	@bash scripts/build.sh pdf

all:
	@bash scripts/build.sh all

clean:
	@bash scripts/build.sh clean

translations:
	@bash scripts/install-translations.sh

figures:
	@if [ -x .venv/bin/python ]; then \
		.venv/bin/python scripts/generate_linear_algebra_figures.py --format svg; \
	else \
		python3 scripts/generate_linear_algebra_figures.py --format svg; \
	fi

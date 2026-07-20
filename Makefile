.PHONY: serve build pdf all clean translations

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

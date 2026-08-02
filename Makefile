PY        := experiments/re_attn_aae_kaggle.py
IPYNB     := experiments/re_attn_aae_kaggle.ipynb

PY_RSNA      := experiments/re_attn_aae_kaggle-RSNA.py
IPYNB_RSNA   := experiments/re_attn_aae_kaggle-RSNA.ipynb

PY_RESNET    := experiments/re_attn_aae_kaggle-RSNA-ResNet.py
IPYNB_RESNET := experiments/re_attn_aae_kaggle-RSNA-ResNet.ipynb

PY_Learn    := Learn project/reattn-resnet.py
IPYNB_Learn := Learn project/reattn-resnet.ipynb

# GNU Make splits prerequisite lists on whitespace, so a bare path containing a
# space (like "Learn project/...") is silently parsed as two separate targets.
# Escaping the space lets Make (>=3.82) treat it as one token instead.
empty :=
space := $(empty) $(empty)
PY_Learn_ESC    := $(subst $(space),\ ,$(PY_Learn))
IPYNB_Learn_ESC := $(subst $(space),\ ,$(IPYNB_Learn))

.PHONY: help notebook notebook-rsna notebook-resnet notebook-Learn notebook-all clean push

help: ## Show this help
	@awk ' \
		/^## /{ desc = substr($$0, 4) } \
		/^[a-zA-Z][a-zA-Z0-9_-]*:/{ \
			if (desc) { \
				target = substr($$0, 1, index($$0, ":") - 1); \
				printf "  \033[36m%-18s\033[0m %s\n", target, desc; \
				desc = "" \
			} \
		} \
	' $(MAKEFILE_LIST)

## Convert KDD99 .py → .ipynb  (default target)
notebook: $(IPYNB)

$(IPYNB): $(PY)
	jupytext --to notebook --output $(IPYNB) $(PY)
	@echo "Generated: $(IPYNB)"

## Convert RSNA .py → .ipynb
notebook-rsna: $(IPYNB_RSNA)

$(IPYNB_RSNA): $(PY_RSNA)
	jupytext --to notebook --output $(IPYNB_RSNA) $(PY_RSNA)
	@echo "Generated: $(IPYNB_RSNA)"

## Convert ResNet experiment .py → .ipynb
notebook-resnet: $(IPYNB_RESNET)

$(IPYNB_RESNET): $(PY_RESNET)
	jupytext --to notebook --output $(IPYNB_RESNET) $(PY_RESNET)
	@echo "Generated: $(IPYNB_RESNET)"

## Convert Learn-project .py → .ipynb
notebook-Learn: $(IPYNB_Learn_ESC)

$(IPYNB_Learn_ESC): $(PY_Learn_ESC)
	jupytext --to notebook --output "$(IPYNB_Learn)" "$(PY_Learn)"
	@echo "Generated: $(IPYNB_Learn)"

## Convert all notebooks
notebook-all: notebook notebook-rsna notebook-resnet notebook-Learn

## Remove generated notebooks
clean:
	rm -f $(IPYNB) $(IPYNB_RSNA) $(IPYNB_RESNET) $(IPYNB_Learn_ESC)
	@echo "Removed generated notebooks"

## Regenerate all notebooks and push
push: notebook-all
	git add $(IPYNB) $(PY) $(IPYNB_RSNA) $(PY_RSNA) $(IPYNB_RESNET) $(PY_RESNET)
	git commit -m "Update Kaggle notebooks (auto-generated from .py sources)"
	git push

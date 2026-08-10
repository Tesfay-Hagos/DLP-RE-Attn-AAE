PY        := experiments/re_attn_aae_kaggle.py
IPYNB     := experiments/re_attn_aae_kaggle.ipynb

PY_RSNA      := experiments/re_attn_aae_kaggle-RSNA.py
IPYNB_RSNA   := experiments/re_attn_aae_kaggle-RSNA.ipynb

PY_RESNET    := experiments/re_attn_aae_kaggle-RSNA-ResNet.py
IPYNB_RESNET := experiments/re_attn_aae_kaggle-RSNA-ResNet.ipynb

PY_Learn    := experiments/Learn project/reattn-resnet.py
IPYNB_Learn := experiments/Learn project/reattn-resnet.ipynb

PY_DENOISE    := experiments/Learn project/donoising_syntehtic/denoising_syntetic_exp.py
IPYNB_DENOISE := experiments/Learn project/donoising_syntehtic/denoising_syntetic_exp.ipynb

PY_DL_PROJECT      := experiments/Learn project/DL/dl_project.py
IPYNB_DL_PROJECT   := experiments/Learn project/DL/dl_project.ipynb

PY_CV_PROJECT      := experiments/Learn project/CV/cv_project.py
IPYNB_CV_PROJECT   := experiments/Learn project/CV/cv_project.ipynb

# GNU Make splits prerequisite lists on whitespace, so a bare path containing a
# space (like "Learn project/...") is silently parsed as two separate targets.
# Escaping the space lets Make (>=3.82) treat it as one token instead.
empty :=
space := $(empty) $(empty)
PY_Learn_ESC    := $(subst $(space),\ ,$(PY_Learn))
IPYNB_Learn_ESC := $(subst $(space),\ ,$(IPYNB_Learn))
PY_DENOISE_ESC    := $(subst $(space),\ ,$(PY_DENOISE))
IPYNB_DENOISE_ESC := $(subst $(space),\ ,$(IPYNB_DENOISE))
PY_DL_PROJECT_ESC         := $(subst $(space),\ ,$(PY_DL_PROJECT))
IPYNB_DL_PROJECT_ESC      := $(subst $(space),\ ,$(IPYNB_DL_PROJECT))
PY_CV_PROJECT_ESC         := $(subst $(space),\ ,$(PY_CV_PROJECT))
IPYNB_CV_PROJECT_ESC      := $(subst $(space),\ ,$(IPYNB_CV_PROJECT))

.PHONY: help notebook notebook-rsna notebook-resnet notebook-Learn notebook-denoise notebook-dl_project notebook-cv_project notebook-all clean push

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

## Convert denoising/synthetic .py → .ipynb
notebook-denoise: $(IPYNB_DENOISE_ESC)

$(IPYNB_DENOISE_ESC): $(PY_DENOISE_ESC)
	jupytext --to notebook --output "$(IPYNB_DENOISE)" "$(PY_DENOISE)"
	@echo "Generated: $(IPYNB_DENOISE)"

## Convert the Deep Learning project (image-level AnoCls) .py → .ipynb
notebook-dl_project: $(IPYNB_DL_PROJECT_ESC)

$(IPYNB_DL_PROJECT_ESC): $(PY_DL_PROJECT_ESC)
	jupytext --to notebook --output "$(IPYNB_DL_PROJECT)" "$(PY_DL_PROJECT)"
	@echo "Generated: $(IPYNB_DL_PROJECT)"

## Convert the Computer Vision project (pixel-level AnoSeg) .py → .ipynb
notebook-cv_project: $(IPYNB_CV_PROJECT_ESC)

$(IPYNB_CV_PROJECT_ESC): $(PY_CV_PROJECT_ESC)
	jupytext --to notebook --output "$(IPYNB_CV_PROJECT)" "$(PY_CV_PROJECT)"
	@echo "Generated: $(IPYNB_CV_PROJECT)"

## Convert all notebooks
notebook-all: notebook notebook-rsna notebook-resnet notebook-Learn notebook-denoise notebook-dl_project notebook-cv_project

## Remove generated notebooks
clean:
	rm -f $(IPYNB) $(IPYNB_RSNA) $(IPYNB_RESNET) $(IPYNB_Learn_ESC) $(IPYNB_DENOISE_ESC) $(IPYNB_DL_PROJECT_ESC) $(IPYNB_CV_PROJECT_ESC)
	@echo "Removed generated notebooks"

## Regenerate all notebooks and push
push: notebook-all
	git add $(IPYNB) $(PY) $(IPYNB_RSNA) $(PY_RSNA) $(IPYNB_RESNET) $(PY_RESNET)
	git commit -m "Update Kaggle notebooks (auto-generated from .py sources)"
	git push

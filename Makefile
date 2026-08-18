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

# ML4H 2026 paper (LaTeX, jmlr/PMLR class)
PAPER_DIR := experiments/Learn project/report/DL/paper

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

.PHONY: help notebook notebook-rsna notebook-resnet notebook-Learn notebook-denoise notebook-dl_project notebook-cv_project notebook-all clean push paper paper-open paper-check paper-clean

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

## Build the ML4H paper PDF (pdflatex -> bibtex -> pdflatex x2)
paper:
	@command -v pdflatex >/dev/null 2>&1 || { \
	  echo "pdflatex not found. Install TeX Live, or use the Overleaf template."; exit 1; }
	@missing=""; for p in algorithm2e siunitx; do \
	  kpsewhich $$p.sty >/dev/null 2>&1 || missing="$$missing $$p"; done; \
	if [ -n "$$missing" ]; then \
	  echo "MISSING LaTeX packages:$$missing"; \
	  echo "  fix: sudo apt install texlive-science"; \
	  echo "  (or build on Overleaf, which has them)"; exit 1; fi
	@cd "$(PAPER_DIR)" && \
	  pdflatex -interaction=nonstopmode -file-line-error main.tex >build.log 2>&1; \
	  bibtex main >>build.log 2>&1; \
	  pdflatex -interaction=nonstopmode -file-line-error main.tex >>build.log 2>&1; \
	  pdflatex -interaction=nonstopmode -file-line-error main.tex >>build.log 2>&1; \
	  if [ ! -f main.pdf ]; then \
	    echo "BUILD FAILED — first errors:"; grep -E "^[^ ]+:[0-9]+:|^!" build.log | head -15; exit 1; fi; \
	  echo "Built: $(PAPER_DIR)/main.pdf"; \
	  echo -n "  content pages (refs excluded): "; \
	  pages=$$(pdfinfo main.pdf | awk '/^Pages/{print $$2}'); \
	  refpage=$$(for i in $$(seq 1 $$pages); do \
	      if pdftotext -f $$i -l $$i main.pdf - 2>/dev/null | grep -q "^References"; then echo $$i; break; fi; \
	    done); \
	  if [ -n "$$refpage" ]; then echo "$$((refpage-1)) of 8 used   (total $$pages incl. refs)"; \
	  else echo "$$pages total"; fi; \
	  echo -n "  overfull boxes: "; grep -c Overfull build.log || true; \
	  echo -n "  undefined refs: "; grep -c "undefined" build.log || true

## Build the paper and open the PDF
paper-open: paper
	@xdg-open "$(PAPER_DIR)/main.pdf" >/dev/null 2>&1 &

## Report paper warnings without rebuilding (needs a prior `make paper`)
paper-check:
	@cd "$(PAPER_DIR)" && test -f build.log || { echo "no build.log — run 'make paper' first"; exit 1; }
	@cd "$(PAPER_DIR)" && \
	  echo "== undefined references and citations =="; \
	  grep -E "undefined (reference|citation)|Citation .* undefined" build.log | sort -u | head -20; \
	  echo "== overfull/underfull boxes =="; \
	  grep -E "Overfull|Underfull" build.log | head -10; \
	  echo "== bibtex =="; grep -iE "warning|error" build.log | grep -i bib | head

## Remove LaTeX build artefacts (keeps main.pdf)
paper-clean:
	@cd "$(PAPER_DIR)" && rm -f main.aux main.bbl main.blg main.log main.out build.log main.lof main.lot
	@echo "Removed LaTeX build artefacts"

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

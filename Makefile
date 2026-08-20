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
# The 128px resolution test. A separate RUN_VERSION namespace (res128-v1-*), so it
# cannot touch the dl-v1 runs the paper reports.
PY_DL_RES128       := experiments/Learn project/DL/dl_res128.py
IPYNB_DL_RES128    := experiments/Learn project/DL/dl_res128.ipynb

PY_CV_PROJECT      := experiments/Learn project/CV/cv_project.py
IPYNB_CV_PROJECT   := experiments/Learn project/CV/cv_project.ipynb

# ML4H 2026 paper (LaTeX, jmlr/PMLR class)
PAPER_DIR := experiments/Learn project/report/DL/paper
PAPER_BUILD := .paper-build
PDFNAME := main.pdf

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
PY_DL_RES128_ESC          := $(subst $(space),\ ,$(PY_DL_RES128))
IPYNB_DL_RES128_ESC       := $(subst $(space),\ ,$(IPYNB_DL_RES128))
PY_CV_PROJECT_ESC         := $(subst $(space),\ ,$(PY_CV_PROJECT))
IPYNB_CV_PROJECT_ESC      := $(subst $(space),\ ,$(IPYNB_CV_PROJECT))

.PHONY: paper-read help notebook notebook-rsna notebook-resnet notebook-Learn notebook-denoise notebook-dl_project notebook-res128 notebook-cv_project notebook-all clean push paper paper-open paper-check paper-clean

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

## Convert the 128px resolution test .py → .ipynb
notebook-res128: $(IPYNB_DL_RES128_ESC)

$(IPYNB_DL_RES128_ESC): $(PY_DL_RES128_ESC)
	jupytext --to notebook --output "$(IPYNB_DL_RES128)" "$(PY_DL_RES128)"
	@echo "Generated: $(IPYNB_DL_RES128)"

## Convert the Computer Vision project (pixel-level AnoSeg) .py → .ipynb
notebook-cv_project: $(IPYNB_CV_PROJECT_ESC)

$(IPYNB_CV_PROJECT_ESC): $(PY_CV_PROJECT_ESC)
	jupytext --to notebook --output "$(IPYNB_CV_PROJECT)" "$(PY_CV_PROJECT)"
	@echo "Generated: $(IPYNB_CV_PROJECT)"

# NOLINES=1 strips \linenumbers in the BUILD COPY only. The submitted paper keeps them:
# ML4H reviews are line-referenced and the template enables them under \finalfalse.
## Build the ML4H paper PDF (local preview; Overleaf is authoritative)
paper:
	@command -v pdflatex >/dev/null 2>&1 || { \
	  echo "pdflatex not found. Install TeX Live, or build on Overleaf."; exit 1; }
	@rm -rf "$(PAPER_BUILD)" && mkdir -p "$(PAPER_BUILD)"
	@cp "$(PAPER_DIR)"/main.tex "$(PAPER_DIR)"/ref.bib "$(PAPER_DIR)"/jmlr.cls \
	    "$(PAPER_DIR)"/jmlrutils.sty "$(PAPER_BUILD)/"
	@if [ "$(NOLINES)" = "1" ]; then \
	  sed -i 's/^\\iffinal\\else\\linenumbers\\fi/%% line numbers suppressed for reading/' \
	    "$(PAPER_BUILD)/main.tex"; \
	  echo "note: line numbers suppressed in this build only; main.tex is unchanged"; \
	fi
	@cp -r "$(PAPER_DIR)"/figures "$(PAPER_BUILD)/" 2>/dev/null || true
	@# texlive-science ships algorithm2e and siunitx; jmlr.cls loads algorithm2e
	@# unconditionally. Where they are missing we stub them IN THE BUILD DIRECTORY ONLY,
	@# so the preview compiles without polluting the submission or the artifact.
	@if ! kpsewhich algorithm2e.sty >/dev/null 2>&1; then \
	  printf '%s\n' \
	    '\ProvidesPackage{algorithm2e}' \
	    '\DeclareOption*{}\ProcessOptions\relax' \
	    '\newlength{\algomargin}\setlength{\algomargin}{0pt}' \
	    '\newcounter{algocf}' > "$(PAPER_BUILD)/algorithm2e.sty"; \
	  echo "note: algorithm2e stubbed for local preview (install texlive-science for a true build)"; \
	fi
	@cd "$(PAPER_BUILD)" && \
	  pdflatex -interaction=nonstopmode -file-line-error main.tex >pass1.log 2>&1; \
	  bibtex main >bibtex.log 2>&1; \
	  pdflatex -interaction=nonstopmode main.tex >pass2.log 2>&1; \
	  pdflatex -interaction=nonstopmode -file-line-error main.tex >build.log 2>&1; \
	  if [ ! -f main.pdf ]; then \
	    echo "BUILD FAILED:"; grep -E "^[^ ]+:[0-9]+:|^!" build.log | head -15; exit 1; fi
	@cp "$(PAPER_BUILD)/main.pdf" "$(PAPER_DIR)/$(PDFNAME)"
	@echo "Built: $(PAPER_DIR)/$(PDFNAME)"
	@cd "$(PAPER_BUILD)" && python3 -c "\
import subprocess; \
n=int([l for l in subprocess.run(['pdfinfo','main.pdf'],capture_output=True,text=True).stdout.splitlines() if l.startswith('Pages')][0].split()[-1]); \
tx=lambda p: subprocess.run(['pdftotext','-f',str(p),'-l',str(p),'main.pdf','-'],capture_output=True,text=True).stdout; \
last=max([p for p in range(1,n+1) if 'Use of AI Assistance' in tx(p) or 'Conclusion' in tx(p)] or [n]); \
print('  main text ends on page %d  ->  %d of 8  %s' % (last,last,'OK' if last<=8 else 'OVER LIMIT BY %d'%(last-8))); \
print('  total     : %d pages (appendix and references excluded)' % n)"
	@cd "$(PAPER_BUILD)" && \
	  echo "  overfull  : $$(grep -c Overfull build.log)"; \
	  echo "  undefined : $$(grep -ci 'reference.*undefined' build.log)"; \
	  echo "  citations : $$(grep -c bibitem main.bbl)"
	@if [ "$(NOLINES)" != "1" ] && [ -f "$(PAPER_DIR)/main-reading.pdf" ]; then \
	  $(MAKE) --no-print-directory paper NOLINES=1 PDFNAME=main-reading.pdf >/dev/null && \
	  echo "  (reading copy refreshed: $(PAPER_DIR)/main-reading.pdf)"; \
	fi

## Build a reading copy with no line numbers (submission keeps them)
paper-read:
	@$(MAKE) --no-print-directory paper NOLINES=1 PDFNAME=main-reading.pdf

## Build the paper and open the PDF
paper-open: paper
	@xdg-open "$(PAPER_DIR)/main.pdf" >/dev/null 2>&1 &

## Report paper warnings without rebuilding (needs a prior `make paper`)
paper-check:
	@test -f "$(PAPER_BUILD)/build.log" || {  echo "no build.log — run 'make paper' first"; exit 1; }
	@cd "$(PAPER_BUILD)" && \
	  echo "== undefined references and citations =="; \
	  grep -E "undefined (reference|citation)|Citation .* undefined" build.log | sort -u | head -20; \
	  echo "== overfull/underfull boxes =="; \
	  grep -E "Overfull|Underfull" build.log | head -10; \
	  echo "== bibtex =="; grep -iE "warning|error" build.log | grep -i bib | head

## Remove LaTeX build artefacts (keeps main.pdf)
paper-clean:
	@rm -rf "$(PAPER_BUILD)"
	@cd "$(PAPER_DIR)" && rm -f main.aux main.bbl main.blg main.log main.out build.log main-reading.pdf
	@echo "Removed LaTeX build artefacts"

## Convert all notebooks
notebook-all: notebook notebook-rsna notebook-resnet notebook-Learn notebook-denoise notebook-dl_project notebook-res128 notebook-cv_project

## Remove generated notebooks
clean:
	rm -f $(IPYNB) $(IPYNB_RSNA) $(IPYNB_RESNET) $(IPYNB_Learn_ESC) $(IPYNB_DENOISE_ESC) $(IPYNB_DL_PROJECT_ESC) $(IPYNB_DL_RES128_ESC) $(IPYNB_CV_PROJECT_ESC)
	@echo "Removed generated notebooks"

## Regenerate all notebooks and push
push: notebook-all
	git add $(IPYNB) $(PY) $(IPYNB_RSNA) $(PY_RSNA) $(IPYNB_RESNET) $(PY_RESNET)
	git commit -m "Update Kaggle notebooks (auto-generated from .py sources)"
	git push

# ---------------------------------------------------------------------------
# Prose passes over the paper. Each is presentation-only; the guard proves it.
# ---------------------------------------------------------------------------
SKILLS   := .claude/skills
PAPER_TEX := $(PAPER_DIR)/main.tex

## Snapshot the paper's numbers, citations and headings before an editing pass
paper-snapshot:
	@python3 $(SKILLS)/paper-guard/scripts/guard.py snapshot "$(PAPER_TEX)"

## Prove an editing pass changed no number, citation, heading or claim
paper-guard:
	@python3 $(SKILLS)/paper-guard/scripts/guard.py check "$(PAPER_TEX)"

## Rank paragraphs by reading difficulty, worst first
paper-prose:
	@python3 $(SKILLS)/paper-humanizer/scripts/readability.py "$(PAPER_TEX)"

## Print the argument spine and flag broken joins between paragraphs
paper-flow:
	@python3 $(SKILLS)/paper-flow/scripts/flow_map.py "$(PAPER_TEX)" --audit

## Report source spacing inconsistencies (add FIX=1 to apply the safe ones)
paper-spacing:
	@python3 $(SKILLS)/paper-formatter/scripts/spacing.py "$(PAPER_TEX)" \
	  $(if $(FIX),--fix --backup,)

.PHONY: paper-snapshot paper-guard paper-prose paper-flow paper-spacing

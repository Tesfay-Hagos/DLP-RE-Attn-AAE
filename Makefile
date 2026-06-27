PY        := experiments/re_attn_aae_kaggle.py
IPYNB     := experiments/re_attn_aae_kaggle.ipynb

PY_RSNA      := experiments/re_attn_aae_kaggle-RSNA.py
IPYNB_RSNA   := experiments/re_attn_aae_kaggle-RSNA.ipynb

PY_RESNET    := experiments/re_attn_aae_kaggle-RSNA-ResNet.py
IPYNB_RESNET := experiments/re_attn_aae_kaggle-RSNA-ResNet.ipynb

.PHONY: notebook notebook-rsna notebook-resnet notebook-all clean push

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

## Convert all notebooks
notebook-all: notebook notebook-rsna notebook-resnet

## Remove generated notebooks
clean:
	rm -f $(IPYNB) $(IPYNB_RSNA) $(IPYNB_RESNET)
	@echo "Removed generated notebooks"

## Regenerate all notebooks and push
push: notebook-all
	git add $(IPYNB) $(PY) $(IPYNB_RSNA) $(PY_RSNA) $(IPYNB_RESNET) $(PY_RESNET)
	git commit -m "Update Kaggle notebooks (auto-generated from .py sources)"
	git push

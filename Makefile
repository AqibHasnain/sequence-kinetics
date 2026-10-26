PY := .venv/bin/python
PYTEST := .venv/bin/pytest
RUFF := .venv/bin/ruff

.PHONY: help setup lint test data g0 g2 phase1 inputs train grid nuts eval report clean

help:
	@grep -E '^[a-z0-9-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

setup: ## Create the venv and install the package with dev extras
	uv venv --python 3.12 && uv pip install -e ".[dev,nuts]"

lint: ## Lint and format-check
	$(RUFF) check src tests
	$(RUFF) format --check src tests

test: ## Test suite (CPU, no network)
	$(PYTEST) tests

data: ## Download every source and build the gene, K562 and variant tables (idempotent)
	$(PY) -m sk.data.sources
	$(PY) -m sk.data.genome
	$(PY) -m sk.data.k562
	$(PY) -m sk.data.variants

g0: ## Evaluate Gate G0 and write results/phase0/gate_g0.json
	$(PY) -m sk.eval.gate_g0

g2: ## Count the routing sets under cross-validation for Gate G2 (results/phase0/gate_g2.json)
	$(PY) -m sk.data.gate_g2

phase1: ## Fit per-gene posteriors, identifiability analysis, SBC and Gate G1 (CPU)
	$(PY) -m sk.eval.phase1 all --family full

ARM ?= KIN
SEED ?= 0
FOLD ?= fold0

inputs: ## Encode promoter and transcript inputs for every protein-coding gene, and scan leakage
	$(PY) -m sk.model.inputs
	$(PY) -m sk.data.leakage

train: ## Train one arm, seed and fold and score its held-out chromosomes (ARM, SEED, FOLD)
	nice -n 15 $(PY) -m sk.train --arm $(ARM) --seed $(SEED) --fold $(FOLD)

runs/phase2/queue.txt: configs/phase2_queue.txt
	mkdir -p runs/phase2/logs
	cp $< $@

grid: runs/phase2/queue.txt ## Train every run in configs/phase2_queue.txt, one job at a time
	scripts/run_queue.sh

nuts: ## NUTS reference posteriors for the amortization gap (CPU)
	nice -n 15 $(PY) -m sk.infer.nuts --arms KIN SS-EI KIN-1T

eval: ## Routing metrics, hypotheses, gene-level metrics and the motif baseline
	$(PY) -m sk.eval.motif_baseline
	$(PY) -m sk.eval.phase3
	$(PY) -m sk.eval.routing_diagnostic
	$(PY) -m sk.eval.gene_metrics
	$(PY) -m sk.eval.gene_diagnostics

report: ## Regenerate the Phase 5 figures from results and runs
	$(PY) -m sk.eval.figures

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__

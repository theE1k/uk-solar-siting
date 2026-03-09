.PHONY: setup download era5 run pipeline dashboard score help

## Install Python dependencies
setup:
	pip install -r requirements.txt

## Download all datasets (auto-detects credentials from .env)
download:
	python src/download_all.py

## Download ERA5 climate data (requires CDS_KEY in .env)
era5:
	python -m src.download_era5

## Run the full pipeline (phases 1–4)
run:
	python run_pipeline.py

## Run a specific phase: make pipeline PHASE="1"  or  make pipeline PHASE="2 3"
pipeline:
	python run_pipeline.py --phase $(PHASE)

## Score NW candidate sites (financial + composite ranking)
score:
	python -m src.score_financial

## Launch the interactive dashboard
dashboard:
	streamlit run dashboard/app.py

## Show this help
help:
	@echo ""
	@echo "UK Solar Siting — available commands:"
	@echo ""
	@echo "  make setup      Install Python dependencies"
	@echo "  make download   Download all datasets"
	@echo "  make era5       Download ERA5 (needs CDS_KEY in .env)"
	@echo "  make run        Run full pipeline (phases 1–4)"
	@echo "  make pipeline PHASE='2 3'  Run specific phases"
	@echo "  make score      Score NW candidate sites"
	@echo "  make dashboard  Launch Streamlit dashboard"
	@echo ""
	@echo "First-time setup:"
	@echo "  1. make setup"
	@echo "  2. cp .env.example .env  # add CDS_KEY for ERA5"
	@echo "  3. make download"
	@echo "  4. make run"
	@echo "  5. make dashboard"
	@echo ""

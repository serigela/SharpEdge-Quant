# SharpEdge Quant

A Python signal pipeline and C++ backtester for exploring simple daily trading strategies. The repository is a research demo, not evidence of profitable trading.

## Evaluation design

Technical indicators are computed from the current and earlier bars. The latest row has no known future return and is excluded from labels. The chronological split reserves the final 20% for evaluation, purges the training row whose label crosses the split, and fits the scaler on training rows only. Signals are generated only from held-out feature rows and executed at the **next bar's open**, with configurable slippage. `SignalGenerator.holdout_metrics` records classification accuracy, sample count, and test start; backtest metrics cover only the held-out execution period.

This is a single split on historical data. It is not a walk-forward evaluation, and neither the synthetic unit-test prices nor a favorable backtest establish future performance. The C++ latency parameter assumes a fixed 0.1-second interval between CSV rows, so leave latency at zero for daily bars.

## Setup

Use Python 3.10–3.12, a C++17 compiler, and CMake. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cmake -S . -B build -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build
python src/python/main.py --ticker AAPL --period 2y --model random_forest
```

The full workflow downloads market prices via yfinance. You can instead run `python src/python/signal_generation.py --help` against an existing OHLCV CSV. `src/python/main.py` needs the compiled `quant_cpp_engine` module in `build/` for the backtest.

## Checks

```bash
PYTHONPATH=src/python python -m unittest discover -s tests -p 'test_*.py' -v
g++ -std=c++17 -O2 -Isrc/cpp tests/backtester_repeat.cpp src/cpp/backtester.cpp -o /tmp/backtester_repeat
/tmp/backtester_repeat
```

The Python checks cover unknown future labels, train-only scaling, and next-bar held-out execution. The C++ check confirms a repeated backtest on the same object returns the same results. CI runs both on pushes and pull requests.

## Structure

`src/python/data_ingestion.py` fetches and stores prices; `src/python/signal_generation.py` trains and exports signals; `src/cpp/backtester.cpp` applies trades and calculates returns; `src/cpp/binding.cpp` exposes the engine to Python.

MIT licensed.

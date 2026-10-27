#include "backtester.h"
#include <cassert>
#include <cmath>
#include <fstream>

int main() {
    const char* path = "repeat_signals.csv";
    {
        std::ofstream file(path);
        file << "timestamp,price,signal\n"
             << "2024-01-01,100,1\n2024-01-02,102,1\n"
             << "2024-01-03,101,0\n2024-01-04,105,1\n";
    }
    Backtester engine(10000, 0.0005, 0);
    assert(engine.loadSignalsFromCSV(path));
    engine.runBacktest();
    const auto first = engine.getResults();
    engine.runBacktest();
    const auto second = engine.getResults();
    assert(std::abs(first.finalEquity - second.finalEquity) < 1e-9);
    assert(std::abs(first.sharpeRatio - second.sharpeRatio) < 1e-9);
    assert(first.totalTrades == second.totalTrades);
    std::remove(path);
}

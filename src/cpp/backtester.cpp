#include "backtester.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

Backtester::Backtester() 
    : m_initialCapital(10000.0), 
      m_cash(10000.0), 
      m_position(0),
      m_slippage(0.0005),
      m_latency(0.0) {}

Backtester::Backtester(double initialCapital, double slippage, double latency) 
    : m_initialCapital(initialCapital), 
      m_cash(initialCapital), 
      m_position(0),
      m_slippage(slippage),
      m_latency(latency) {}

bool Backtester::loadSignalsFromCSV(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filePath << std::endl;
        return false;
    }

    // Clear previous data
    m_signals.clear();
    m_equity.clear();
    m_drawdowns.clear();
    m_returns.clear();
    
    // Reset cash and position
    m_cash = m_initialCapital;
    m_position = 0;

    // Read the header
    std::string line;
    std::getline(file, line);
    
    // Parse CSV data
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string timestamp, priceStr, signalStr;
        
        // Parse CSV row
        std::getline(ss, timestamp, ',');
        std::getline(ss, priceStr, ',');
        std::getline(ss, signalStr, ',');
        
        try {
            double price = std::stod(priceStr);
            int signal = std::stoi(signalStr);
            
            // Add to signals
            m_signals.push_back({timestamp, price, signal});
        } catch (const std::exception& e) {
            std::cerr << "Error parsing line: " << line << " - " << e.what() << std::endl;
        }
    }
    
    file.close();
    return !m_signals.empty();
}

void Backtester::runBacktest() {
    if (m_signals.empty()) {
        std::cerr << "Error: No signals loaded" << std::endl;
        return;
    }
    
    // Initialize tracking variables
    m_cash = m_initialCapital;
    m_position = 0;
    m_equity.clear();
    m_trades.clear();
    m_drawdowns.clear();
    m_returns.clear();
    
    double lastEquity = m_initialCapital;
    double highWaterMark = m_initialCapital;
    int currentSignal = 0;
    
    // Process each signal
    for (size_t i = 0; i < m_signals.size(); ++i) {
        const auto& signal = m_signals[i];
        
        // Check if signal has changed
        if (signal.signal != currentSignal) {
            // Apply latency if specified
            double effectivePrice = signal.price;
            if (m_latency > 0.0) {
                // Find the price after latency
                size_t latencySteps = static_cast<size_t>(m_latency * 10);  // Assume 0.1 second per step
                size_t nextIdx = std::min(i + latencySteps, m_signals.size() - 1);
                effectivePrice = m_signals[nextIdx].price;
            }
            
            // Apply slippage
            if (signal.signal == 1) {  // Buy
                effectivePrice *= (1.0 + m_slippage);
            } else {  // Sell
                effectivePrice *= (1.0 - m_slippage);
            }
            
            // Execute trade
            if (signal.signal == 1 && m_position == 0) {  // Buy
                // Calculate how many shares we can buy
                int shares = static_cast<int>(m_cash / effectivePrice);
                if (shares > 0) {
                    m_position = shares;
                    m_cash -= shares * effectivePrice;
                    
                    // Record trade
                    m_trades.push_back({
                        signal.timestamp,
                        "BUY",
                        shares,
                        effectivePrice,
                        shares * effectivePrice
                    });
                }
            } else if (signal.signal == 0 && m_position > 0) {  // Sell
                double proceeds = m_position * effectivePrice;
                
                // Record trade
                m_trades.push_back({
                    signal.timestamp,
                    "SELL",
                    m_position,
                    effectivePrice,
                    proceeds
                });
                
                m_cash += proceeds;
                m_position = 0;
            }
            
            currentSignal = signal.signal;
        }
        
        // Calculate equity at this point
        double equity = m_cash;
        if (m_position > 0) {
            equity += m_position * signal.price;
        }
        
        // Record equity
        m_equity.push_back({signal.timestamp, equity});
        
        // Calculate drawdown
        highWaterMark = std::max(highWaterMark, equity);
        double drawdown = (highWaterMark - equity) / highWaterMark * 100.0;
        m_drawdowns.push_back(drawdown);
        
        // Calculate returns
        double dailyReturn = equity / lastEquity - 1.0;
        m_returns.push_back(dailyReturn);
        lastEquity = equity;
    }
}

BacktestResults Backtester::getResults() const {
    BacktestResults results;
    
    if (m_equity.empty()) {
        return results;
    }
    
    // Calculate final return
    double finalEquity = m_equity.back().equity;
    results.finalEquity = finalEquity;
    results.finalReturn = (finalEquity / m_initialCapital - 1.0) * 100.0;
    
    // Calculate max drawdown
    results.maxDrawdown = *std::max_element(m_drawdowns.begin(), m_drawdowns.end());
    
    // Calculate Sharpe ratio
    double totalReturn = 0.0;
    double totalRisk = 0.0;
    
    // Skip first entry which has no return
    for (size_t i = 0; i < m_returns.size(); ++i) {
        totalReturn += m_returns[i];
        totalRisk += m_returns[i] * m_returns[i];
    }
    
    double meanReturn = totalReturn / m_returns.size();
    double stdDev = std::sqrt(totalRisk / m_returns.size() - meanReturn * meanReturn);
    
    // Annualized Sharpe ratio (assuming daily returns)
    if (stdDev > 0) {
        results.sharpeRatio = (meanReturn * 252) / (stdDev * std::sqrt(252));
    } else {
        results.sharpeRatio = 0;
    }
    
    // Trading statistics
    results.totalTrades = m_trades.size();
    
    return results;
}

void Backtester::printResults() const {
    BacktestResults results = getResults();
    
    std::cout << "===== BACKTEST RESULTS =====" << std::endl;
    std::cout << "Initial Capital: $" << m_initialCapital << std::endl;
    std::cout << "Final Equity: $" << results.finalEquity << std::endl;
    std::cout << "Final Return: " << results.finalReturn << "%" << std::endl;
    std::cout << "Max Drawdown: " << results.maxDrawdown << "%" << std::endl;
    std::cout << "Sharpe Ratio: " << results.sharpeRatio << std::endl;
    std::cout << "Total Trades: " << results.totalTrades << std::endl;
    
    // Print some trade details
    std::cout << std::endl << "===== SAMPLE TRADES =====" << std::endl;
    size_t numTradesToShow = std::min(m_trades.size(), static_cast<size_t>(5));
    for (size_t i = 0; i < numTradesToShow; ++i) {
        const auto& trade = m_trades[i];
        std::cout << trade.timestamp << ": " << trade.action 
                  << " " << trade.shares << " shares @ $" << trade.price 
                  << " = $" << trade.value << std::endl;
    }
}

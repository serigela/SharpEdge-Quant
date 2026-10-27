import tempfile
import unittest

import numpy as np
import pandas as pd

from signal_generation import FeatureEngineering, SignalGenerator


class SignalIntegrityTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        close = 100 + np.cumsum(rng.normal(0.15, 1.5, 200))
        self.prices = pd.DataFrame({
            'Open': close + 0.1, 'High': close + 1, 'Low': close - 1,
            'Close': close, 'Volume': np.full(200, 1000),
        }, index=pd.date_range('2023-01-01', periods=200, freq='D'))

    def test_unknown_future_is_not_a_negative_label(self):
        features = FeatureEngineering.add_technical_indicators(self.prices)
        labeled = FeatureEngineering.create_target(features)
        self.assertEqual(len(labeled), len(features) - 1)
        self.assertLess(labeled.index.max(), features.index.max())

    def test_scaler_and_execution_use_only_valid_periods(self):
        with tempfile.TemporaryDirectory() as directory:
            generator = SignalGenerator('logistic_regression', directory)
            x_train, x_test, _, _, columns = generator.prepare_data(self.prices)
            features = FeatureEngineering.create_target(FeatureEngineering.add_technical_indicators(self.prices))
            split = int(len(features) * 0.8)
            np.testing.assert_allclose(generator.scaler.mean_, features.iloc[:split-1][columns].mean())
            self.assertEqual(len(x_train), split - 1)
            self.assertEqual(len(x_test), len(features) - split)
            generator.train(self.prices)
            signals = generator.generate_signals(self.prices)
            self.assertGreater(len(signals), 0)
            self.assertGreater(signals['timestamp'].min(), generator.test_start)
            first_execution = self.prices.loc[signals['timestamp'].iloc[0]]
            self.assertEqual(signals['price'].iloc[0], first_execution['Open'])
            self.assertEqual(len(signals), len(x_test))


if __name__ == '__main__':
    unittest.main()

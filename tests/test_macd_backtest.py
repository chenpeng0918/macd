import unittest

import numpy as np
import pandas as pd

from macd_backtest import backtest, signals_macd


class TestMacdBacktest(unittest.TestCase):
    def test_macd_warmup_stays_flat(self):
        df = pd.DataFrame(
            {
                "MACD": [np.nan, np.nan, 0.1],
                "MACD_Signal": [np.nan, 0.0, 0.2],
            }
        )
        positions = signals_macd(df, long_short=True).tolist()
        self.assertEqual(positions, [0, 0, -1])

    def test_backtest_aligns_execution_and_cost(self):
        idx = pd.date_range("2024-01-01", periods=3)
        df = pd.DataFrame({"Close": [100, 100, 110], "position": [0, 1, 1]}, index=idx)

        result, _ = backtest(df, cost_bps=10)

        self.assertEqual(result["position"].tolist(), [0.0, 0.0, 1.0])
        self.assertEqual(result["turnover"].tolist(), [0.0, 0.0, 1.0])
        self.assertAlmostEqual(result["cost"].iloc[2], 0.001, places=8)
        self.assertAlmostEqual(result["strat_ret"].iloc[2], 0.099, places=8)

    def test_trade_stats_are_counted_by_trade_segment(self):
        idx = pd.date_range("2024-01-01", periods=4)
        df = pd.DataFrame({"Close": [100, 100, 110, 110], "position": [0, 1, 0, 0]}, index=idx)

        _, metrics = backtest(df, cost_bps=0)

        self.assertEqual(metrics.n_trades, 1)
        self.assertAlmostEqual(metrics.win_rate, 1.0, places=8)


if __name__ == "__main__":
    unittest.main()

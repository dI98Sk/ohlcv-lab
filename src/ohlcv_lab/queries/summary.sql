-- Per-symbol statistics; the SQL twin of (part of) stats.summary.
--
-- skewness() and kurtosis() use the same bias-corrected sample formulas as
-- pandas, and kurtosis() is already the excess kurtosis (normal = 0).
WITH moments AS (
    SELECT
        symbol,
        count(log_return) AS n,
        stddev_samp(log_return) * sqrt($ppy) AS ann_vol,
        avg(log_return) / stddev_samp(log_return) * sqrt($ppy) AS sharpe,
        skewness(log_return) AS skew,
        kurtosis(log_return) AS excess_kurtosis,
        quantile_cont(log_return, 0.01) AS "var_1%",
        quantile_cont(log_return, 0.05) AS "var_5%"
    FROM returns
    WHERE interval = $interval
    GROUP BY symbol
),
worst AS (
    SELECT
        symbol,
        min(drawdown) AS max_drawdown,
        arg_min(open_time, drawdown) AS trough
    FROM drawdowns
    WHERE interval = $interval
    GROUP BY symbol
)
SELECT *
FROM moments
JOIN worst USING (symbol)
ORDER BY symbol

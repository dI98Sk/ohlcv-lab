-- Annualised volatility over a trailing time window; the SQL twin of stats.rolling_vol.
--
-- RANGE ... PRECEDING includes both ends, [t - window, t], while the pandas
-- window "30D" is (t - window, t]: one candle more. Taking one millisecond off
-- the width makes both windows hold the same candles.
SELECT
    symbol,
    open_time,
    CASE
        WHEN count(log_return) OVER w >= $min_periods
        THEN stddev_samp(log_return) OVER w * sqrt($ppy)
    END AS ann_vol
FROM returns
WHERE interval = $interval
WINDOW w AS (
    PARTITION BY symbol
    ORDER BY open_time
    RANGE BETWEEN $window::INTERVAL - INTERVAL 1 MILLISECOND PRECEDING AND CURRENT ROW
)
ORDER BY symbol, open_time

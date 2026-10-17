-- Correlation of log returns for every pair of symbols, one row per pair.
--
-- Returns are matched by open_time; corr() skips a pair of values when either
-- is NULL, which is what pandas' DataFrame.corr() does too.
SELECT
    a.symbol AS symbol_a,
    b.symbol AS symbol_b,
    corr(a.log_return, b.log_return) AS corr,
    count(a.log_return + b.log_return) AS n
FROM returns AS a
JOIN returns AS b
    ON a.open_time = b.open_time
    AND a.interval = b.interval
    AND a.symbol < b.symbol
WHERE a.interval = $interval
GROUP BY a.symbol, b.symbol
ORDER BY a.symbol, b.symbol

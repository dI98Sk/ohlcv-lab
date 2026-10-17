-- Runs of consecutive missing candles in every series; the SQL twin of clean.gap_ranges.
--
-- "Gaps and islands": inside one run of gaps, the row number among all candles
-- and the row number among gap candles grow in step, so their difference is
-- constant along the run and changes between runs.
WITH numbered AS (
    SELECT
        symbol,
        interval,
        open_time,
        is_gap,
        row_number() OVER (PARTITION BY symbol, interval ORDER BY open_time)
            - row_number() OVER (PARTITION BY symbol, interval, is_gap ORDER BY open_time) AS run
    FROM candles
)
SELECT
    symbol,
    interval,
    min(open_time) AS start,
    max(open_time) AS "end",
    count(*) AS candles
FROM numbered
WHERE is_gap
GROUP BY symbol, interval, run
ORDER BY symbol, interval, start

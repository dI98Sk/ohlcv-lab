-- Aggregate candles into wider ones (e.g. 1h -> 1 day), buckets aligned to UTC.
--
-- Bucketing runs on a plain UTC TIMESTAMP: time_bucket on TIMESTAMPTZ does
-- time-zone calendar math per row and was ~30x slower on 542k minute candles.
-- timezone('UTC', ...) converts explicitly, so the result does not depend on
-- the session TimeZone either.
--
-- open/close come from the first/last *existing* candle of the bucket. A bucket
-- that misses candles (a gap, or the unfinished edge of the series) is kept but
-- marked complete = false: its range and volume cover only part of the period.
SELECT
    symbol,
    timezone('UTC', time_bucket($bucket::INTERVAL, timezone('UTC', open_time))) AS bucket,
    arg_min(open, open_time) AS open,
    max(high) AS high,
    min(low) AS low,
    arg_max(close, open_time) AS close,
    sum(volume) AS volume,
    sum(turnover) AS turnover,
    count(*) AS candles,
    count(*) = $candles_per_bucket AS complete
FROM candles
WHERE interval = $interval AND NOT is_gap
GROUP BY symbol, bucket
ORDER BY symbol, bucket

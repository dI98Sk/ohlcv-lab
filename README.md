# ohlcv-lab

Небольшой, но настоящий пайплайн рыночных данных:
**Bybit API → сырые JSON → чистый Parquet → SQL → статистика и отчёт.**

- Основной датасет: BTC, ETH, SOL, XRP, DOGE (USDT perpetual), часовые свечи за 3 года.
- Дополнительный: BTCUSDT, минутные свечи за 6–12 месяцев — для проверки ресемплинга и замеров скорости.

> 🚧 В разработке.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
nbstripout --install   # очищать выводы ноутбуков при коммите (настраивается на каждый клон)
```

## Загрузка данных

```bash
# основной датасет: 5 монет, часовые свечи с 2023-10-01
ohlcv-fetch --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT DOGEUSDT --interval 60 --start 2023-10-01
# дополнительный: BTCUSDT, минутные свечи за год
ohlcv-fetch --symbols BTCUSDT --interval 1 --start 2025-10-01
```

Ответы API сохраняются как есть в `data/raw/bybit/<category>/<symbol>/<interval>/`:
один JSON на окно из 1000 свечей, окна выровнены по эпохе. Повторный запуск
докачивает только недостающее. Окно, в котором последняя свеча ещё не закрыта,
пишется как `*.partial.json` и перекачивается при следующем запуске.

## Структура

```
src/ohlcv_lab/   код пайплайна (импортируется как пакет ohlcv_lab)
tests/           pytest
notebooks/       ноутбуки — только вызывают функции из src/
data/            raw/ и clean/ — не коммитятся, воспроизводятся из API
```

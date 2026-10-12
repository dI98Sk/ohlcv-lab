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

## Структура

```
src/ohlcv_lab/   код пайплайна (импортируется как пакет ohlcv_lab)
tests/           pytest
notebooks/       ноутбуки — только вызывают функции из src/
data/            raw/ и clean/ — не коммитятся, воспроизводятся из API
```

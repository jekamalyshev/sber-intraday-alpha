# Data Directory

This directory is **excluded from version control** (see `.gitignore`).

## How to Get the Data

1. Go to [Finam Export](https://www.finam.ru/profile/moex-akcii/sberbank/export/)
2. Select:
   - **Instrument**: Сбербанк (SBER)
   - **Period**: 5 minutes
   - **Date range**: as needed
   - **Format**: `.csv`, separator `;`, header row included
3. Save the file as:

```
data/Сбербанк/year_result.csv
```

## Expected Format

```
<TICKER>;<PER>;<DATE>;<TIME>;<OPEN>;<HIGH>;<LOW>;<CLOSE>;<VOL>
SBER;5;20200102;100000;262.50;263.10;262.30;262.90;1540200
SBER;5;20200102;100500;262.90;263.40;262.70;263.20;987300
...
```

## Column Description

| Column | Type | Description |
|---|---|---|
| `<TICKER>` | string | Stock ticker symbol |
| `<PER>` | int | Timeframe in minutes (5) |
| `<DATE>` | int | Date in `YYYYMMDD` format |
| `<TIME>` | int | Time in `HHMMSS` format |
| `<OPEN>` | float | Open price |
| `<HIGH>` | float | High price |
| `<LOW>` | float | Low price |
| `<CLOSE>` | float | Close price |
| `<VOL>` | int | Trade volume |

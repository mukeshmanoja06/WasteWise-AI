# WasteWise AI — Dataset Documentation

## Source

**Official publication:**
Ministry of Statistics and Programme Implementation (MoSPI), Government of India
*EnviStats India 2024*, Volume II: Environment Accounts

**Statement / Table:** Statement 3.06
*Municipal Solid Waste Generation in Metro Cities and State Capitals*

The data were manually extracted from Statement 3.06 of EnviStats India 2024 and
compiled into a tabular CSV file for this project.

---

## Original Extracted Dataset

| Property | Value |
|---|---|
| File | `data/raw/india_msw_city_year.csv` |
| Rows (data) | 414 |
| Rows (header) | 1 |
| Columns | 6 |
| Cities | 46 |
| Fiscal-year snapshots | 9 |

### Columns

| Column | Type | Description |
|---|---|---|
| `fiscal_year` | string | Fiscal year label, e.g. `2022-23` |
| `city` | string | City name |
| `estimated_population_in_2011` | integer | 2011 Census population (anchor value, static per city) |
| `waste_generated` | float | Municipal solid waste generation in **tonnes per day** |
| `unit` | string | Constant: `"estimated_population in absolute number, waste_generated in tonnes per day"` |
| `note` | float | Entirely null — no content |

### Fiscal-Year Snapshots Present

The dataset contains observations for the following nine fiscal years only.
No additional years exist in the source statement.

```
1999-00
2004-05
2010-11
2015-16
2018-19
2019-20
2020-21
2021-22
2022-23
```

The time spacing is **non-uniform**: gaps of 5 years, 6 years, 5 years, then
consecutive annual snapshots from 2018-19 onwards.

---

## Missing-Data Pattern

107 of the 414 city-year rows (25.8 %) have a null `waste_generated` value.

The missing values are **not random**. A systematic block of approximately
25 cities has no recorded waste generation for fiscal years 2018-19 and
2019-20. This is a known data-collection gap in the source table and is
not imputed or filled in this project.

Additionally, several smaller cities have sparse early-period coverage
(no records in 1999-00 or 2004-05).

**Aurangabad** (population 1,171,330 in 2011) has no recorded waste generation
across **all nine** fiscal years. It was removed during preprocessing because a
city with no observed target values cannot be used for model training or
evaluation.

---

## Preprocessing Summary

| Step | Detail |
|---|---|
| Drop columns | `unit` (constant, zero variance) and `note` (100 % null) |
| Remove Aurangabad | 9 rows removed — no observed waste values |
| Result | 405 rows × 4 core columns |

### Cleaned Dataset

| Property | Value |
|---|---|
| File | `data/processed/india_msw_cleaned.csv` |
| Rows | 405 |
| Columns | 13 (original 4 + 9 engineered features) |
| Cities | 45 |
| Observed waste rows | **307** |
| Null-target rows (retained) | 98 |

The 98 null-target rows are retained in the cleaned dataset with null
`waste_generated` and `log_waste_generated`. They are excluded from model
training and evaluation but can be used for inference.

### Engineered Features in the Cleaned Dataset

| Feature | Description |
|---|---|
| `year_start` | Starting calendar year of the fiscal year (integer) |
| `year_rank` | Ordinal rank 0–8 over the 9 fiscal years |
| `years_since_base` | Years elapsed since 1999-00 (0–23) |
| `log_population` | Natural log of `estimated_population_in_2011` |
| `pop_tier` | City size tier: 1 = Mega (>4 M), 2 = Large (2–4 M), 3 = Medium (<2 M) |
| `log_waste_generated` | `log1p(waste_generated)` — primary model target |
| `waste_per_capita` | Approximate kg / person / day (2011 population anchor) |
| `city_mean_waste_all` | Per-city mean waste across all available years (analytics only) |
| `dq_flag_low_early` | 1 for Delhi and Bengaluru 1999-00 (data-quality suspects) |

---

## Data-Quality Notes

1. **1999-00 Delhi and Bengaluru values are flagged as suspect.**
   Delhi reported 400 t/day in 1999-00 and 5,922 t/day in 2004-05 (14.8× jump).
   Bengaluru reported 200 t/day in 1999-00 and 1,669 t/day in 2004-05 (8.3× jump).
   These rows are retained with `dq_flag_low_early = 1` and can be excluded in
   sensitivity analyses.

2. **Static 2011 population.** The `estimated_population_in_2011` column is a
   fixed 2011 Census anchor. It does not reflect actual city populations in
   other fiscal years (1999 through 2023). Per-capita figures derived from this
   column are approximations.

3. **Non-uniform time spacing.** The dataset is a series of point-in-time
   snapshots, not a continuous annual time series. The model treats
   `year_start` as a numeric covariate rather than a proper lag structure.

---

## City List (45 modelling cities)

Agra, Ahmedabad, Allahabad, Amritsar, Bengaluru, Bhopal, Chennai,
Coimbatore, Delhi, Dhanbad, Faridabad, Gaziabad, Gwalior, Howrah,
Hyderabad, Indore, Jabalpur, Jaipur, Jodhpur, Kalyan-dombivali, Kanpur,
Kolkata, Kota, Lucknow, Ludhiana, Madurai, Meerut, Mumbai, Nagpur,
Nashik, Navi Mumbai, Patna, Pimpri-chinchwad, Pune, Raipur, Rajkot,
Ranchi, Srinagar, Surat, Thane, Vadodara, Varanasi, Vasai-virar,
Vijayawada, Vishakhapatnam

---

## Limitations

- The dataset covers only metropolitan cities and state capitals listed in
  EnviStats India 2024 Statement 3.06. Smaller municipalities are not included.
- Waste-generation figures are government-reported estimates and may themselves
  carry measurement or reporting uncertainty.
- No information on waste composition, treatment method, collection efficiency,
  or disposal infrastructure is available in this dataset.
- No GDP, income, urbanisation-rate, or demographic-growth data have been joined.
- The project does not include a DOI, open-data licence, or programmatic
  download mechanism for the raw data. The source publication must be consulted
  directly for the official figures.

# Parking Receipt Management & Revenue Analysis

A Python data-analysis project that generates synthetic parking records, cleans the data, creates parking receipts, and calculates revenue, occupancy, payment, and vehicle-type insights.

The project uses a reproducible dataset covering January to June 2026 and is designed for a parking lot with 15 slots: 6 Bike, 6 Car, and 3 SUV slots.

## Features

- Generates 950 unique parking transactions plus 10 duplicate receipt records
- Creates realistic Indian-style vehicle registration numbers
- Introduces missing values and duplicate receipt IDs for data-cleaning practice
- Removes duplicates and reconstructs missing fields
- Generates formatted parking receipts with full-hour billing
- Implements `ParkingReceipt` and `ParkingManagement` classes
- Calculates revenue, occupancy, peak hours, monthly trends, and vehicle-type performance
- Detects unusually high parking fees using the IQR method
- Produces five recommendations for parking operations

## Project Files

| File | Purpose |
| --- | --- |
| `generate_parking_data.py` | Generates the raw synthetic dataset in `parking_records.csv` |
| `parking_records.csv` | Raw data, including duplicates and missing values |
| `parking_analysis.py` | Cleans and analyzes the data |
| `parking_cleaned.csv` | Cleaned dataset with added date and time columns |
| `parking_receipts.txt` | Generated formatted receipts for all 950 transactions |
| `analysis_output.txt` | Complete text report produced by `parking_analysis.py` |

## Requirements

- Python 3
- NumPy
- Pandas

Install the dependencies:

```bash
pip install numpy pandas
```

## Usage

Run the scripts from the project directory in this order:

```bash
python generate_parking_data.py
python parking_analysis.py
```

The generator uses a fixed NumPy random seed, so it recreates the same data each time.

`parking_analysis.py` reads `parking_records.csv`, then creates:

- `parking_cleaned.csv`
- `parking_receipts.txt`

It also prints the complete analysis to the terminal and is redirected to `analysis_output.txt` when run with:

```bash
python parking_analysis.py > analysis_output.txt
```

## Dataset

The raw dataset has 9 columns:

- `Receipt_ID`
- `Vehicle_Number`
- `Vehicle_Type`
- `Entry_Time`
- `Exit_Time`
- `Parking_Duration`
- `Parking_Fee`
- `Payment_Method`
- `Parking_Slot`

The cleaned dataset adds `Entry_Date`, `Month_Number`, `Month`, `Weekday`, and `Entry_Hour`.

Parking rates are:

| Vehicle type | Rate |
| --- | --- |
| Bike | Rs. 10/hour |
| Car | Rs. 30/hour |
| SUV | Rs. 50/hour |

Charges are rounded up to the next started hour.

## Analysis Summary

The included output reports:

- 950 cleaned parking receipts
- Rs. 135,762 total recorded revenue
- 475 distinct vehicles
- 4.09-hour average parking duration
- March as the highest-revenue month
- UPI as the most-used payment method
- Saturday as the busiest weekday
- 17:00-18:59 as the busiest entry-hour period
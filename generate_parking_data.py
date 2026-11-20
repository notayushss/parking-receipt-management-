"""
generate_parking_data.py
------------------------
Creates a realistic SYNTHETIC dataset called  parking_records.csv  for the
project "Parking Receipt Management & Revenue Analysis".

Only built-in Python modules, NumPy and Pandas are used.
The data is reproducible because we fix the random seed (42).

How to run:
    python generate_parking_data.py
"""

import os                      # built-in module, used only to check that the file exists
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# SECTION 1 : SETTINGS (change these numbers if you want a different dataset)
# ------------------------------------------------------------------
np.random.seed(42)                      # makes the random data the same every time

NUMBER_OF_RECORDS = 950                 # unique parking transactions
NUMBER_OF_DUPLICATES = 10               # extra repeated Receipt_ID rows
MISSING_PERCENT = 0.03                  # 3% missing values in selected columns
NUMBER_OF_HIGH_FEE_RECORDS = 12         # unusually high charges (for IQR practice)
NUMBER_OF_DIFFERENT_VEHICLES = 600      # some vehicles visit more than once

START_DATE = pd.Timestamp("2026-01-01")  # data starts on 1 Jan 2026
NUMBER_OF_DAYS = 181                     # ... and ends on 30 Jun 2026

# Parking charge per hour for each vehicle type (Rupees)
HOURLY_RATE = {"Bike": 10, "Car": 30, "SUV": 50}

# Parking zones: slot names look like B-01 (Bike), C-01 (Car), S-01 (SUV)
SLOT_PREFIX = {"Bike": "B", "Car": "C", "SUV": "S"}
SLOTS_IN_ZONE = {"Bike": 6, "Car": 6, "SUV": 3}      # 15 slots in total

# ------------------------------------------------------------------
# SECTION 2 : CREATE A POOL OF INDIAN-STYLE VEHICLE NUMBERS
# Format example:  WB02AB1234  (state + RTO number + 2 letters + 4 digits)
# ------------------------------------------------------------------
# "WB" appears many times so that most vehicles are from West Bengal
state_codes = ["WB", "WB", "WB", "WB", "WB", "UP", "BR", "JH", "OD", "DL", "MH"]
letters = list("ABCDEFGHJKLMNPRSTUVWXYZ")


def make_vehicle_number():
    """Build one random vehicle registration number."""
    state = np.random.choice(state_codes)
    rto_number = np.random.randint(1, 100)               # 1 to 99
    series = "".join(np.random.choice(letters, 2))       # two random letters
    last_digits = np.random.randint(1000, 10000)         # four digits
    return f"{state}{rto_number:02d}{series}{last_digits}"


# Keep making numbers until we have enough DIFFERENT ones
vehicle_numbers_set = set()
while len(vehicle_numbers_set) < NUMBER_OF_DIFFERENT_VEHICLES:
    vehicle_numbers_set.add(make_vehicle_number())

pool_numbers = np.array(sorted(vehicle_numbers_set))

# Every vehicle in the pool has ONE fixed type (a car stays a car)
pool_types = np.random.choice(
    ["Bike", "Car", "SUV"], size=NUMBER_OF_DIFFERENT_VEHICLES, p=[0.40, 0.42, 0.18]
)

# Each parking transaction picks a random vehicle from the pool
chosen_vehicles = np.random.randint(0, NUMBER_OF_DIFFERENT_VEHICLES, size=NUMBER_OF_RECORDS)
vehicle_numbers = pool_numbers[chosen_vehicles]
vehicle_types = pool_types[chosen_vehicles]

# ------------------------------------------------------------------
# SECTION 3 : ENTRY DATE AND TIME
# Weekends are busier than weekdays, and some months are busier than others.
# ------------------------------------------------------------------
day_numbers = np.arange(NUMBER_OF_DAYS)                       # 0, 1, 2, ... 180
all_days = START_DATE + pd.to_timedelta(day_numbers, unit="D")

# Weekend days (Saturday = 5, Sunday = 6) get a higher chance of being picked
day_weights = np.where(all_days.dayofweek >= 5, 1.5, 1.0)

# Small month-wise change in traffic (Jan ... Jun)
month_factor = {1: 0.90, 2: 0.95, 3: 1.15, 4: 1.10, 5: 1.00, 6: 0.85}
day_weights = day_weights * np.array([month_factor[m] for m in all_days.month])
day_probabilities = day_weights / day_weights.sum()           # probabilities must add up to 1

chosen_day_numbers = np.random.choice(day_numbers, size=NUMBER_OF_RECORDS, p=day_probabilities)
entry_dates = START_DATE + pd.to_timedelta(chosen_day_numbers, unit="D")
is_weekend = np.array(entry_dates.dayofweek >= 5)

# Relative chance of arriving in each hour (0 to 23)
weekday_hour_weights = np.array([0.2, 0.1, 0.1, 0.1, 0.2, 0.5, 1.5, 3.5, 7, 9, 8, 6,
                                 5, 5, 4, 4, 5, 8, 9, 6, 3, 1.5, 0.8, 0.4])
weekend_hour_weights = np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.3, 0.8, 1.5, 2.5, 4, 6, 8,
                                 9, 8, 7, 7, 8, 9, 9, 8, 6, 3, 1.5, 0.7])
weekday_hour_probs = weekday_hour_weights / weekday_hour_weights.sum()
weekend_hour_probs = weekend_hour_weights / weekend_hour_weights.sum()

weekday_hours = np.random.choice(np.arange(24), size=NUMBER_OF_RECORDS, p=weekday_hour_probs)
weekend_hours = np.random.choice(np.arange(24), size=NUMBER_OF_RECORDS, p=weekend_hour_probs)
entry_hours = np.where(is_weekend, weekend_hours, weekday_hours)   # pick by day type
entry_minutes = np.random.randint(0, 60, size=NUMBER_OF_RECORDS)

entry_times = (entry_dates
               + pd.to_timedelta(entry_hours, unit="h")
               + pd.to_timedelta(entry_minutes, unit="m"))

# ------------------------------------------------------------------
# SECTION 4 : PARKING DURATION AND EXIT TIME
# ------------------------------------------------------------------
# Average stay (in hours) for each vehicle type
average_stay = {"Bike": 2.5, "Car": 3.5, "SUV": 4.0}
mean_hours = np.array([average_stay[t] for t in vehicle_types])
mean_hours = np.where(is_weekend, mean_hours * 1.2, mean_hours)   # people stay longer on weekends

# A gamma distribution gives many short stays and a few long stays (realistic)
duration_hours = np.random.gamma(shape=2.0, scale=mean_hours / 2)
duration_hours = np.clip(duration_hours, 0.25, 12)                # between 15 minutes and 12 hours

# About 4% of vehicles stay for a very long time (10 to 24 hours)
extended_stay = np.random.rand(NUMBER_OF_RECORDS) < 0.04
duration_hours[extended_stay] = np.random.uniform(10, 24, size=extended_stay.sum())

duration_minutes = np.round(duration_hours * 60).astype(int)      # whole minutes
exit_times = entry_times + pd.to_timedelta(duration_minutes, unit="m")   # always AFTER entry
parking_duration = np.round(duration_minutes / 60, 2)             # duration in hours

# ------------------------------------------------------------------
# SECTION 5 : PARKING FEE  (rate per hour x hours, any started hour is charged fully)
# ------------------------------------------------------------------
billable_hours = np.ceil(duration_minutes / 60).astype(int)       # 1h 10min counts as 2 hours
hourly_rates = np.array([HOURLY_RATE[t] for t in vehicle_types])
parking_fee = billable_hours * hourly_rates

# A few unusually high charges (for example lost-ticket / overstay penalty)
high_fee_rows = np.random.choice(NUMBER_OF_RECORDS, size=NUMBER_OF_HIGH_FEE_RECORDS, replace=False)
parking_fee[high_fee_rows] = parking_fee[high_fee_rows] + np.random.randint(
    500, 1201, size=NUMBER_OF_HIGH_FEE_RECORDS
)

# ------------------------------------------------------------------
# SECTION 6 : PAYMENT METHOD AND PARKING SLOT
# ------------------------------------------------------------------
# Payment preference changes with vehicle type: bikes use more UPI/cash, SUVs use more cards
payment_chances = {
    "Bike": [0.30, 0.60, 0.10],      # Cash, UPI, Card
    "Car": [0.15, 0.50, 0.35],
    "SUV": [0.10, 0.35, 0.55],
}
payment_methods = [
    np.random.choice(["Cash", "UPI", "Card"], p=payment_chances[t]) for t in vehicle_types
]

# Slot is picked randomly from the zone of that vehicle type
parking_slots = [
    f"{SLOT_PREFIX[t]}-{np.random.randint(1, SLOTS_IN_ZONE[t] + 1):02d}" for t in vehicle_types
]

# ------------------------------------------------------------------
# SECTION 7 : PUT EVERYTHING IN A DATAFRAME
# ------------------------------------------------------------------
df = pd.DataFrame({
    "Vehicle_Number": vehicle_numbers,
    "Vehicle_Type": vehicle_types,
    "Entry_Time": entry_times,
    "Exit_Time": exit_times,
    "Parking_Duration": parking_duration,
    "Parking_Fee": parking_fee.astype(float),
    "Payment_Method": payment_methods,
    "Parking_Slot": parking_slots,
})

# Sort by entry time and give receipt numbers in that order (RCP00001, RCP00002, ...)
df = df.sort_values("Entry_Time").reset_index(drop=True)
df.insert(0, "Receipt_ID", [f"RCP{number:05d}" for number in range(1, NUMBER_OF_RECORDS + 1)])

# Check: every Exit_Time must be later than Entry_Time
assert (df["Exit_Time"] > df["Entry_Time"]).all(), "Some Exit_Time values are not after Entry_Time!"

# Save times as simple text such as 2026-03-14 09:25:00
df["Entry_Time"] = df["Entry_Time"].dt.strftime("%Y-%m-%d %H:%M:%S")
df["Exit_Time"] = df["Exit_Time"].dt.strftime("%Y-%m-%d %H:%M:%S")

# ------------------------------------------------------------------
# SECTION 8 : ADD MISSING VALUES (3% in three columns)
# ------------------------------------------------------------------
number_missing = int(NUMBER_OF_RECORDS * MISSING_PERCENT)
for column_name in ["Vehicle_Type", "Parking_Duration", "Parking_Fee"]:
    rows_to_blank = np.random.choice(df.index, size=number_missing, replace=False)
    df.loc[rows_to_blank, column_name] = np.nan

# ------------------------------------------------------------------
# SECTION 9 : ADD DUPLICATE Receipt_ID ROWS
# ------------------------------------------------------------------
duplicate_rows = df.sample(n=NUMBER_OF_DUPLICATES, random_state=42)    # copy some random rows
df = pd.concat([df, duplicate_rows])
df = df.sort_values("Receipt_ID", kind="stable").reset_index(drop=True)  # copies sit next to originals

# ------------------------------------------------------------------
# SECTION 10 : SAVE THE FILE
# ------------------------------------------------------------------
df.to_csv("parking_records.csv", index=False)

# ------------------------------------------------------------------
# SECTION 11 : VERIFY THAT THE CSV WAS CREATED CORRECTLY
# ------------------------------------------------------------------
pd.set_option("display.max_columns", None)   # show every column when printing
pd.set_option("display.width", 200)

print("File created? ->", os.path.exists("parking_records.csv"))
check_df = pd.read_csv("parking_records.csv")
print("Rows and columns :", check_df.shape)
print("Columns          :", list(check_df.columns))
print("\nFirst 5 rows:")
print(check_df.head())
print("\nMissing values in each column:")
print(check_df.isnull().sum())
print("\nDuplicate Receipt_ID rows:", check_df.duplicated(subset="Receipt_ID").sum())
print("\nVehicle type counts:")
print(check_df["Vehicle_Type"].value_counts())
print("\nPayment method counts:")
print(check_df["Payment_Method"].value_counts())
print("\nHighest parking fees:")
print(check_df["Parking_Fee"].sort_values(ascending=False).head(5).tolist())

"""
parking_analysis.py
-------------------
Project : Parking Receipt Management & Revenue Analysis
Tools   : Python, NumPy, Pandas, basic OOP (classes and objects)

This program
  1. cleans the parking records,
  2. generates parking receipts using two classes,
  3. analyses revenue, payments, occupancy and parking duration,
  4. gives five recommendations to the parking management.

Before running this file, create the dataset once with:
    python generate_parking_data.py
Then run:
    python parking_analysis.py
"""

import os                      # built-in module, used only to check that the CSV file exists
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)   # show every column when printing
pd.set_option("display.width", 200)

# ==================================================================
# SECTION 0 : SETTINGS AND SMALL HELPER FUNCTIONS
# ==================================================================
TOTAL_CAPACITY = 15                                   # total parking slots (6 Bike + 6 Car + 3 SUV)
HOURLY_RATES = {"Bike": 10, "Car": 30, "SUV": 50}     # parking charge per hour (Rs.)
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def print_section(title):
    """Print a heading so the output is easy to read."""
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def calculate_base_fee(vehicle_type, duration_hours):
    """
    Parking rule used by the whole project:
    - every STARTED hour is charged fully (1 h 10 min is charged as 2 hours)
    - minimum charge is 1 hour
    """
    billable_hours = max(1, int(np.ceil(duration_hours)))
    return billable_hours * HOURLY_RATES[vehicle_type]


def hour_label(hour):
    """Turn 9 into the text '09:00-09:59'."""
    return f"{hour:02d}:00-{hour:02d}:59"


def hours_to_ranges(hour_list):
    """Join hours that follow each other: [0, 1, 2, 21, 22] -> '00:00-02:59, 21:00-22:59'."""
    hour_list = sorted(hour_list)
    if len(hour_list) == 0:
        return "None"
    range_texts = []
    start_hour = previous_hour = hour_list[0]
    for hour in hour_list[1:]:
        if hour != previous_hour + 1:                       # gap found -> close the current range
            range_texts.append(f"{start_hour:02d}:00-{previous_hour:02d}:59")
            start_hour = hour
        previous_hour = hour
    range_texts.append(f"{start_hour:02d}:00-{previous_hour:02d}:59")
    return ", ".join(range_texts)


# ==================================================================
# SECTION 1 : LOAD AND INSPECT THE PARKING RECORDS   (Task 1)
# ==================================================================
print_section("SECTION 1 : LOAD AND INSPECT THE DATA")

# Stop with a clear message if the dataset has not been created yet
if not os.path.exists("parking_records.csv"):
    raise SystemExit("parking_records.csv not found. Run generate_parking_data.py first.")

df = pd.read_csv("parking_records.csv")

print("Shape (rows, columns):", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn data types:")
print(df.dtypes)
print("\nBasic statistics of number columns:")
print(df.describe())

# ==================================================================
# SECTION 2 : FIND MISSING VALUES AND DUPLICATE RECEIPT IDs   (Task 2)
# ==================================================================
print_section("SECTION 2 : MISSING VALUES AND DUPLICATE Receipt_ID")

missing_count = df.isnull().sum()
missing_percent = (missing_count / len(df) * 100).round(2)
missing_report = pd.DataFrame({"Missing_Count": missing_count, "Missing_Percent": missing_percent})
print("Missing values in each column:")
print(missing_report)

# keep=False marks ALL copies of a repeated Receipt_ID so we can see them
duplicate_mask = df.duplicated(subset="Receipt_ID", keep=False)
duplicate_rows = df[duplicate_mask]
print("\nNumber of rows that belong to a repeated Receipt_ID:", len(duplicate_rows))
print("Extra (repeated) rows that will be removed         :", df.duplicated(subset="Receipt_ID").sum())
print("\nRepeated receipts:")
print(duplicate_rows[["Receipt_ID", "Vehicle_Number", "Entry_Time", "Parking_Fee"]])

# A receipt number must be unique, so we keep the first copy and remove the others
rows_before = len(df)
df = df.drop_duplicates(subset="Receipt_ID", keep="first").reset_index(drop=True)
print(f"\nRows before removing duplicates: {rows_before}")
print(f"Rows after  removing duplicates: {len(df)}")

# ==================================================================
# SECTION 3 : HANDLE MISSING VALUES   (Task 3)
# ==================================================================
print_section("SECTION 3 : HANDLE MISSING VALUES")

# Entry_Time and Exit_Time are needed to rebuild missing durations,
# so we convert them to datetime format here (Section 4 uses them further).
df["Entry_Time"] = pd.to_datetime(df["Entry_Time"])
df["Exit_Time"] = pd.to_datetime(df["Exit_Time"])

# Safety check: Exit_Time must be later than Entry_Time
invalid_rows = df["Exit_Time"] <= df["Entry_Time"]
print("Rows where Exit_Time is not after Entry_Time:", invalid_rows.sum())
df = df[~invalid_rows].reset_index(drop=True)

# --- 3a. Vehicle_Type -----------------------------------------------
# Justification: the slot name starts with a letter that shows the zone
# (B = Bike, C = Car, S = SUV), so the missing type can be recovered exactly.
slot_letter_to_type = {"B": "Bike", "C": "Car", "S": "SUV"}
type_from_slot = df["Parking_Slot"].str[0].map(slot_letter_to_type)
df["Vehicle_Type"] = df["Vehicle_Type"].fillna(type_from_slot)

# If the slot was also unusable, fall back to the most common vehicle type (mode)
if df["Vehicle_Type"].isnull().any():
    df["Vehicle_Type"] = df["Vehicle_Type"].fillna(df["Vehicle_Type"].mode()[0])

# --- 3b. Parking_Duration ------------------------------------------
# Justification: duration = Exit_Time - Entry_Time, and both times are available,
# so we calculate the real value instead of guessing with an average.
calculated_duration = ((df["Exit_Time"] - df["Entry_Time"]).dt.total_seconds() / 3600).round(2)
df["Parking_Duration"] = df["Parking_Duration"].fillna(calculated_duration)

# --- 3c. Parking_Fee -------------------------------------------------
# Justification: the fee follows a fixed rule (rate per hour x billable hours),
# so we apply the same rule to the rows where the fee is missing.
# LAMBDA #1: a small one-line function that is applied to each row.
fee_missing = df["Parking_Fee"].isnull()
if fee_missing.any():
    df.loc[fee_missing, "Parking_Fee"] = df[fee_missing].apply(
        lambda row: calculate_base_fee(row["Vehicle_Type"], row["Parking_Duration"]), axis=1
    )

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ==================================================================
# SECTION 4 : DATE / TIME FEATURES   (Task 4)
# ==================================================================
print_section("SECTION 4 : DATETIME FEATURES")

print("Data types of the time columns:")
print(df[["Entry_Time", "Exit_Time"]].dtypes)

df["Entry_Date"] = df["Entry_Time"].dt.date             # only the date
df["Month_Number"] = df["Entry_Time"].dt.month          # 1 to 12 (used for sorting)
df["Month"] = df["Entry_Time"].dt.month_name()          # January, February ...
df["Weekday"] = df["Entry_Time"].dt.day_name()          # Monday, Tuesday ...
df["Entry_Hour"] = df["Entry_Time"].dt.hour             # 0 to 23

print("\nNew columns:")
print(df[["Entry_Time", "Entry_Date", "Month", "Weekday", "Entry_Hour"]].head())

# Deliverable 1: the cleaned dataset
df.to_csv("parking_cleaned.csv", index=False)
print("\nCleaned dataset saved as parking_cleaned.csv  (", len(df), "rows )")

# ==================================================================
# SECTION 5 : OVERALL PARKING NUMBERS   (Task 5)
# ==================================================================
print_section("SECTION 5 : OVERALL PARKING SUMMARY")

total_receipts = df["Receipt_ID"].nunique()
total_vehicle_entries = len(df)
unique_vehicles = df["Vehicle_Number"].nunique()
average_duration = df["Parking_Duration"].mean()
total_revenue = df["Parking_Fee"].sum()
average_fee = df["Parking_Fee"].mean()

print(f"Total parking receipts            : {total_receipts}")
print(f"Total vehicle entries             : {total_vehicle_entries}")
print(f"Total different vehicles parked   : {unique_vehicles}")
print(f"Average parking duration          : {average_duration:.2f} hours")
print(f"Total parking revenue             : Rs. {total_revenue:,.2f}")
print(f"Average parking fee               : Rs. {average_fee:,.2f}")

# ==================================================================
# SECTION 6 : OOP - ParkingReceipt AND ParkingManagement CLASSES   (Task 6)
# ==================================================================
print_section("SECTION 6 : OOP - AUTOMATIC PARKING RECEIPTS")


class ParkingReceipt:
    """
    One object of this class = ONE parking transaction (one vehicle, one visit).
    It can find the parking time, find the fee and print a receipt.
    """

    def __init__(self, receipt_id, vehicle_number, vehicle_type, entry_time,
                 payment_method, parking_slot, exit_time=None, extra_charge=0):
        # The constructor stores the details given when the object is created
        self.receipt_id = receipt_id
        self.vehicle_number = vehicle_number
        self.vehicle_type = vehicle_type
        self.entry_time = entry_time
        self.exit_time = exit_time
        self.payment_method = payment_method
        self.parking_slot = parking_slot
        self.extra_charge = extra_charge       # penalty such as lost ticket (normally 0)

        # These values are calculated later by the methods
        self.duration_hours = 0.0
        self.billable_hours = 0
        self.base_fee = 0
        self.parking_fee = 0

    def calculate_duration(self):
        """Parking duration in hours = Exit_Time - Entry_Time."""
        if self.exit_time is None:
            print("Vehicle", self.vehicle_number, "has not exited yet.")
            return 0.0
        time_difference = self.exit_time - self.entry_time
        self.duration_hours = round(time_difference.total_seconds() / 3600, 2)
        return self.duration_hours

    def calculate_fee(self):
        """Parking fee = hourly rate x billable hours + extra charge."""
        self.billable_hours = max(1, int(np.ceil(self.duration_hours)))
        self.base_fee = calculate_base_fee(self.vehicle_type, self.duration_hours)
        self.parking_fee = self.base_fee + self.extra_charge
        return self.parking_fee

    def generate_receipt(self):
        """Return the receipt as a text that can be printed or saved."""
        line = "=" * 42
        receipt_text = (
            f"{line}\n"
            f"          CITY PARKING RECEIPT\n"
            f"{line}\n"
            f"Receipt ID      : {self.receipt_id}\n"
            f"Vehicle Number  : {self.vehicle_number}\n"
            f"Vehicle Type    : {self.vehicle_type}\n"
            f"Parking Slot    : {self.parking_slot}\n"
            f"Entry Time      : {self.entry_time.strftime('%Y-%m-%d %H:%M')}\n"
            f"Exit Time       : {self.exit_time.strftime('%Y-%m-%d %H:%M')}\n"
            f"Duration        : {self.duration_hours:.2f} hrs ({self.billable_hours} hrs billed)\n"
            f"Rate per Hour   : Rs. {HOURLY_RATES[self.vehicle_type]}\n"
            f"Parking Charge  : Rs. {self.base_fee:,.2f}\n"
            f"Extra Charge    : Rs. {self.extra_charge:,.2f}\n"
            f"TOTAL FEE       : Rs. {self.parking_fee:,.2f}\n"
            f"Payment Method  : {self.payment_method}\n"
            f"{line}"
        )
        return receipt_text


class ParkingManagement:
    """
    This class manages the whole parking lot:
    vehicles coming in, vehicles going out, records and revenue.
    """

    def __init__(self, total_capacity, id_prefix="RCP"):
        self.total_capacity = total_capacity
        self.id_prefix = id_prefix
        self.parked_vehicles = {}        # vehicles inside right now  -> {receipt_id: ParkingReceipt}
        self.completed_receipts = []     # vehicles that already left -> list of ParkingReceipt
        self.receipt_counter = 0         # used to create new receipt numbers

    def add_vehicle(self, vehicle_number, vehicle_type, entry_time,
                    payment_method, parking_slot, receipt_id=None):
        """A vehicle enters. Returns the receipt ID (or None if the parking is full)."""
        if len(self.parked_vehicles) >= self.total_capacity:
            print("Parking is FULL. Vehicle", vehicle_number, "cannot enter.")
            return None

        if receipt_id is None:           # create a new receipt number automatically
            self.receipt_counter += 1
            receipt_id = f"{self.id_prefix}{self.receipt_counter:05d}"

        new_receipt = ParkingReceipt(receipt_id, vehicle_number, vehicle_type,
                                     entry_time, payment_method, parking_slot)
        self.parked_vehicles[receipt_id] = new_receipt
        return receipt_id

    def vehicle_exit(self, receipt_id, exit_time, extra_charge=0):
        """A vehicle leaves. Calculates duration and fee and completes the receipt."""
        if receipt_id not in self.parked_vehicles:
            print("Receipt", receipt_id, "was not found in the parked vehicles.")
            return None

        receipt = self.parked_vehicles.pop(receipt_id)    # remove it from 'parked'
        receipt.exit_time = exit_time
        receipt.extra_charge = extra_charge
        receipt.calculate_duration()
        receipt.calculate_fee()
        self.completed_receipts.append(receipt)           # and keep it in the records
        return receipt

    def display_records(self, number_of_rows=5):
        """Show the completed receipts as a table."""
        records = []
        for receipt in self.completed_receipts:
            records.append({
                "Receipt_ID": receipt.receipt_id,
                "Vehicle_Number": receipt.vehicle_number,
                "Vehicle_Type": receipt.vehicle_type,
                "Duration_Hours": receipt.duration_hours,
                "Parking_Fee": receipt.parking_fee,
            })
        records_table = pd.DataFrame(records)
        print(records_table.head(number_of_rows))
        return records_table

    def calculate_revenue(self):
        """Total money collected from all completed receipts."""
        total = 0
        for receipt in self.completed_receipts:
            total = total + receipt.parking_fee
        return total

    def get_available_spaces(self):
        """Free parking slots right now."""
        return self.total_capacity - len(self.parked_vehicles)


# --- 6a. Small live demo of the two classes ------------------------
print("DEMO: a tiny parking lot with only 2 slots\n")
demo_parking = ParkingManagement(total_capacity=2, id_prefix="DEMO")
first_id = demo_parking.add_vehicle("WB02AB1234", "Car", pd.Timestamp("2026-07-01 09:00"), "UPI", "C-01")
second_id = demo_parking.add_vehicle("WB74CD5678", "Bike", pd.Timestamp("2026-07-01 09:10"), "Cash", "B-01")
demo_parking.add_vehicle("WB06EF9012", "SUV", pd.Timestamp("2026-07-01 09:20"), "Card", "S-01")  # lot is full
print("Free spaces now:", demo_parking.get_available_spaces())

finished_receipt = demo_parking.vehicle_exit(first_id, pd.Timestamp("2026-07-01 12:20"))
print("\nReceipt of the vehicle that just left:")
print(finished_receipt.generate_receipt())
print("Free spaces after one vehicle left:", demo_parking.get_available_spaces())

# --- 6b. Generate a receipt for EVERY record of the cleaned dataset ---
parking_system = ParkingManagement(total_capacity=TOTAL_CAPACITY)

for row in df.itertuples(index=False):
    # Some records contain an extra penalty above the normal rate; keep it separately
    normal_fee = calculate_base_fee(row.Vehicle_Type, row.Parking_Duration)
    extra_charge = max(0, round(row.Parking_Fee - normal_fee, 2))

    new_id = parking_system.add_vehicle(row.Vehicle_Number, row.Vehicle_Type, row.Entry_Time,
                                        row.Payment_Method, row.Parking_Slot, receipt_id=row.Receipt_ID)
    parking_system.vehicle_exit(new_id, row.Exit_Time, extra_charge)

# Deliverable 2: all receipts saved in one text file
with open("parking_receipts.txt", "w") as receipt_file:
    for receipt in parking_system.completed_receipts:
        receipt_file.write(receipt.generate_receipt() + "\n\n")

print(f"\n{len(parking_system.completed_receipts)} receipts generated and saved in parking_receipts.txt")
print("\nSample receipt (the first one):")
print(parking_system.completed_receipts[0].generate_receipt())
print("\nFirst 5 records from display_records():")
parking_system.display_records(5)

system_revenue = parking_system.calculate_revenue()
print(f"\nRevenue calculated by ParkingManagement : Rs. {system_revenue:,.2f}")
print(f"Revenue calculated from the dataset     : Rs. {total_revenue:,.2f}")
print("Both values match?                      :", bool(np.isclose(system_revenue, total_revenue)))

# ==================================================================
# SECTION 7 : VEHICLE TYPE COMPARISON   (Task 7)
# ==================================================================
print_section("SECTION 7 : REVENUE AND DURATION BY VEHICLE TYPE")

vehicle_summary = df.groupby("Vehicle_Type").agg(
    Total_Vehicles=("Receipt_ID", "count"),
    Total_Revenue=("Parking_Fee", "sum"),
    Average_Fee=("Parking_Fee", "mean"),
    Average_Duration_Hours=("Parking_Duration", "mean"),
).round(2)
vehicle_summary["Revenue_Share_%"] = (vehicle_summary["Total_Revenue"] / total_revenue * 100).round(2)
vehicle_summary = vehicle_summary.sort_values("Total_Revenue", ascending=False)
print(vehicle_summary)

# ==================================================================
# SECTION 8 : PAYMENT METHOD ANALYSIS   (Task 8)
# ==================================================================
print_section("SECTION 8 : PAYMENT METHOD ANALYSIS")

payment_summary = df.groupby("Payment_Method").agg(
    Transactions=("Receipt_ID", "count"),
    Total_Revenue=("Parking_Fee", "sum"),
    Average_Fee=("Parking_Fee", "mean"),
).round(2)
payment_summary["Volume_Share_%"] = (payment_summary["Transactions"] / len(df) * 100).round(2)
payment_summary["Revenue_Share_%"] = (payment_summary["Total_Revenue"] / total_revenue * 100).round(2)

# rank 1 = highest. method="min" gives the same rank to equal values
payment_summary["Volume_Rank"] = payment_summary["Transactions"].rank(ascending=False, method="min").astype(int)
payment_summary["Revenue_Rank"] = payment_summary["Total_Revenue"].rank(ascending=False, method="min").astype(int)
payment_summary = payment_summary.sort_values("Transactions", ascending=False)
print(payment_summary)

# ==================================================================
# SECTION 9 : OCCUPANCY - PEAK HOURS, BUSIEST DAYS, LOW DEMAND   (Task 9)
# ==================================================================
print_section("SECTION 9 : PARKING OCCUPANCY PATTERNS")

# --- 9a. How many vehicles ENTER in each hour of the day -------------
# reindex(range(24)) makes sure that hours with zero entries are also shown
hourly_entries = df["Entry_Hour"].value_counts().reindex(range(24), fill_value=0).sort_index()

peak_hours = hourly_entries.nlargest(3)
print("Top 3 peak entry hours:")
for hour, count in peak_hours.items():
    print(f"   {hour_label(hour)}  ->  {count} vehicles")

# --- 9b. Busiest weekdays ---------------------------------------------
# Each weekday does not occur the same number of times in the data period,
# so we also calculate "vehicles per day" for a fair comparison.
first_date = pd.Timestamp(df["Entry_Date"].min())
last_date = pd.Timestamp(df["Entry_Date"].max())
all_days_in_period = pd.date_range(first_date, last_date)          # every calendar day
total_days = len(all_days_in_period)
days_per_weekday = all_days_in_period.day_name().value_counts()

weekday_summary = pd.DataFrame({"Total_Vehicles": df["Weekday"].value_counts()}).reindex(WEEKDAY_ORDER)
weekday_summary["Days_In_Period"] = days_per_weekday.reindex(WEEKDAY_ORDER).values
weekday_summary["Vehicles_Per_Day"] = (weekday_summary["Total_Vehicles"] / weekday_summary["Days_In_Period"]).round(2)
print("\nVehicles by weekday:")
print(weekday_summary)
busiest_weekday = weekday_summary["Vehicles_Per_Day"].idxmax()
quietest_weekday = weekday_summary["Vehicles_Per_Day"].idxmin()
print(f"\nBusiest weekday  : {busiest_weekday}")
print(f"Quietest weekday : {quietest_weekday}")

# --- 9c. Low-demand periods -------------------------------------------
# An hour is "low demand" if its entries are less than half of the average hourly entries
average_hourly_entries = hourly_entries.mean()
low_demand_hours = hourly_entries[hourly_entries < 0.5 * average_hourly_entries]
print(f"\nAverage entries per hour of the day: {average_hourly_entries:.1f}")
print("Low-demand hours (less than half of the average):")
print("   " + hours_to_ranges(list(low_demand_hours.index)))

# --- 9d. Hourly parking occupancy trend (Deliverable 5) ---------------
# For every parking record we count the clock-hours during which the vehicle was inside.
vehicles_present_by_hour = np.zeros(24)
for row in df.itertuples(index=False):
    current_hour = row.Entry_Time.replace(minute=0, second=0, microsecond=0)   # start of entry hour
    while current_hour < row.Exit_Time:
        vehicles_present_by_hour[current_hour.hour] += 1
        current_hour = current_hour + pd.Timedelta(hours=1)

average_vehicles_present = vehicles_present_by_hour / total_days    # per day, on average

hourly_trend = pd.DataFrame({
    "Hour_Slot": [hour_label(hour) for hour in range(24)],
    "Vehicles_Entered": hourly_entries.values,
    "Avg_Vehicles_Inside": average_vehicles_present.round(2),
    "Avg_Occupancy_%": (average_vehicles_present / TOTAL_CAPACITY * 100).round(2),
})
print("\nHourly parking occupancy trend:")
print(hourly_trend.to_string(index=False))

# ==================================================================
# SECTION 10 : MONTHLY REVENUE   (Task 10)
# ==================================================================
print_section("SECTION 10 : MONTHLY PARKING REVENUE")

monthly_revenue = df.groupby(["Month_Number", "Month"]).agg(
    Total_Vehicles=("Receipt_ID", "count"),
    Total_Revenue=("Parking_Fee", "sum"),
    Average_Fee=("Parking_Fee", "mean"),
).round(2).reset_index()

monthly_revenue = monthly_revenue.sort_values("Month_Number")          # January first
print(monthly_revenue.drop(columns="Month_Number").to_string(index=False))

top_three_months = monthly_revenue.nlargest(3, "Total_Revenue")
print("\nTop 3 revenue-generating months:")
for rank_number, month_row in enumerate(top_three_months.itertuples(index=False), start=1):
    print(f"   {rank_number}. {month_row.Month:<10} Rs. {month_row.Total_Revenue:,.2f}")

# ==================================================================
# SECTION 11 : NUMPY STATISTICS   (Task 11)
# ==================================================================
print_section("SECTION 11 : NUMPY STATISTICS")

duration_values = df["Parking_Duration"].to_numpy()      # convert columns to NumPy arrays
fee_values = df["Parking_Fee"].to_numpy()

print(f"{'':<22}{'Duration (hours)':>18}{'Fee (Rs.)':>14}")
print(f"{'Mean':<22}{np.mean(duration_values):>18.2f}{np.mean(fee_values):>14.2f}")
print(f"{'Median':<22}{np.median(duration_values):>18.2f}{np.median(fee_values):>14.2f}")
print(f"{'Standard deviation':<22}{np.std(duration_values):>18.2f}{np.std(fee_values):>14.2f}")

# ==================================================================
# SECTION 12 : UNUSUALLY HIGH PARKING CHARGES (IQR METHOD)   (Task 12)
# ==================================================================
print_section("SECTION 12 : OUTLIERS IN PARKING FEE (IQR METHOD)")

q1 = np.percentile(fee_values, 25)       # 25% of fees are below this value
q3 = np.percentile(fee_values, 75)       # 75% of fees are below this value
iqr = q3 - q1                            # Inter-Quartile Range
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
print(f"Q1 = {q1:.2f}   Q3 = {q3:.2f}   IQR = {iqr:.2f}")
print(f"Lower fence = {lower_fence:.2f}   Upper fence = {upper_fence:.2f}")

# LAMBDA #2: label each fee as "Unusually High" or "Normal"
df["Fee_Status"] = df["Parking_Fee"].apply(lambda fee: "Unusually High" if fee > upper_fence else "Normal")

high_fee_records = df[df["Fee_Status"] == "Unusually High"].sort_values("Parking_Fee", ascending=False)
print(f"\nUnusually high charges found: {len(high_fee_records)} "
      f"({len(high_fee_records) / len(df) * 100:.2f}% of all receipts)")
print("\nTop 10 unusually high charges:")
print(high_fee_records[["Receipt_ID", "Vehicle_Type", "Parking_Duration", "Parking_Fee", "Payment_Method"]].head(10).to_string(index=False))

# ==================================================================
# SECTION 13 : PARKING DURATION CATEGORIES   (Task 13)
# ==================================================================
print_section("SECTION 13 : PARKING DURATION CATEGORIES")

duration_bins = [0, 2, 5, 10, np.inf]          # np.inf means "no upper limit"
duration_labels = ["Short Stay (0-2 hrs)", "Medium Stay (2-5 hrs)",
                   "Long Stay (5-10 hrs)", "Extended Stay (10+ hrs)"]
df["Stay_Category"] = pd.cut(df["Parking_Duration"], bins=duration_bins, labels=duration_labels)

stay_summary = df.groupby("Stay_Category", observed=False).agg(
    Vehicles=("Receipt_ID", "count"),
    Total_Revenue=("Parking_Fee", "sum"),
    Average_Fee=("Parking_Fee", "mean"),
    Average_Duration_Hours=("Parking_Duration", "mean"),
).round(2)
stay_summary["Vehicle_Share_%"] = (stay_summary["Vehicles"] / len(df) * 100).round(2)
stay_summary["Revenue_Share_%"] = (stay_summary["Total_Revenue"] / total_revenue * 100).round(2)
print(stay_summary)

# ==================================================================
# SECTION 14 : OCCUPANCY AND SPACE AVAILABILITY   (Task 14)
# ==================================================================
print_section(f"SECTION 14 : OCCUPANCY AND AVAILABILITY (capacity = {TOTAL_CAPACITY} slots)")

# --- 14a. How many vehicles are inside at every moment? ---------------
# +1 when a vehicle enters, -1 when a vehicle leaves; the running total = vehicles inside.
entry_events = pd.DataFrame({"Time": df["Entry_Time"], "Change": 1})
exit_events = pd.DataFrame({"Time": df["Exit_Time"], "Change": -1})
all_events = pd.concat([entry_events, exit_events])
all_events = all_events.sort_values(["Time", "Change"]).reset_index(drop=True)   # exits first if same time
all_events["Vehicles_Inside"] = all_events["Change"].cumsum()

peak_vehicles = int(all_events["Vehicles_Inside"].max())
peak_time = all_events.loc[all_events["Vehicles_Inside"].idxmax(), "Time"]
times_lot_was_full = int((all_events["Vehicles_Inside"] >= TOTAL_CAPACITY).sum())

print(f"Maximum vehicles parked at the same time : {peak_vehicles}  (on {peak_time})")
print(f"Peak occupancy                           : {peak_vehicles / TOTAL_CAPACITY * 100:.1f}%")
print(f"Free spaces at that peak moment          : {TOTAL_CAPACITY - peak_vehicles}")
print(f"Times the parking lot reached full       : {times_lot_was_full}")

# --- 14b. Overall space utilisation -----------------------------------
total_parked_hours = df["Parking_Duration"].sum()
available_slot_hours = TOTAL_CAPACITY * total_days * 24
overall_utilisation = total_parked_hours / available_slot_hours * 100
print(f"\nTotal vehicle-hours parked               : {total_parked_hours:,.1f}")
print(f"Total slot-hours available               : {available_slot_hours:,}")
print(f"Overall space utilisation                : {overall_utilisation:.2f}%")

# --- 14c. Typical free spaces in the busiest and quietest hours ---------
hourly_trend["Avg_Free_Spaces"] = (TOTAL_CAPACITY - hourly_trend["Avg_Vehicles_Inside"]).round(2)
busiest_hour_row = hourly_trend.loc[hourly_trend["Avg_Vehicles_Inside"].idxmax()]
quietest_hour_row = hourly_trend.loc[hourly_trend["Avg_Vehicles_Inside"].idxmin()]
print(f"\nBusiest hour of the day  : {busiest_hour_row['Hour_Slot']}  "
      f"(avg {busiest_hour_row['Avg_Vehicles_Inside']} vehicles inside, "
      f"{busiest_hour_row['Avg_Free_Spaces']} free spaces)")
print(f"Quietest hour of the day : {quietest_hour_row['Hour_Slot']}  "
      f"(avg {quietest_hour_row['Avg_Vehicles_Inside']} vehicles inside, "
      f"{quietest_hour_row['Avg_Free_Spaces']} free spaces)")


# --- 14d. Availability at any chosen time ----------------------------
def parking_status_at(time_point):
    """How many vehicles are inside, and how many slots are free, at a given time?"""
    still_inside = df[(df["Entry_Time"] <= time_point) & (df["Exit_Time"] > time_point)]
    occupied = len(still_inside)
    return occupied, TOTAL_CAPACITY - occupied


sample_time = pd.Timestamp("2026-03-14 18:00:00")
occupied_now, free_now = parking_status_at(sample_time)
print(f"\nStatus at {sample_time}: {occupied_now} slots occupied, {free_now} slots free "
      f"({occupied_now / TOTAL_CAPACITY * 100:.1f}% occupancy)")

# ==================================================================
# SECTION 15 : FIVE RECOMMENDATIONS   (Task 15)
# ==================================================================
print_section("SECTION 15 : RECOMMENDATIONS FOR PARKING MANAGEMENT")

# Numbers used inside the recommendations (taken from the analysis above)
peak_hour_text = ", ".join([hour_label(hour) for hour in peak_hours.index])
peak_share = peak_hours.sum() / len(df) * 100
low_hour_text = hours_to_ranges(list(low_demand_hours.index))
top_vehicle_type = vehicle_summary.index[0]                      # highest total revenue
top_vehicle_share = vehicle_summary["Revenue_Share_%"].iloc[0]
best_fee_type = vehicle_summary["Average_Fee"].idxmax()          # highest fee per visit
best_fee_value = vehicle_summary["Average_Fee"].max()
cash_share = payment_summary.loc["Cash", "Volume_Share_%"]
digital_share = 100 - cash_share
extended_share = stay_summary.loc["Extended Stay (10+ hrs)", "Vehicle_Share_%"]
extended_revenue_share = stay_summary.loc["Extended Stay (10+ hrs)", "Revenue_Share_%"]
outlier_count = len(high_fee_records)
short_stay_high_fee = int((high_fee_records["Parking_Duration"] <= 5).sum())   # high fee but NOT a long stay

print(f"1. PEAK-HOUR PRICING: The busiest entry hours are {peak_hour_text}, which bring "
      f"{peak_share:.1f}% of all vehicles. Charge a slightly higher hourly rate in these hours "
      f"and show live 'slots available' boards to spread the traffic.")
print(f"\n2. OFF-PEAK OFFERS: Very few vehicles arrive during {low_hour_text}. "
      f"Offer early-bird / night discounts or monthly passes so that empty slots earn money.")
print(f"\n3. PROMOTE DIGITAL PAYMENTS: {digital_share:.1f}% of payments are already UPI or card. "
      f"Put UPI QR codes at every exit gate and give a small discount for digital payment "
      f"to reduce cash handling, queues and chances of cash leakage.")
print(f"\n4. OFFER A DAILY CAP FOR LONG STAYS: {top_vehicle_type}s bring the most revenue "
      f"({top_vehicle_share:.1f}%), while {best_fee_type}s pay the most per visit (Rs. {best_fee_value:.0f}). "
      f"Extended stays (10+ hrs) are only {extended_share:.1f}% of vehicles but give "
      f"{extended_revenue_share:.1f}% of revenue. A fixed daily / overnight pass gives these customers a "
      f"clear price and moves some parking into the quiet night hours.")
print(f"\n5. AUDIT HIGH CHARGES AND USE SPARE CAPACITY: {outlier_count} receipts are unusually high by the "
      f"IQR method; {short_stay_high_fee} of them are stays of 5 hours or less, so check those first for "
      f"lost-ticket penalties or billing mistakes. The lot used only {overall_utilisation:.1f}% of its "
      f"slot-hours and peaked at {peak_vehicles} of {TOTAL_CAPACITY} slots, so adding more space is not "
      f"needed; extra income can come from reserved slots or monthly passes.")

print("\n" + "=" * 75)
print("PROGRAM FINISHED - files created: parking_cleaned.csv, parking_receipts.txt")
print("=" * 75)

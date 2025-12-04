"""campus_energy_dashboard.py

Complete script implementing:
- Task 1: Read multiple CSVs from /data/ and merge into df_combined (with robust error handling)
- Task 2: Aggregation functions for daily/weekly totals and building summaries
- Task 3: OOP modeling (Building, MeterReading, BuildingManager)
- Task 4: Matplotlib dashboard (trend line, bar chart, scatter plot) saved as dashboard.png
- Task 5: Persistence: cleaned CSVs, building_summary.csv, summary.txt

The script will also create sample data into ./data/ if no CSVs are found, so you can run it out-of-the-box.

Usage:
$ python campus_energy_dashboard.py

Outputs will be written to ./output/ (created if missing).

"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
import os
import textwrap

# ---------- Configuration ----------
DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_FILE = OUTPUT_DIR / "processing.log"

# Logging setup
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler(LOG_FILE, mode='w')
                    ])
logger = logging.getLogger(__name__)

# ---------- Sample Data Generator (helpful for quick testing) ----------

def generate_sample_data(data_dir: Path, n_buildings=3, months=3):
    """Generate sample CSV files like buildingA_2025-01.csv with hourly kWh readings."""
    data_dir.mkdir(exist_ok=True)
    base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    for b in range(1, n_buildings + 1):
        building = f"Building{b}"
        for m in range(months):
            month_start = (base_date - pd.DateOffset(months=m)).replace(day=1)
            # create hourly timestamps for the month
            period_end = (month_start + pd.DateOffset(months=1)) - pd.Timedelta(hours=1)
            rng = pd.date_range(start=month_start, end=period_end, freq='H')
            # sample kwh with daily pattern + noise
            hours = rng.hour
            base = 0.5 + 0.5 * np.sin(2 * np.pi * (hours - 6) / 24)  # morning peak
            kwh = np.abs(base * (1 + 0.1 * np.random.randn(len(rng)))) * (10 + b)  # vary by building

            df = pd.DataFrame({
                'timestamp': rng,
                'kwh': kwh
            })
            fname = data_dir / f"{building}_{month_start.strftime('%Y-%m')}.csv"
            df.to_csv(fname, index=False)
    logger.info(f"Generated sample data for {n_buildings} buildings, {months} months each in {data_dir}")

# ---------- Task 1: Read and Combine CSVs ----------

def read_all_csvs(data_dir: Path) -> pd.DataFrame:
    """Read all CSVs from data_dir, robust to missing/corrupt files.

    Rules applied:
    - Accept files with columns that contain a timestamp and some kwh-like column.
    - If building or month metadata missing, infer from filename.
    - Skip bad lines in corrupt files.
    - Log missing or invalid files.
    """
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        logger.warning(f"No CSV files found in {data_dir}")
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            # Use pandas robust parameters to skip bad lines
            df = pd.read_csv(f, engine='python', on_bad_lines='skip')
            if df.empty:
                logger.warning(f"File {f.name} read but empty - skipping")
                continue

            # Normalize column names
            df.columns = [c.strip().lower() for c in df.columns]

            # Try to find timestamp and kwh columns
            ts_col = None
            kwh_col = None
            for c in df.columns:
                if 'time' in c or 'timestamp' in c or 'date' in c:
                    ts_col = c
                if 'kwh' in c or 'kw' in c or ('energy' in c and 'kwh' in c):
                    kwh_col = c
            # Fallback possibilities
            if not ts_col and 'index' in df.columns:
                ts_col = 'index'
            if not kwh_col:
                # try the second column as kwh
                possible = [c for c in df.columns if c not in (ts_col,)]
                if possible:
                    kwh_col = possible[0]

            if not ts_col or not kwh_col:
                logger.error(f"File {f.name} doesn't contain obvious timestamp/kwh columns - skipping")
                continue

            # Keep only needed columns and rename
            df = df[[ts_col, kwh_col]].rename(columns={ts_col: 'timestamp', kwh_col: 'kwh'})

            # Parse timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df['kwh'] = pd.to_numeric(df['kwh'], errors='coerce')
            df = df.dropna(subset=['timestamp', 'kwh'])

            # Add metadata inferred from filename
            name_parts = f.stem.split('_')
            building_name = name_parts[0] if name_parts else 'Unknown'
            month_info = None
            if len(name_parts) > 1:
                month_info = name_parts[1]
            df['building'] = building_name
            if month_info:
                df['month'] = month_info
            else:
                df['month'] = df['timestamp'].dt.to_period('M').astype(str)

            frames.append(df)
            logger.info(f"Successfully read {f.name} ({len(df)} rows)")
        except FileNotFoundError:
            logger.exception(f"File not found: {f}")
        except pd.errors.ParserError:
            logger.exception(f"Parser error reading {f.name} - skipping file")
        except Exception as e:
            logger.exception(f"Unexpected error reading {f.name}: {e}")

    if not frames:
        logger.error("No valid data frames could be created from CSVs")
        return pd.DataFrame()

    df_combined = pd.concat(frames, ignore_index=True)
    # Ensure timestamp is datetime and set index
    df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'])
    df_combined = df_combined.sort_values('timestamp')
    df_combined = df_combined.reset_index(drop=True)

    return df_combined

# ---------- Task 2: Aggregation Functions ----------

def calculate_daily_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Return daily totals for all buildings combined and per-building daily totals."""
    if df.empty:
        return pd.DataFrame()
    df2 = df.copy()
    df2 = df2.set_index('timestamp')
    # overall daily totals
    daily_total = df2['kwh'].resample('D').sum().rename('total_kwh').to_frame()
    # building-wise daily totals
    building_daily = df2.groupby('building')['kwh'].resample('D').sum().unstack(level=0).fillna(0)
    return daily_total.join(building_daily, how='left')


def calculate_weekly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df2 = df.copy().set_index('timestamp')
    weekly_total = df2['kwh'].resample('W').sum().rename('total_kwh').to_frame()
    building_weekly = df2.groupby('building')['kwh'].resample('W').sum().unstack(level=0).fillna(0)
    return weekly_total.join(building_weekly, how='left')


def building_wise_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table per building (mean, min, max, total)."""
    if df.empty:
        return pd.DataFrame()
    grp = df.groupby('building')['kwh']
    summary = grp.agg(['mean', 'min', 'max', 'sum']).rename(columns={'sum': 'total'})
    summary = summary.reset_index()
    return summary

# ---------- Task 3: Object-Oriented Modeling ----------

class MeterReading:
    def __init__(self, timestamp: pd.Timestamp, kwh: float):
        self.timestamp = pd.to_datetime(timestamp)
        self.kwh = float(kwh)

    def __repr__(self):
        return f"MeterReading({self.timestamp}, {self.kwh:.3f} kWh)"


class Building:
    def __init__(self, name: str):
        self.name = name
        self.meter_readings = []  # list of MeterReading

    def add_reading(self, reading: MeterReading):
        self.meter_readings.append(reading)

    def calculate_total_consumption(self) -> float:
        return sum(r.kwh for r in self.meter_readings)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.meter_readings:
            return pd.DataFrame(columns=['timestamp', 'kwh'])
        df = pd.DataFrame([{'timestamp': r.timestamp, 'kwh': r.kwh} for r in self.meter_readings])
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    def generate_report(self) -> dict:
        df = self.to_dataframe()
        if df.empty:
            return {'name': self.name, 'mean': 0, 'min': 0, 'max': 0, 'total': 0}
        return {
            'name': self.name,
            'mean': df['kwh'].mean(),
            'min': df['kwh'].min(),
            'max': df['kwh'].max(),
            'total': df['kwh'].sum()
        }


class BuildingManager:
    def __init__(self):
        self.buildings = {}

    def ingest_dataframe(self, df: pd.DataFrame):
        if df.empty:
            return
        for _, row in df.iterrows():
            bname = row.get('building', 'Unknown')
            if bname not in self.buildings:
                self.buildings[bname] = Building(bname)
            reading = MeterReading(row['timestamp'], row['kwh'])
            self.buildings[bname].add_reading(reading)

    def summaries(self) -> dict:
        return {name: b.generate_report() for name, b in self.buildings.items()}

# ---------- Task 4: Visualization (Matplotlib) ----------

def create_dashboard(df: pd.DataFrame, daily_df: pd.DataFrame, weekly_df: pd.DataFrame, output_path: Path):
    """Create a 3-panel dashboard and save as PNG.

    Panels:
    1. Trend Line – daily consumption over time for all buildings.
    2. Bar Chart – compare average weekly usage across buildings.
    3. Scatter Plot – peak-hour consumption vs. time/building.
    """
    if df.empty:
        logger.error("No data to plot")
        return

    # Prepare data
    # daily totals per building (from daily_df)
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), constrained_layout=True)

    # 1 - Trend line: total daily consumption
    axes[0].plot(daily_df.index, daily_df['total_kwh'], label='Total daily kWh')
    axes[0].set_title('Daily Total Consumption')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('kWh')
    axes[0].legend()

    # 2 - Bar chart: average weekly usage across buildings
    # compute average weekly per building from weekly_df
    if not weekly_df.empty:
        # building columns are present in weekly_df beyond 'total_kwh'
        building_cols = [c for c in weekly_df.columns if c != 'total_kwh']
        avg_weekly = weekly_df[building_cols].mean().sort_values(ascending=False)
        axes[1].bar(avg_weekly.index, avg_weekly.values)
        axes[1].set_title('Average Weekly Usage by Building')
        axes[1].set_xlabel('Building')
        axes[1].set_ylabel('Avg kWh per week')
        axes[1].tick_params(axis='x', rotation=45)
    else:
        axes[1].text(0.5, 0.5, 'No weekly data', ha='center')

    # 3 - Scatter plot: peak-hour consumption (we'll define peak-hour as top 5% readings)
    df2 = df.copy()
    # find top 5% kwh readings
    quant = df2['kwh'].quantile(0.95)
    peaks = df2[df2['kwh'] >= quant]
    if not peaks.empty:
        # encode building to numeric for plotting y position
        buildings = sorted(df2['building'].unique())
        mapping = {b: i for i, b in enumerate(buildings)}
        axes[2].scatter(peaks['timestamp'], peaks['kwh'], alpha=0.7)
        axes[2].set_title('Peak-Hour Consumption (top 5%)')
        axes[2].set_xlabel('Timestamp')
        axes[2].set_ylabel('kWh')
    else:
        axes[2].text(0.5, 0.5, 'No peak readings found', ha='center')

    plt.suptitle('Campus Energy Dashboard', fontsize=16)
    out_file = output_path / 'dashboard.png'
    plt.savefig(out_file, dpi=150)
    plt.close(fig)
    logger.info(f"Dashboard saved to {out_file}")

# ---------- Task 5: Persistence and Executive Summary ----------

def persist_results(df_combined: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path):
    cleaned_file = output_dir / 'cleaned_energy_data.csv'
    summary_file = output_dir / 'building_summary.csv'
    summary_txt = output_dir / 'summary.txt'

    df_combined.to_csv(cleaned_file, index=False)
    summary_df.to_csv(summary_file, index=False)

    # Build a short text summary
    total_campus = df_combined['kwh'].sum()
    building_summ = summary_df.copy()
    if not building_summ.empty:
        highest_building = building_summ.sort_values('total', ascending=False).iloc[0]
        highest_name = highest_building['building']
        highest_total = highest_building['total']
    else:
        highest_name = 'N/A'
        highest_total = 0

    # Peak load time
    if not df_combined.empty:
        peak_row = df_combined.loc[df_combined['kwh'].idxmax()]
        peak_time = peak_row['timestamp']
        peak_value = peak_row['kwh']
    else:
        peak_time = 'N/A'
        peak_value = 0

    text = textwrap.dedent(f"""
    Campus Energy Summary Report
    ----------------------------
    Total campus consumption (kWh): {total_campus:,.2f}
    Highest-consuming building: {highest_name} ({highest_total:,.2f} kWh)
    Peak single measurement: {peak_value:,.2f} kWh at {peak_time}

    Notes:
    - Cleaned dataset: {cleaned_file}
    - Building summary CSV: {summary_file}
    - Visual dashboard: {output_dir / 'dashboard.png'}
    """)

    summary_txt.write_text(text)
    logger.info(f"Persisted cleaned data, summary CSV, and summary text to {output_dir}")
    print(text)

# ---------- Main orchestration ----------

def main():
    # Create data folder and sample data if empty
    if not DATA_DIR.exists() or not any(DATA_DIR.glob('*.csv')):
        logger.info("No input CSVs found. Generating sample data...")
        generate_sample_data(DATA_DIR, n_buildings=4, months=3)

    df_combined = read_all_csvs(DATA_DIR)
    if df_combined.empty:
        logger.error("No combined data available. Exiting.")
        return

    # Ensure we have a 'total' column by grouping if multiple readings per timestamp exist
    # For safety, keep original readings (don't aggregate here yet)

    # Create daily and weekly aggregates
    daily = calculate_daily_totals(df_combined)
    weekly = calculate_weekly_aggregates(df_combined)
    summary = building_wise_summary(df_combined)

    # Add total_kwh to daily if not present
    if 'total_kwh' not in daily.columns:
        daily['total_kwh'] = daily.sum(axis=1)

    # OOP ingestion
    manager = BuildingManager()
    manager.ingest_dataframe(df_combined)
    building_summaries = manager.summaries()
    logger.info(f"Generated building summaries for {len(building_summaries)} buildings")

    # Visualize
    create_dashboard(df_combined, daily, weekly, OUTPUT_DIR)

    # Persist
    persist_results(df_combined, summary, OUTPUT_DIR)

    logger.info("All tasks completed.")


if __name__ == '__main__':
    main()

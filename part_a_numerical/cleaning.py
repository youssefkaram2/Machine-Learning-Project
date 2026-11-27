import pandas as pd
import numpy as np

REF_YEAR = 2025
MIN_PRICE = 2000
MAX_PRICE = 500000

NUM_COLS = ["milage", "car_age", "horsepower", "engine_liters", "cylinders"]
CAT_COLS = ["brand", "fuel_type", "transmission_type", "accident", "clean_title"]


def clean_data(df):
    df = df.copy()

    # 1. remove duplicate rows
    df = df.drop_duplicates()

    # 2. price and milage: text to number (same steps as the exploration notebook)
    df["price"] = df["price"].str.replace("$", "", regex=False).str.replace(",", "", regex=False)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["milage"] = df["milage"].str.replace(" mi.", "", regex=False).str.replace(",", "", regex=False)
    df["milage"] = pd.to_numeric(df["milage"], errors="coerce")

    # 3. drop rows without price and rows with extreme prices
    df = df.dropna(subset=["price"])
    df = df[(df["price"] >= MIN_PRICE) & (df["price"] <= MAX_PRICE)].copy()

    # 4. car age instead of model year
    df["car_age"] = REF_YEAR - df["model_year"]

    # 5. numbers from the engine text
    engine = df["engine"].fillna("")
    df["horsepower"] = engine.str.extract(r"(\d+\.?\d*)\s*HP", expand=False).astype(float)
    df["engine_liters"] = engine.str.extract(r"(\d+\.?\d*)\s*(?:L|Liter)\b", expand=False).astype(float)
    cyl_a = engine.str.extract(r"(\d+)\s*Cylinder", expand=False)
    cyl_b = engine.str.extract(r"\b[VIHW](\d{1,2})\b", expand=False)
    df["cylinders"] = cyl_a.fillna(cyl_b).astype(float)

    # 6. electric cars have no liters and no cylinders (only fill where the value is missing)
    is_electric = engine.str.contains(r"Electric|Battery|Motor|kW|\dAh") & ~engine.str.contains("Gas|Hybrid")
    df.loc[is_electric & df["engine_liters"].isna(), "engine_liters"] = 0
    df.loc[is_electric & df["cylinders"].isna(), "cylinders"] = 0

    # 7. fuel type: keep only the valid values, everything else becomes missing
    valid_fuels = ["Gasoline", "Hybrid", "E85 Flex Fuel", "Diesel", "Plug-In Hybrid"]
    df.loc[~df["fuel_type"].isin(valid_fuels), "fuel_type"] = np.nan
    df.loc[df["fuel_type"].isna() & is_electric, "fuel_type"] = "Electric"
    df["fuel_type"] = df["fuel_type"].fillna("Unknown")

    # 8. simple transmission type (later lines override earlier ones)
    t = df["transmission"].fillna("").str.lower()
    df["transmission_type"] = "Other"
    df.loc[t.str.contains("m/t|manual"), "transmission_type"] = "Manual"
    df.loc[t.str.contains("a/t|automatic"), "transmission_type"] = "Automatic"
    df.loc[t.str.contains("dual"), "transmission_type"] = "Dual-Shift"
    df.loc[t.str.contains("cvt|variable"), "transmission_type"] = "CVT"

    # 9. accident: missing means unknown, not "no accident"
    df["accident"] = df["accident"].replace({"None reported": "No",
                                             "At least 1 accident or damage reported": "Yes"})
    df["accident"] = df["accident"].fillna("Unknown")

    # clean_title: only "Yes" or empty, empty means not reported (not "no")
    df["clean_title"] = df["clean_title"].fillna("Unknown")

    # 10. keep only the useful columns
    return df[CAT_COLS + NUM_COLS + ["price"]]

import pandas as pd
import numpy as np
import shutil
from pathlib import Path

# Read the downloaded UCI dataset
df = pd.read_excel("Online Retail.xlsx", engine="openpyxl")

# Clean data: remove cancelled invoices and invalid sales
df["InvoiceNo"] = df["InvoiceNo"].astype(str)
df = df[~df["InvoiceNo"].str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
df = df.dropna(subset=["StockCode", "Description", "InvoiceDate"])

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"]).dt.normalize()
df["StockCode"] = df["StockCode"].astype(str)
df["Description"] = df["Description"].astype(str).str.strip()

# Select the 20 products with the highest total sales quantity
top_products = (
    df.groupby("StockCode")["Quantity"]
    .sum()
    .nlargest(20)
    .index
)

df = df[df["StockCode"].isin(top_products)].copy()

# Aggregate transactions into daily product sales
sales = (
    df.groupby(["InvoiceDate", "StockCode"], as_index=False)
    .agg(
        product_name=("Description", "first"),
        units_sold=("Quantity", "sum"),
        unit_price=("UnitPrice", "median")
    )
)

# Fill dates with zero sales for each selected product
all_dates = pd.date_range(
    sales["InvoiceDate"].min(),
    sales["InvoiceDate"].max(),
    freq="D"
)

product_info = sales.groupby("StockCode").agg(
    product_name=("product_name", "first"),
    unit_price=("unit_price", "median")
)

full_index = pd.MultiIndex.from_product(
    [all_dates, product_info.index],
    names=["date", "product_id"]
)

sales = (
    sales.rename(columns={
        "InvoiceDate": "date",
        "StockCode": "product_id"
    })
    .set_index(["date", "product_id"])
    .reindex(full_index)
    .reset_index()
)

sales["units_sold"] = sales["units_sold"].fillna(0)
sales["product_name"] = sales["product_id"].map(
    product_info["product_name"]
)
sales["unit_price"] = sales["product_id"].map(
    product_info["unit_price"]
)

sales["date"] = sales["date"].dt.strftime("%Y-%m-%d")
sales.to_csv("sample_sales.csv", index=False)

# Create inventory estimates (the dataset has no actual stock levels)
daily_avg = sales.groupby("product_id")["units_sold"].mean()

inventory = product_info.reset_index()
inventory["category"] = "Online Retail"
inventory["current_stock"] = inventory["StockCode"].map(
    lambda x: max(1, int(np.ceil(daily_avg[x] * 7)))
)
inventory["lead_time_days"] = 5
inventory["safety_stock"] = inventory["StockCode"].map(
    lambda x: max(1, int(np.ceil(daily_avg[x] * 3)))
)

inventory = inventory.rename(columns={
    "StockCode": "product_id"
})

inventory = inventory[
    ["product_id", "product_name", "category", "current_stock",
     "unit_price", "lead_time_days", "safety_stock"]
]
inventory.to_csv("inventory.csv", index=False)

print("Success! Real sales data prepared.")
print("Products:", len(inventory))
print("Daily sales records:", len(sales))
print("Updated sample_sales.csv and inventory.csv")
print("Note: stock, lead time and safety stock are estimates.")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Smart Inventory Manager", page_icon="📦", layout="wide")
st.title("📦 Smart Inventory Management System")
st.caption("Inventory-first operations dashboard with demand forecasting and replenishment guidance")

@st.cache_data
def read_data():
    return pd.read_csv("sample_sales.csv", parse_dates=["date"]), pd.read_csv("inventory.csv")

sales_default, inventory_default = read_data()
if "sales" not in st.session_state: st.session_state.sales = sales_default.copy()
if "inventory" not in st.session_state: st.session_state.inventory = inventory_default.copy()
if "transactions" not in st.session_state:
    st.session_state.transactions = pd.DataFrame(columns=["timestamp","product_id","product_name","type","quantity","note"])

sales = st.session_state.sales
inventory = st.session_state.inventory.copy()
inventory["current_stock"] = pd.to_numeric(inventory["current_stock"], errors="coerce").fillna(0).clip(lower=0)
inventory["lead_time_days"] = pd.to_numeric(inventory["lead_time_days"], errors="coerce").fillna(1).clip(lower=1)
inventory["safety_stock"] = pd.to_numeric(inventory["safety_stock"], errors="coerce").fillna(0).clip(lower=0)

with st.sidebar:
    st.header("Controls")
    horizon = st.slider("Forecast horizon (days)", 3, 14, 7)
    cover_days = st.slider("Target cover after lead time (days)", 7, 30, 14)
    st.caption("Sample data is synthetic. Changes made in this session are temporary.")

def forecast_for_product(group, horizon):
    daily = group.groupby("date")["units_sold"].sum().sort_index().asfreq("D", fill_value=0)
    if len(daily) < 15:
        avg = float(daily.tail(min(7, len(daily))).mean()) if len(daily) else 0
        return np.repeat(max(0, avg), horizon), "7-day average baseline", None
    feat = pd.DataFrame({"y":daily.values}, index=daily.index)
    feat["dow"] = feat.index.dayofweek
    feat["t"] = np.arange(len(feat))
    feat["lag1"] = feat["y"].shift(1)
    feat["lag7"] = feat["y"].shift(7)
    feat["roll7"] = feat["y"].shift(1).rolling(7).mean()
    feat = feat.dropna()
    X=feat[["dow","t","lag1","lag7","roll7"]]; y=feat["y"]
    split=max(1,int(len(X)*0.8))
    model=RandomForestRegressor(n_estimators=100,random_state=42,min_samples_leaf=2)
    model.fit(X.iloc[:split],y.iloc[:split])
    mae=mean_absolute_error(y.iloc[split:],np.maximum(0,model.predict(X.iloc[split:]))) if split<len(X) else np.nan
    model.fit(X,y)
    history=list(daily.values.astype(float)); future=[]
    for step in range(1,horizon+1):
        d=daily.index[-1]+pd.Timedelta(days=step)
        row=pd.DataFrame([{"dow":d.dayofweek,"t":len(history),"lag1":history[-1],
                           "lag7":history[-7] if len(history)>=7 else history[-1],
                           "roll7":float(np.mean(history[-7:]))}])
        pred=max(0,float(model.predict(row)[0]))
        future.append(pred); history.append(pred)
    return np.array(future), "Random Forest", mae

forecast_rows=[]
recommendations=[]
for _, item in inventory.iterrows():
    pid=str(item["product_id"])
    group=sales[sales["product_id"].astype(str)==pid].copy()
    if group.empty: continue
    group["date"]=pd.to_datetime(group["date"])
    preds, model_name, mae=forecast_for_product(group,horizon)
    avg=float(np.mean(preds))
    lead=int(item["lead_time_days"]); safety=float(item["safety_stock"])
    rop=avg*lead+safety
    target=avg*(lead+cover_days)+safety
    stock=float(item["current_stock"])
    qty=max(0,int(np.ceil(target-stock)))
    status="REORDER" if stock<=rop else ("WATCH" if stock<=target else "OK")
    recommendations.append({"Product ID":pid,"Product":item["product_name"],"Category":item.get("category","General"),
        "Current stock":int(stock),"Forecast avg/day":round(avg,1),"Lead time":lead,"Safety stock":int(safety),
        "Reorder point":int(np.ceil(rop)),"Suggested order qty":qty,"Inventory value":round(stock*float(item["unit_price"]),2),
        "Status":status,"Model":model_name,"Holdout MAE":round(float(mae),2) if mae is not None and not np.isnan(mae) else "N/A"})
    last=group["date"].max()
    for i,p in enumerate(preds,1):
        forecast_rows.append({"product_id":pid,"product_name":item["product_name"],
            "forecast_date":(last+pd.Timedelta(days=i)).date().isoformat(),"predicted_units":round(float(p),2)})
rec=pd.DataFrame(recommendations)
fc=pd.DataFrame(forecast_rows)

# Main navigation
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Stock Operations", "Demand Forecast", "Reorder Planning"])

with tab1:
    st.subheader("Inventory overview")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Products",len(inventory))
    c2.metric("Total units in stock",int(inventory["current_stock"].sum()))
    c3.metric("Inventory value",f"₹{(inventory['current_stock']*inventory['unit_price']).sum():,.2f}")
    c4.metric("Items to reorder",int((rec["Status"]=="REORDER").sum()) if len(rec) else 0)
    st.markdown("#### Stock status")
    st.dataframe(rec[["Product ID","Product","Current stock","Reorder point","Suggested order qty","Status"]],use_container_width=True,hide_index=True)
    st.markdown("#### Recent stock transactions")
    if st.session_state.transactions.empty: st.info("No stock transactions recorded in this session yet.")
    else: st.dataframe(st.session_state.transactions.tail(10),use_container_width=True,hide_index=True)

with tab2:
    st.subheader("Record stock-in / stock-out")
    product_options={f"{r.product_id} — {r.product_name}":str(r.product_id) for r in inventory.itertuples()}
    if product_options:
        with st.form("transaction_form"):
            selected=st.selectbox("Product",list(product_options.keys()))
            txn_type=st.radio("Transaction type",["Stock In","Stock Out"],horizontal=True)
            quantity=st.number_input("Quantity",min_value=1,step=1,value=1)
            note=st.text_input("Note / reason",placeholder="e.g., supplier delivery or sale")
            submitted=st.form_submit_button("Save transaction")
        if submitted:
            pid=product_options[selected]
            idx=st.session_state.inventory.index[st.session_state.inventory["product_id"].astype(str)==pid][0]
            current=int(st.session_state.inventory.loc[idx,"current_stock"])
            if txn_type=="Stock Out" and quantity>current:
                st.error(f"Cannot issue {quantity} units; only {current} available.")
            else:
                new=current+int(quantity) if txn_type=="Stock In" else current-int(quantity)
                st.session_state.inventory.loc[idx,"current_stock"]=new
                row={"timestamp":pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),"product_id":pid,
                     "product_name":st.session_state.inventory.loc[idx,"product_name"],"type":txn_type,
                     "quantity":int(quantity),"note":note}
                st.session_state.transactions=pd.concat([st.session_state.transactions,pd.DataFrame([row])],ignore_index=True)
                st.success(f"{txn_type} saved. Updated stock: {new}")
                st.rerun()
    st.markdown("#### Edit inventory master")
    st.caption("Edit current stock, lead time, safety stock, price, or product details. Changes last for this running session.")
    edited=st.data_editor(st.session_state.inventory,use_container_width=True,num_rows="dynamic",key="inventory_editor")
    if st.button("Save inventory changes"):
        st.session_state.inventory=edited.copy()
        st.success("Inventory master updated.")
        st.rerun()
    st.download_button("Download transaction log",st.session_state.transactions.to_csv(index=False),"stock_transactions.csv","text/csv")

with tab3:
    st.subheader(f"Demand forecast — next {horizon} days")
    if len(fc):
        names=sorted(fc["product_name"].unique())
        chosen=st.selectbox("Choose product",names)
        chart=fc[fc["product_name"]==chosen].set_index("forecast_date")["predicted_units"]
        st.line_chart(chart)
        st.dataframe(fc[fc["product_name"]==chosen],use_container_width=True,hide_index=True)
        st.download_button("Download all forecasts",fc.to_csv(index=False),"demand_forecasts.csv","text/csv")
    st.caption("Forecasts are estimates, not guaranteed sales. Validate against real historical data before operational use.")

with tab4:
    st.subheader("Replenishment recommendations")
    st.write("Reorder point = forecast average daily demand × supplier lead time + safety stock.")
    st.write("Suggested order quantity = max(0, target stock − current stock).")
    st.dataframe(rec,use_container_width=True,hide_index=True)
    st.download_button("Download reorder plan",rec.to_csv(index=False),"reorder_plan.csv","text/csv")
    st.caption("REORDER = at/below reorder point; WATCH = below target stock; OK = above target stock.")

with st.expander("Methodology and limitations"):
    st.markdown("""
    **Forecasting:** Random Forest regression using weekday, time index, previous-day demand, demand seven days earlier, and previous seven-day average. A chronological 80/20 holdout is used to calculate MAE before the model is retrained on all available history.

    **Operations:** Current stock is tracked through stock-in and stock-out transactions. Reorder point uses average predicted daily demand, supplier lead time, and configured safety stock. The target level covers supplier lead time plus the selected additional cover period.

    **Limitations:** Included data is synthetic; session changes are temporary; no database, supplier integration, purchase-order dispatch, or user authentication is included. This is a learning prototype, not a production inventory system.
    """)

# Smart Inventory Management System

## Features
- Inventory overview and stock valuation
- Stock-in / stock-out transactions with quantity validation
- Editable inventory master
- Random Forest demand forecast with time-ordered holdout MAE
- Reorder point and suggested order quantity
- CSV downloads for transactions, forecasts and reorder plan
- No chatbot

## Run locally
1. Extract this folder.
2. Open a terminal in this folder.
3. Install: `pip install -r requirements.txt`
4. Run: `streamlit run app.py`
5. Open the local URL Streamlit prints, usually http://localhost:8501.

Included data is synthetic and for demonstration only. Changes made in the dashboard are session-only and reset when the app restarts.

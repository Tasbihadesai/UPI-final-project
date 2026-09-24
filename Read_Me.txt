# UPI Fraud Realtime Demo (local)

1. create venv and install:
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   # on Windows: venv\Scripts\activate
   pip install -r requirements.txt

2. (Optional) Put real data into data/ and run training:
   python src/train.py --data data/upi_transactions.csv --label is_fraud

   This produces models/lightgbm_model.pkl

3. Start the Streamlit dashboard:
   streamlit run src/upi_fraud_app.py

4. Use the dashboard:
   - Start Stream to see simulated transactions
   - Adjust threshold, label transactions for feedback

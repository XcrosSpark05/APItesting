from fastapi import FastAPI, HTTPException
import yfinance as yf
import uvicorn
from datetime import datetime

app = FastAPI()
cached_data = {}

# ✅ Dynamically fetch NIFTY 50 stock symbols
def get_nifty50_symbols():
    nifty_tickers = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "HINDUNILVR.NS", "SBIN.NS",
        "BHARTIARTL.NS", "HCLTECH.NS", "ASIANPAINT.NS", "KOTAKBANK.NS", "BAJFINANCE.NS", "ITC.NS",
        "LT.NS", "AXISBANK.NS", "WIPRO.NS", "DMART.NS", "MARUTI.NS", "ULTRACEMCO.NS", "SUNPHARMA.NS",
        "TITAN.NS", "TECHM.NS", "HDFCLIFE.NS", "JSWSTEEL.NS", "INDUSINDBK.NS", "M&M.NS", "BAJAJFINSV.NS",
        "NTPC.NS", "POWERGRID.NS", "NESTLEIND.NS", "ONGC.NS", "GRASIM.NS", "ADANIPORTS.NS", "CIPLA.NS",
        "SBILIFE.NS", "HINDALCO.NS", "DRREDDY.NS", "TATAMOTORS.NS", "BRITANNIA.NS", "BPCL.NS",
        "COALINDIA.NS", "EICHERMOT.NS", "DIVISLAB.NS", "IOC.NS", "HEROMOTOCO.NS", "UPL.NS",
        "APOLLOHOSP.NS", "TATASTEEL.NS", "BAJAJ-AUTO.NS", "ADANIENT.NS"
    ]
    return nifty_tickers

@app.get("/")
def home():
    return {"message": "Welcome to Invest Genius API"}

@app.get("/top-gainers")
def get_top_gainers():
    try:
        current_time = datetime.now()
        stocks_data = []
        nifty_stocks = get_nifty50_symbols()

        for symbol in nifty_stocks:
            print(f"Fetching data for: {symbol}")  # Debugging log
            stock = yf.Ticker(symbol)
            stock_data = stock.history(period="2d")  # Fetch last 2 days data

            if stock_data.empty:
                continue  # Skip stocks with no data

            try:
                prev_close = stock_data["Close"].iloc[-2]
                current_price = stock_data["Close"].iloc[-1]
                price_change = ((current_price - prev_close) / prev_close) * 100
            except IndexError:
                continue  # Skip if data is missing

            stock_info = {
                "symbol": symbol,
                "company_name": stock.info.get("shortName", "N/A"),
                "price": round(current_price, 2),
                "price_change": round(price_change, 2),
                "logo": stock.info.get("logo_url", "N/A"),
                "timestamp": current_time
            }
            stocks_data.append(stock_info)

        if not stocks_data:
            raise HTTPException(status_code=404, detail="No stock data available.")

        # ✅ Sort by % price change and get top 10 gainers
        top_gainers = sorted(stocks_data, key=lambda x: x["price_change"], reverse=True)[:10]

        return {"top_gainers": top_gainers}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/top-losers")
def get_top_losers():
    try:
        current_time = datetime.now()
        stocks_data = []
        nifty_stocks = get_nifty50_symbols()

        for symbol in nifty_stocks:
            stock = yf.Ticker(symbol)
            stock_data = stock.history(period="2d")

            if stock_data.empty:
                continue  # Skip stocks with no data

            try:
                prev_close = stock_data["Close"].iloc[-2]
                current_price = stock_data["Close"].iloc[-1]
                price_change = ((current_price - prev_close) / prev_close) * 100
            except IndexError:
                continue  # Skip if data is missing

            stock_info = {
                "symbol": symbol,
                "company_name": stock.info.get("shortName", "N/A"),
                "price": round(current_price, 2),
                "price_change": round(price_change, 2),
                "logo": stock.info.get("logo_url", "N/A"),
                "timestamp": current_time
            }
            stocks_data.append(stock_info)

        if not stocks_data:
            raise HTTPException(status_code=404, detail="No stock data available.")

        # ✅ Sort by % price change and get top 10 losers
        top_losers = sorted(stocks_data, key=lambda x: x["price_change"])[:10]

        return {"top_losers": top_losers}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# NEW ENDPOINT: Fetch all stocks
@app.get("/all-stocks")
def get_all_stocks():
    try:
        current_time = datetime.now()
        stocks_data = []
        nifty_stocks = get_nifty50_symbols()

        for symbol in nifty_stocks:
            print(f"Fetching data for: {symbol}")  # Debugging log
            stock = yf.Ticker(symbol)
            stock_data = stock.history(period="2d")

            if stock_data.empty:
                continue

            try:
                prev_close = stock_data["Close"].iloc[-2]
                current_price = stock_data["Close"].iloc[-1]
                price_change = ((current_price - prev_close) / prev_close) * 100
            except IndexError:
                continue

            stock_info = {
                "symbol": symbol,
                "company_name": stock.info.get("shortName", "N/A"),
                "price": round(current_price, 2),
                "price_change": round(price_change, 2),
                "logo": stock.info.get("logo_url", "N/A"),
                "timestamp": current_time
            }
            stocks_data.append(stock_info)

        if not stocks_data:
            raise HTTPException(status_code=404, detail="No stock data available.")

        return {"all_stocks": stocks_data}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

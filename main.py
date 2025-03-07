from fastapi import FastAPI, HTTPException
import yfinance as yf
import uvicorn
import time
from datetime import datetime, timedelta

app = FastAPI()
cached_data = {}

@app.get("/")
def home():
    return {"message": "Welcome to Invest Genius API"}

@app.get("/stock/{symbol}")
def get_stock_details(symbol: str):
    try:
        current_time = datetime.now()
        
        if symbol in cached_data and cached_data[symbol]['timestamp'] > current_time - timedelta(minutes=1):
            cached_info = cached_data[symbol]
            return {
                "symbol": symbol,
                "company_name": cached_info['company_name'],
                "price": cached_info['price'],
                "logo": cached_info['logo']
            }
        
        time.sleep(1)  # Prevent rate limiting
        stock = yf.Ticker(symbol)
        stock_data = stock.history(period="1d")
        stock_info = stock.info
        
        if stock_data.empty or not stock_info:
            raise ValueError("No data available for this stock symbol.")
        
        price = stock_data["Close"].iloc[-1]
        company_name = stock_info.get("shortName", "N/A")
        logo = stock_info.get("logo_url", "N/A")
        
        cached_data[symbol] = {
            'timestamp': current_time,
            'price': price,
            'company_name': company_name,
            'logo': logo
        }
        
        return {
            "symbol": symbol,
            "company_name": company_name,
            "price": price,
            "logo": logo
        }
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

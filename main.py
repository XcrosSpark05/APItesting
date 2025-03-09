from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yfinance as yf
import uvicorn
from datetime import datetime
import requests
import openai
import os

app = FastAPI()

# Function to fetch NIFTY 50 symbols (for stocks endpoints)
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
                "timestamp": current_time.isoformat()
            }
            stocks_data.append(stock_info)
        if not stocks_data:
            raise HTTPException(status_code=404, detail="No stock data available.")
        top_gainers = sorted(stocks_data, key=lambda x: x["price_change"], reverse=True)[:10]
        return {"top_gainers": top_gainers}
    except Exception as e:
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
                "timestamp": current_time.isoformat()
            }
            stocks_data.append(stock_info)
        if not stocks_data:
            raise HTTPException(status_code=404, detail="No stock data available.")
        top_losers = sorted(stocks_data, key=lambda x: x["price_change"])[:10]
        return {"top_losers": top_losers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/all-stocks")
def get_all_stocks():
    try:
        current_time = datetime.now()
        stocks_data = []
        nifty_stocks = get_nifty50_symbols()
        for symbol in nifty_stocks:
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
                "timestamp": current_time.isoformat()
            }
            stocks_data.append(stock_info)
        if not stocks_data:
            raise HTTPException(status_code=404, detail="No stock data available.")
        return {"all_stocks": stocks_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/indian-indices")
def get_indian_indices():
    try:
        current_time = datetime.now()
        index_data = {}
        indices_info = {
            "NIFTY 50": "^NSEI",
            "SENSEX": "^BSESN",
            "NIFTY BANK": "^NSEBANK",
            "NIFTY IT": "^CNXIT",
            "NIFTY MIDCAP 50": "^NSEMDCP50",
            "NIFTY 100": "^CNX100",
            "NIFTY FINANCIAL SERVICES (FINNIFTY)": "^CNXFIN",
            "INDIA VIX": "^INDIAVIX"
        }
        for name, symbol in indices_info.items():
            try:
                data = yf.Ticker(symbol).history(period="1d")
                index_data[name] = {
                    "Current Value": round(data["Close"].iloc[-1], 2)
                }
            except Exception as e:
                index_data[name] = f"Error: {e}"
        if not index_data:
            raise HTTPException(status_code=404, detail="No index data available.")
        return {"indian_indices": index_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# NEW ENDPOINT: Stock Market News using NewsAPI
@app.get("/stock-news")
def get_stock_news():
    try:
        # Replace with your NewsAPI.org API key
        NEWS_API_KEY = "YOUR_NEWS_API_KEY"
        NEWS_API_URL = "https://newsapi.org/v2/everything"
        params = {
            "q": "stock market OR finance OR investment OR trading",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 20,
            "apiKey": NEWS_API_KEY
        }
        response = requests.get(NEWS_API_URL, params=params)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Error fetching news")
        data = response.json()
        articles = data.get("articles", [])
        return {"stock_news": articles}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# NEW ENDPOINT: Chatbot Endpoint for Stock Market Questions
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chatbot", response_model=ChatResponse)
async def chatbot_endpoint(chat: ChatRequest):
    try:
        # Set your OpenAI API key (or use an environment variable)
        openai.api_key = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")
        prompt = (
            f"Answer the following stock market related question: {chat.question}\n\n"
            "Provide clear and concise analysis, and if applicable, suggest whether to buy, sell, or hold. "
            "Disclaimer: This is not financial advice."
        )
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            temperature=0.7
        )
        answer = response.choices[0].text.strip()
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

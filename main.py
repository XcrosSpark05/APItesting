from fastapi import FastAPI, HTTPException
import yfinance as yf
import uvicorn
import time
from datetime import datetime, timedelta

app = FastAPI()

# Cache to store stock data and timestamps for rate limiting
cached_data = {}

@app.get("/")
def home():
    return {"message": "Welcome to Invest Genius API"}

@app.get("/stock/{symbol}")
def get_stock_price(symbol: str):
    try:
        # Implement caching and rate limiting
        current_time = datetime.now()
        
        if symbol in cached_data and cached_data[symbol]['timestamp'] > current_time - timedelta(minutes=1):
            # Return cached data if it's less than 1 minute old
            return {"symbol": symbol, "price": cached_data[symbol]['data']["Close"].iloc[-1]}
        
        # Throttle the request to prevent rate limiting
        time.sleep(1)  # Sleep for 1 second to avoid hitting rate limits
        
        # Try to get the stock data
        stock = yf.Ticker(symbol)
        stock_data = stock.history(period="1d")
        
        if stock_data.empty:
            raise ValueError("No data available for this stock symbol.")
        
        # Cache the data
        cached_data[symbol] = {
            'timestamp': current_time,
            'data': stock_data
        }
        
        price = stock_data["Close"].iloc[-1]
        return {"symbol": symbol, "price": price}
    
    except ValueError as e:
        # Specific error message for missing data
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        # General error handling for network issues or invalid symbols
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/stocks")
def get_multiple_stock_prices(symbols: str):
    symbols_list = symbols.split(",")  # Split the comma-separated symbols into a list
    result = []
    
    for symbol in symbols_list:
        try:
            # Implement caching and rate limiting for each symbol
            current_time = datetime.now()
            
            if symbol in cached_data and cached_data[symbol]['timestamp'] > current_time - timedelta(minutes=1):
                # Return cached data if it's less than 1 minute old
                result.append({symbol: cached_data[symbol]['data']["Close"].iloc[-1]})
                continue
            
            # Throttle the request to prevent rate limiting
            time.sleep(1)  # Sleep for 1 second to avoid hitting rate limits
            
            stock = yf.Ticker(symbol)
            stock_data = stock.history(period="1d")
            
            if stock_data.empty:
                result.append({symbol: "No data available"})
            else:
                # Cache the data
                cached_data[symbol] = {
                    'timestamp': current_time,
                    'data': stock_data
                }
                
                result.append({symbol: stock_data["Close"].iloc[-1]})
        
        except Exception as e:
            result.append({symbol: f"Error: {str(e)}"})
    
    return {"stocks": result}

@app.get("/stock/{symbol}/history")
def get_stock_history(symbol: str, period: str = "1mo"):
    try:
        # Implement caching and rate limiting
        current_time = datetime.now()
        
        if symbol in cached_data and cached_data[symbol]['timestamp'] > current_time - timedelta(minutes=1):
            # Return cached data if it's less than 1 minute old
            return {"symbol": symbol, "history": cached_data[symbol]['history']}
        
        # Throttle the request to prevent rate limiting
        time.sleep(1)  # Sleep for 1 second to avoid hitting rate limits
        
        stock = yf.Ticker(symbol)
        stock_data = stock.history(period=period)
        
        if stock_data.empty:
            raise ValueError("No data available for this stock symbol.")
        
        # Prepare the historical data
        history_data = [{"date": str(date.date()), "price": price} 
                        for date, price in zip(stock_data.index, stock_data["Close"])]
        
        # Cache the historical data
        cached_data[symbol] = {
            'timestamp': current_time,
            'history': history_data,
            'data': stock_data
        }
        
        return {"symbol": symbol, "history": history_data}
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
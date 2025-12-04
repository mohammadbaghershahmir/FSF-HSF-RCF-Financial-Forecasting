from summarize_module.summarizer import Summarizer
import os, json
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta


class DataLoader:
    def __init__(self, args):
        self.price_dir = args.price_dir
        self.tweet_dir = args.tweet_dir
        self.seq_len = args.seq_len
        self.summarizer = Summarizer()
        self.main_market = "BTC"  # بازار اصلی
        self.secondary_markets = ["GOLD"]  # بازارهای ثانویه
        self.markets = [self.main_market] + self.secondary_markets  # همه بازارها


    def daterange(self, start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)


    def get_sentiment(self, date_str, market):
        """تحلیل احساسات برای یک بازار خاص"""
        tweet_path = os.path.join(self.tweet_dir, market, date_str)
        if not os.path.exists(tweet_path):
            return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

        with open(tweet_path, "r", encoding="utf-8") as f:
            tweets = f.readlines()

        total_sentiment = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        count = 0
        for tweet in tweets:
            try:
                tweet_data = json.loads(tweet)
                sentiment = tweet_data["score"]
                for key in total_sentiment:
                    total_sentiment[key] += sentiment[key]
                count += 1
            except:
                continue

        if count > 0:
            for key in total_sentiment:
                total_sentiment[key] /= count

        return total_sentiment

    def get_market_data(self, date_str, market):
        """خواندن داده‌های قیمت برای یک بازار خاص - ساختار جدید"""
        price_path = os.path.join(self.price_dir, f"{market}.txt")
        
        try:
            with open(price_path, 'r') as f:
                for line in f:
                    # Skip header line
                    if line.strip().startswith('date'):
                        continue
                    
                    # Split by multiple spaces
                    line_data = line.strip().split()
                    if len(line_data) >= 7 and line_data[0] == date_str:
                        # Structure: date, close_norm, open_norm, high_norm, low_norm, price_change, volume
                        return {
                            "close": float(line_data[1]),      # close price (normalized)
                            "open": float(line_data[2]),       # open price (normalized)
                            "high": float(line_data[3]),       # high price (normalized)
                            "low": float(line_data[4]),        # low price (normalized)
                            "price_change": float(line_data[5]), # close price change
                            "volume": float(line_data[6])      # volume
                        }
        except Exception as e:
            print(f"Error reading {market} data for {date_str}: {e}")
        
        return None

    def load(self, flag="train"):
        """لود داده‌های همه بازارها با تمرکز بر بازار اصلی و تبدیل به DataFrame"""
        data = []
        
        # خواندن داده‌های قیمت بازار اصلی برای تعیین محدوده تاریخ
        main_market_price_path = os.path.join(self.price_dir, f"{self.main_market}.txt")
        dates = []
        
        # خواندن تاریخ‌ها به صورت خط به خط
        with open(main_market_price_path, 'r') as f:
            for line in f:
                # Skip header line
                if line.strip().startswith('date'):
                    continue
                
                line_data = line.strip().split()
                if len(line_data) >= 7:
                    dates.append(line_data[0])
        
        print(f"Total available dates for {self.main_market}: {len(dates)}")
        
        # انتخاب رندوم 50% از داده‌ها
        random.seed(42)  # برای reproducible results
        selected_dates = random.sample(dates, min(len(dates), len(dates)//1))
        print(f"Selected 50% of data randomly: {len(selected_dates)} dates")
        
        for date_str in selected_dates:
            # ابتدا داده‌های بازار اصلی را جمع‌آوری می‌کنیم
            main_market_data = self.get_market_data(date_str, self.main_market)
            main_market_sentiment = self.get_sentiment(date_str, self.main_market)
            
            if main_market_data is None:
                continue
                
            # جمع‌آوری داده‌های بازارهای ثانویه
            secondary_data = {}
            use_secondary_data = True
            
            for market in self.secondary_markets:
                market_data = self.get_market_data(date_str, market)
                sentiment_data = self.get_sentiment(date_str, market)
                
                if market_data is None:
                    # اگر داده‌های ثانویه موجود نباشد، فقط از BTC استفاده می‌کنیم
                    print(f"Warning: Missing {market} data for {date_str}, using BTC only")
                    use_secondary_data = False
                    break
                    
                secondary_data[market] = {
                    "data": market_data,
                    "sentiment": sentiment_data
                }

            # ساخت دیکشنری market_data با ساختار مورد نیاز PredictAgent
            market_data_dict = {}
            
            # اضافه کردن داده‌های بازار اصلی
            market_data_dict[f"{self.main_market.lower()}_data"] = main_market_data
            market_data_dict[f"{self.main_market.lower()}_sentiment"] = main_market_sentiment
            
            # اضافه کردن داده‌های بازارهای ثانویه (فقط اگر موجود باشند)
            if use_secondary_data:
                for market, data_dict in secondary_data.items():
                    market_prefix = market.lower()
                    market_data_dict[f"{market_prefix}_data"] = data_dict["data"]
                    market_data_dict[f"{market_prefix}_sentiment"] = data_dict["sentiment"]
            else:
                # اگر داده‌های ثانویه موجود نباشند، از داده‌های پیش‌فرض استفاده می‌کنیم
                for market in self.secondary_markets:
                    market_prefix = market.lower()
                    market_data_dict[f"{market_prefix}_data"] = {
                        "close": 0.0,
                        "open": 0.0,
                        "high": 0.0,
                        "low": 0.0,
                        "price_change": 0.0,
                        "volume": 0.0
                    }
                    market_data_dict[f"{market_prefix}_sentiment"] = {
                        "positive": 0.33,
                        "negative": 0.33,
                        "neutral": 0.34
                    }

            # ساخت نمونه با ساختار مورد نیاز
            # استفاده از price_change به عنوان target
            target_price_change = main_market_data["price_change"]
            
            sample = {
                "date": date_str,
                "ticker": self.main_market,
                "market_data": market_data_dict,
                "price_change": target_price_change,  # close price change
                "target": "Positive" if target_price_change > 0 else "Negative"
            }
            
            data.append(sample)

        # تبدیل لیست به DataFrame
        df = pd.DataFrame(data)
        
        # Debug information
        print(f"Total samples collected: {len(data)}")
        if len(data) > 0:
            print(f"Sample data structure: {data[0]}")
        
        if len(df) == 0:
            print("WARNING: DataFrame is empty! No data loaded.")
            print(f"Sample list: {data[:3]}")
        else:
            print(f"DataFrame columns: {list(df.columns)}")
            print(f"DataFrame shape: {df.shape}")
            print(f"Sample data:")
            print(df.head())
        
        # تقسیم رندوم داده‌ها به train/val/test
        total_len = len(df)
        if total_len == 0:
            return df
            
        # Shuffle the data randomly
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        if flag == "train":
            return df.iloc[:int(0.7 * total_len)]
        elif flag == "val":
            return df.iloc[int(0.7 * total_len):int(0.9 * total_len)]
        else:  # test
            return df.iloc[int(0.9 * total_len):]

    def get_tweets_texts(self, date_str, market):
        """Get tweet texts for a specific market and date"""
        tweet_path = os.path.join(self.tweet_dir, market, date_str)
        if not os.path.exists(tweet_path):
            return []

        texts = []
        try:
            with open(tweet_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        tweet_data = json.loads(line)
                        if 'text' in tweet_data:
                            texts.append(tweet_data['text'])
                    except:
                        continue
        except Exception as e:
            print(f"Error reading tweets for {market} on {date_str}: {e}")
            return []
        
        return texts

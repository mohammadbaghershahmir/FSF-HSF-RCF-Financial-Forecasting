from summarize_module.summarizer import Summarizer
import os, json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class DataLoader:
    NEUTRAL_EPS = 0.001
    def __init__(self, args):
        self.price_dir = args.price_dir
        self.tweet_dir = args.tweet_dir
        self.seq_len = args.seq_len
        self.summarizer = Summarizer()
        self.main_market = "BTC"  # بازار اصلی
        self.secondary_markets = ["GOLD"]  # بازارهای ثانویه
        self.markets = [self.main_market] + self.secondary_markets  # همه بازارها
        self.data_ratio = getattr(args, 'data_ratio', 1)  # Default to 10% of data


    def daterange(self, start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)


    def get_sentiment(self, date_str, market):
        """تحلیل احساسات برای یک بازار خاص"""
        tweet_path = os.path.join(self.tweet_dir, market, date_str)
        if not os.path.isfile(tweet_path):
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
        """دریافت داده‌های قیمت برای یک بازار خاص با مدیریت داده‌های ناقص"""
        try:
            price_path = os.path.join(self.price_dir, f"{market}.txt")
            # خواندن فایل به صورت خط به خط
            with open(price_path, 'r') as f:
                for line in f:
                    data = line.strip().split('\t')
                    if data[0] == date_str:
                        # اطمینان از وجود حداقل 6 ستون
                        if len(data) >= 6:
                            market_data = {
                                # "price_change": float(data[1]),  # Removed to prevent data leakage
                                "high": float(data[2]),
                                "low": float(data[3]),
                                "open": float(data[4]),
                                "close": float(data[5]),
                                "volume": float(data[6])
                            }
                            
                            # Add technical features
                            technical_features = self.calculate_technical_features(market_data)
                            market_data.update(technical_features)
                            
                            return market_data
            return None
        except Exception as e:
            print(f"Error loading market data for {market} on {date_str}: {str(e)}")
            return None

    def get_price_change(self, date_str, market):
        """دریافت تغییر قیمت فقط برای ارزیابی (نه برای مدل)"""
        try:
            price_path = os.path.join(self.price_dir, f"{market}.txt")
            with open(price_path, 'r') as f:
                for line in f:
                    data = line.strip().split('\t')
                    if data[0] == date_str and len(data) >= 2:
                        return float(data[1])
            return None
        except Exception as e:
            print(f"Error loading price change for {market} on {date_str}: {str(e)}")
            return None

    def calculate_technical_features(self, market_data):
        """محاسبه ویژگی‌های تکنیکال از داده‌های موجود"""
        features = {}
        
        if market_data:
            high = market_data.get('high', 0)
            low = market_data.get('low', 0)
            open_price = market_data.get('open', 0)
            close = market_data.get('close', 0)
            volume = market_data.get('volume', 0)
            
            # Calculate technical indicators
            if open_price > 0:
                features['price_range'] = (high - low) / open_price
                features['close_open_ratio'] = close / open_price
                features['high_open_ratio'] = high / open_price
                features['low_open_ratio'] = low / open_price
            
            if volume > 0:
                features['volume_log'] = np.log(volume)
            else:
                features['volume_log'] = 0
                
        return features


    def load(self, flag="train"):
        """لود داده‌های همه بازارها با تمرکز بر بازار اصلی و تبدیل به DataFrame"""
        data = []
        
        # خواندن داده‌های قیمت بازار اصلی برای تعیین محدوده تاریخ
        main_market_price_path = os.path.join(self.price_dir, f"{self.main_market}.txt")
        dates = []
        
        # خواندن تاریخ‌ها به صورت خط به خط
        with open(main_market_price_path, 'r') as f:
            for line in f:
                dates.append(line.strip().split('\t')[0])
        
        # Sample only data_ratio of the dates
        total_dates = len(dates)
        sample_size = int(total_dates * self.data_ratio)
        
        # Use stratified sampling to maintain temporal distribution
        if sample_size < total_dates:
            # Sort dates to maintain temporal order
            dates.sort()
            # Take evenly spaced samples
            step = total_dates // sample_size
            sampled_dates = dates[::step][:sample_size]
            print(f"Sampling {len(sampled_dates)} dates from {total_dates} total dates ({self.data_ratio*100:.1f}%)")
        else:
            sampled_dates = dates
            print(f"Using all {len(dates)} dates")
        
        for date_str in sampled_dates:
            # ابتدا داده‌های بازار اصلی را جمع‌آوری می‌کنیم
            main_market_data = self.get_market_data(date_str, self.main_market)
            if main_market_data is None:
                continue  
            main_market_sentiment = self.get_sentiment(date_str, self.main_market)
            
            # Get price change separately for evaluation only
            price_change = self.get_price_change(date_str, self.main_market)
            if price_change is None:
                continue
                
            #print(f"[DEBUG] {date_str}  ->  {main_market_data}")
            if price_change > DataLoader.NEUTRAL_EPS:
                target_label = "Positive"
            elif price_change < -DataLoader.NEUTRAL_EPS:
                target_label = "Negative"
            else:
                target_label = "Neutral"
                
            # جمع‌آوری داده‌های بازارهای ثانویه
            secondary_data = {}
            all_data_available = True
            
            for market in self.secondary_markets:
                market_data = self.get_market_data(date_str, market)
                sentiment_data = self.get_sentiment(date_str, market)
                
                if market_data is None:
                    all_data_available = False
                    break
                    
                secondary_data[market] = {
                    "data": market_data,
                    "sentiment": sentiment_data
                }
            
            if not all_data_available:
                continue

            # ساخت دیکشنری market_data با ساختار مورد نیاز PredictAgent
            market_data_dict = {}
            
            # اضافه کردن داده‌های بازار اصلی
            market_data_dict[f"{self.main_market.lower()}_data"] = main_market_data
            market_data_dict[f"{self.main_market.lower()}_sentiment"] = main_market_sentiment
            
            # اضافه کردن داده‌های بازارهای ثانویه
            for market, data_dict in secondary_data.items():
                market_prefix = market.lower()
                market_data_dict[f"{market_prefix}_data"] = data_dict["data"]
                market_data_dict[f"{market_prefix}_sentiment"] = data_dict["sentiment"]

            # ساخت نمونه با ساختار مورد نیاز
            sample = {
                "date": date_str,
                "ticker": self.main_market,
                "market_data": market_data_dict,
                "price_change": price_change,  # Keep for evaluation only
                "target":target_label
            }
            
            data.append(sample)

        # تبدیل لیست به DataFrame
        df = pd.DataFrame(data)
        
        print(f"Loaded {len(df)} samples for {flag} set")
        
        # تقسیم داده‌ها به train/val/test
        total_len = len(df)
        if flag == "train":
            return df.iloc[:int(0.9* total_len)]
        elif flag == "val":
            return df.iloc[int(0.7 * total_len):int(0.9 * total_len)]
        else:
            return df.iloc[int(0.9 * total_len):]

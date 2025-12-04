from typing import List, Union, Literal, Dict, Tuple
from utils.llm import FastChatLLM
from utils.prompts import (
    CANDLE_SENTIMENT_INSTRUCTION, 
    NEWS_SUMMARY_INSTRUCTION, 
    MODEL_COMPARISON_INSTRUCTION
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_loader import VicunaModel
import torch

class DualModelManager:
    """Manages two Vicuna models for sequential loading to avoid GPU memory issues"""
    
    def __init__(self):
        self.candle_model = None
        self.news_model = None
        self.current_model = None
        
    def load_candle_model(self):
        """Load the candlestick + sentiment model"""
        if self.candle_model is None:
            print("Loading Candle + Sentiment Model...")
            model_data = VicunaModel.get_model(
                model_name="lmsys/vicuna-7b-v1.5",
                quantization=True,
                device_map="auto"
            )
            self.candle_model = model_data
            print("Candle + Sentiment Model Loaded.")
        return self.candle_model
    
    def load_news_model(self):
        """Load the news summary model"""
        if self.news_model is None:
            print("Loading News Summary Model...")
            model_data = VicunaModel.get_model(
                model_name="lmsys/vicuna-7b-v1.5",
                quantization=True,
                device_map="auto"
            )
            self.news_model = model_data
            print("News Summary Model Loaded.")
        return self.news_model
    
    def unload_current_model(self):
        """Unload current model to free GPU memory"""
        if self.current_model is not None:
            # Clear model from GPU memory
            if hasattr(self.current_model, 'model'):
                del self.current_model['model']
            if hasattr(self.current_model, 'tokenizer'):
                del self.current_model['tokenizer']
            torch.cuda.empty_cache()
            self.current_model = None
            print("Current model unloaded from GPU memory.")

class CandleSentimentAgent:
    """Agent for analyzing candlestick data and sentiment"""
    
    def __init__(self, model_manager: DualModelManager):
        self.model_manager = model_manager
        self.llm = None
        
    def predict(self, market_data: Dict, sentiment_data: Dict) -> Tuple[str, str]:
        """Predict Bitcoin price movement based on candlestick and sentiment data"""
        # Load model
        model_data = self.model_manager.load_candle_model()
        self.model_manager.current_model = model_data
        self.llm = FastChatLLM(model=model_data["model"], tokenizer=model_data["tokenizer"])
        
        # Format data
        formatted_market_data = self._format_market_data(market_data)
        formatted_sentiment_data = self._format_sentiment_data(sentiment_data)
        
        # Generate prompt
        prompt = CANDLE_SENTIMENT_INSTRUCTION.format(
            market_data=formatted_market_data,
            sentiment_data=formatted_sentiment_data
        )
        
        print("\n" + "="*80)
        print("🕯️  CANDLE + SENTIMENT MODEL")
        print("="*80)
        print("📊 INPUT DATA:")
        print("-" * 40)
        print(formatted_market_data)
        print("\n📈 SENTIMENT DATA:")
        print("-" * 40)
        print(formatted_sentiment_data)
        print("\n🤖 PROMPT:")
        print("-" * 40)
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        
        # Get prediction
        print("\n🔄 GENERATING PREDICTION...")
        response = self.llm(prompt)
        
        # Parse response
        prediction, explanation = self._parse_response(response)
        
        print("\n📋 MODEL OUTPUT:")
        print("-" * 40)
        print(f"Prediction: {prediction}")
        print(f"Full Response Length: {len(response)} characters")
        print(f"Explanation: {explanation}")
        print("="*80)
        
        return prediction, explanation
    
    def _format_market_data(self, market_data: Dict) -> str:
        """Format market data for the prompt"""
        formatted = "=== Bitcoin Market Data ===\n"
        formatted += f"High: {market_data.get('btc_data', {}).get('high', 'N/A')}\n"
        formatted += f"Low: {market_data.get('btc_data', {}).get('low', 'N/A')}\n"
        formatted += f"Volume: {market_data.get('btc_data', {}).get('volume', 'N/A')}\n\n"
        
        formatted += "=== Gold Market Data ===\n"
        formatted += f"High: {market_data.get('gold_data', {}).get('high', 'N/A')}\n"
        formatted += f"Low: {market_data.get('gold_data', {}).get('low', 'N/A')}\n"
        formatted += f"Volume: {market_data.get('gold_data', {}).get('volume', 'N/A')}\n"
        
        return formatted
    
    def _format_sentiment_data(self, sentiment_data: Dict) -> str:
        """Format sentiment data for the prompt"""
        formatted = "=== Bitcoin Sentiment ===\n"
        btc_sentiment = sentiment_data.get('btc_sentiment', {})
        formatted += f"Positive: {btc_sentiment.get('positive', 0):.1f}%\n"
        formatted += f"Negative: {btc_sentiment.get('negative', 0):.1f}%\n"
        formatted += f"Neutral: {btc_sentiment.get('neutral', 0):.1f}%\n\n"
        
        formatted += "=== Gold Sentiment ===\n"
        gold_sentiment = sentiment_data.get('gold_sentiment', {})
        formatted += f"Positive: {gold_sentiment.get('positive', 0):.1f}%\n"
        formatted += f"Negative: {gold_sentiment.get('negative', 0):.1f}%\n"
        formatted += f"Neutral: {gold_sentiment.get('neutral', 0):.1f}%\n"
        
        return formatted
    
    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Parse the model response to extract prediction and explanation"""
        lines = response.split('\n')
        prediction = "Unknown"
        explanation = response
        
        # Look for prediction in the response with multiple patterns
        for line in lines:
            if "Bitcoin Price Movement:" in line:
                if "Positive" in line:
                    prediction = "Positive"
                elif "Negative" in line:
                    prediction = "Negative"
                break
        
        # If no prediction found, try alternative patterns
        if prediction == "Unknown":
            for line in lines:
                if "Price Movement:" in line:
                    if "Positive" in line:
                        prediction = "Positive"
                    elif "Negative" in line:
                        prediction = "Negative"
                    break
        
        # If still no prediction, check the entire response
        if prediction == "Unknown":
            if "positive" in response.lower():
                prediction = "Positive"
            elif "negative" in response.lower():
                prediction = "Negative"
        
        return prediction, explanation

class NewsSummaryAgent:
    """Agent for analyzing news summaries and sentiment"""
    
    def __init__(self, model_manager: DualModelManager):
        self.model_manager = model_manager
        self.llm = None
        
    def predict(self, news_summary: str, sentiment_data: Dict) -> Tuple[str, str]:
        """Predict Bitcoin price movement based on news summary and sentiment"""
        # Load model
        model_data = self.model_manager.load_news_model()
        self.model_manager.current_model = model_data
        self.llm = FastChatLLM(model=model_data["model"], tokenizer=model_data["tokenizer"])
        
        # Format sentiment data
        formatted_sentiment_data = self._format_sentiment_data(sentiment_data)
        
        # Generate prompt
        prompt = NEWS_SUMMARY_INSTRUCTION.format(
            news_summary=news_summary,
            sentiment_data=formatted_sentiment_data
        )
        
        print("\n" + "="*80)
        print("📰 NEWS SUMMARY MODEL")
        print("="*80)
        print("📰 NEWS SUMMARY:")
        print("-" * 40)
        print(news_summary)
        print("\n📈 SENTIMENT DATA:")
        print("-" * 40)
        print(formatted_sentiment_data)
        print("\n🤖 PROMPT:")
        print("-" * 40)
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        
        # Get prediction
        print("\n🔄 GENERATING PREDICTION...")
        response = self.llm(prompt)
        
        # Parse response
        prediction, explanation = self._parse_response(response)
        
        print("\n📋 MODEL OUTPUT:")
        print("-" * 40)
        print(f"Prediction: {prediction}")
        print(f"Full Response Length: {len(response)} characters")
        print(f"Explanation: {explanation}")
        print("="*80)
        
        return prediction, explanation
    
    def _format_sentiment_data(self, sentiment_data: Dict) -> str:
        """Format sentiment data for the prompt"""
        formatted = "=== Bitcoin Sentiment ===\n"
        btc_sentiment = sentiment_data.get('btc_sentiment', {})
        formatted += f"Positive: {btc_sentiment.get('positive', 0):.1f}%\n"
        formatted += f"Negative: {btc_sentiment.get('negative', 0):.1f}%\n"
        formatted += f"Neutral: {btc_sentiment.get('neutral', 0):.1f}%\n\n"
        
        formatted += "=== Gold Sentiment ===\n"
        gold_sentiment = sentiment_data.get('gold_sentiment', {})
        formatted += f"Positive: {gold_sentiment.get('positive', 0):.1f}%\n"
        formatted += f"Negative: {gold_sentiment.get('negative', 0):.1f}%\n"
        formatted += f"Neutral: {gold_sentiment.get('neutral', 0):.1f}%\n"
        
        return formatted
    
    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Parse the model response to extract prediction and explanation"""
        lines = response.split('\n')
        prediction = "Unknown"
        explanation = response
        
        # Look for prediction in the response with multiple patterns
        for line in lines:
            if "Bitcoin Price Movement:" in line:
                if "Positive" in line:
                    prediction = "Positive"
                elif "Negative" in line:
                    prediction = "Negative"
                break
        
        # If no prediction found, try alternative patterns
        if prediction == "Unknown":
            for line in lines:
                if "Price Movement:" in line:
                    if "Positive" in line:
                        prediction = "Positive"
                    elif "Negative" in line:
                        prediction = "Negative"
                    break
        
        # If still no prediction, check the entire response
        if prediction == "Unknown":
            if "positive" in response.lower():
                prediction = "Positive"
            elif "negative" in response.lower():
                prediction = "Negative"
        
        return prediction, explanation

class DualPredictionAgent:
    """Main agent that coordinates both models and makes final decisions"""
    
    def __init__(self):
        self.model_manager = DualModelManager()
        self.candle_agent = CandleSentimentAgent(self.model_manager)
        self.news_agent = NewsSummaryAgent(self.model_manager)
        
    def predict(self, market_data: Dict, sentiment_data: Dict, news_summary: str, target: str) -> Dict:
        """Make prediction using both models and decide on final output"""
        
        # Get prediction from candlestick + sentiment model
        print("Getting prediction from Candle + Sentiment Model...")
        candle_prediction, candle_explanation = self.candle_agent.predict(market_data, sentiment_data)
        
        # Unload first model to free GPU memory
        self.model_manager.unload_current_model()
        
        # Get prediction from news summary model
        print("Getting prediction from News Summary Model...")
        news_prediction, news_explanation = self.news_agent.predict(news_summary, sentiment_data)
        
        # Unload second model to free GPU memory
        self.model_manager.unload_current_model()
        
        # Determine if models agree
        models_agree = candle_prediction == news_prediction
        
        # Check if either prediction is correct
        candle_correct = candle_prediction.lower() == target.lower()
        news_correct = news_prediction.lower() == target.lower()
        
        print("\n" + "="*80)
        print("🎯 FINAL DECISION ENGINE")
        print("="*80)
        print(f"🎯 TARGET: {target}")
        print(f"🕯️  Candle Model Prediction: {candle_prediction} (Correct: {candle_correct})")
        print(f"📰 News Model Prediction: {news_prediction} (Correct: {news_correct})")
        print(f"🤝 Models Agree: {models_agree}")
        
        result = {
            "candle_prediction": candle_prediction,
            "candle_explanation": candle_explanation,
            "news_prediction": news_prediction,
            "news_explanation": news_explanation,
            "models_agree": models_agree,
            "candle_correct": candle_correct,
            "news_correct": news_correct,
            "target": target,
            "final_prediction": None,
            "final_explanation": None,
            "dataset_type": None
        }
        
        if models_agree:
            # Models agree - merge explanations for supervised learning
            result["final_prediction"] = candle_prediction
            result["final_explanation"] = self._merge_explanations(candle_explanation, news_explanation)
            result["dataset_type"] = "supervised"
            print(f"✅ MODELS AGREE → Supervised Learning Dataset")
            print(f"📝 Final Prediction: {result['final_prediction']}")
        else:
            # Models disagree - determine which is correct
            if candle_correct and not news_correct:
                result["final_prediction"] = candle_prediction
                result["final_explanation"] = candle_explanation
                result["dataset_type"] = "dpo"
                result["chosen"] = candle_explanation
                result["rejected"] = news_explanation
                print(f"🔄 MODELS DISAGREE → DPO Dataset (Candle Model Correct)")
                print(f"📝 Final Prediction: {result['final_prediction']}")
            elif news_correct and not candle_correct:
                result["final_prediction"] = news_prediction
                result["final_explanation"] = news_explanation
                result["dataset_type"] = "dpo"
                result["chosen"] = news_explanation
                result["rejected"] = candle_explanation
                print(f"🔄 MODELS DISAGREE → DPO Dataset (News Model Correct)")
                print(f"📝 Final Prediction: {result['final_prediction']}")
            else:
                # Both wrong - use for DPO with reflection
                result["final_prediction"] = "Unknown"
                result["final_explanation"] = "Both models failed to predict correctly"
                result["dataset_type"] = "dpo"
                result["chosen"] = "Both models need improvement"
                result["rejected"] = f"Candle: {candle_explanation}, News: {news_explanation}"
                print(f"❌ BOTH MODELS WRONG → DPO Dataset (Reflection Needed)")
                print(f"📝 Final Prediction: {result['final_prediction']}")
        
        print(f"📊 Dataset Type: {result['dataset_type']}")
        print("="*80)
        
        return result
    
    def _merge_explanations(self, explanation1: str, explanation2: str) -> str:
        """Merge explanations from both models when they agree"""
        merged = f"Combined Analysis:\n\n"
        merged += f"Model 1 (Candle + Sentiment):\n{explanation1}\n\n"
        merged += f"Model 2 (News Summary):\n{explanation2}\n\n"
        merged += f"Consensus: Both models agree on the prediction, providing complementary analysis."
        return merged
    
    def create_supervised_sample(self, result: Dict) -> Dict:
        """Create sample for supervised learning dataset"""
        return {
            "instruction": "Analyze Bitcoin and Gold market data to predict Bitcoin's price movement.",
            "input": "",
            "output": result["final_explanation"]
        }
    
    def create_dpo_sample(self, result: Dict) -> Dict:
        """Create sample for DPO training dataset"""
        return {
            "prompt": "Analyze Bitcoin and Gold market data to predict Bitcoin's price movement.",
            "chosen": result["chosen"],
            "rejected": result["rejected"],
            "preferred": "chosen"
        } 
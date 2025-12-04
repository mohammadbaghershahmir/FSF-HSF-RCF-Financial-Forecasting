from typing import List, Union, Literal, Dict, Tuple
from utils.llm import FastChatLLM
from utils.enhanced_prompts import (
    CANDLE_SENTIMENT_INSTRUCTION_ENHANCED,
    CANDLE_SENTIMENT_INSTRUCTION_TEST_SHORT,
    NEWS_SUMMARY_INSTRUCTION_ENHANCED,
    NEWS_SUMMARY_INSTRUCTION_TEST_SHORT,
    ENHANCED_CANDLE_EXAMPLES,
    ENHANCED_NEWS_EXAMPLES
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_loader import VicunaModel
from explain_module.reflection_agent import ReflectionAgent
import torch
import random
import re as _re

class DualModelManager:
    """Manages two Vicuna models for sequential loading to avoid GPU memory issues"""
    
    def __init__(self):
        self.candle_model = None
        self.news_model = None
        self.current_model = None
        self.test_mode = False
        
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
        if self.model_manager.test_mode:
            self.llm = FastChatLLM(model=model_data["model"], tokenizer=model_data["tokenizer"], do_sample=False, temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.05, max_new_tokens=512)
        else:
            self.llm = FastChatLLM(model=model_data["model"], tokenizer=model_data["tokenizer"]) 
        
        # Format data
        formatted_market_data = self._format_market_data(market_data)
        formatted_sentiment_data = self._format_sentiment_data(sentiment_data)
        
        # Generate prompt with enhanced instruction
        from utils.enhanced_prompts import CANDLE_SENTIMENT_INSTRUCTION_ENHANCED, CANDLE_SENTIMENT_INSTRUCTION_TEST_SHORT
        prompt_template = CANDLE_SENTIMENT_INSTRUCTION_TEST_SHORT if self.model_manager.test_mode else CANDLE_SENTIMENT_INSTRUCTION_ENHANCED
        prompt = prompt_template.format(
            examples=ENHANCED_CANDLE_EXAMPLES,
            market_data=formatted_market_data,
            sentiment_data=formatted_sentiment_data
        )
        
        print("\n" + "="*100)
        print("🕯️  CANDLE + SENTIMENT MODEL")
        print("="*100)
        print("📊 INPUT DATA:")
        print("-" * 50)
        print(formatted_market_data)
        print("\n📈 SENTIMENT DATA:")
        print("-" * 50)
        print(formatted_sentiment_data)
        print("\n🤖 PROMPT:")
        print("-" * 50)
        print(prompt)
        print("-" * 50)
        
        # Get prediction
        print("\n🔄 GENERATING PREDICTION...")
        response = self.llm(prompt)
        
        # Parse response
        prediction, explanation = self._parse_response(response)
        
        print("\n📋 MODEL OUTPUT:")
        print("-" * 50)
        print("🔤 RAW RESPONSE:")
        print(response)
        print("-" * 50)
        print(f"🎯 PARSED PREDICTION: {prediction}")
        print(f"📄 PARSED EXPLANATION: {explanation}")
        print("="*100)
        
        return prediction, explanation
    
    def _format_market_data(self, market_data: Dict) -> str:
        """Format market data for the prompt"""
        formatted = "=== Bitcoin Market Data ===\n"
        formatted += f"Close: {market_data.get('btc_data', {}).get('close', 'N/A')}\n"
        formatted += f"Open: {market_data.get('btc_data', {}).get('open', 'N/A')}\n"
        formatted += f"High: {market_data.get('btc_data', {}).get('high', 'N/A')}\n"
        formatted += f"Low: {market_data.get('btc_data', {}).get('low', 'N/A')}\n"
        formatted += f"Volume: {market_data.get('btc_data', {}).get('volume', 'N/A')}\n\n"
        
        formatted += "=== Gold Market Data ===\n"
        formatted += f"Close: {market_data.get('gold_data', {}).get('close', 'N/A')}\n"
        formatted += f"Open: {market_data.get('gold_data', {}).get('open', 'N/A')}\n"
        formatted += f"High: {market_data.get('gold_data', {}).get('high', 'N/A')}\n"
        formatted += f"Low: {market_data.get('gold_data', {}).get('low', 'N/A')}\n"
        formatted += f"Volume: {market_data.get('gold_data', {}).get('volume', 'N/A')}\n"
        
        return formatted
    
    def _format_sentiment_data(self, sentiment_data: Dict) -> str:
        """Format sentiment data for the prompt with descriptive analysis"""
        formatted = "=== Bitcoin Sentiment ===\n"
        btc_sentiment = sentiment_data.get('btc_sentiment', {})
        btc_positive = btc_sentiment.get('positive', 0)
        btc_negative = btc_sentiment.get('negative', 0)
        btc_neutral = btc_sentiment.get('neutral', 0)

        # Scale to percentages if inputs are 0-1
        def _scale_percent(v1, v2, v3):
            max_val = max(v1, v2, v3)
            if max_val <= 1.0:
                return v1 * 100.0, v2 * 100.0, v3 * 100.0
            return v1, v2, v3
        btc_positive, btc_negative, btc_neutral = _scale_percent(btc_positive, btc_negative, btc_neutral)
        
        formatted += f"Positive: {btc_positive:.1f}%\n"
        formatted += f"Negative: {btc_negative:.1f}%\n"
        formatted += f"Neutral: {btc_neutral:.1f}%\n\n"
        
        formatted += "=== Gold Sentiment ===\n"
        gold_sentiment = sentiment_data.get('gold_sentiment', {})
        gold_positive = gold_sentiment.get('positive', 0)
        gold_negative = gold_sentiment.get('negative', 0)
        gold_neutral = gold_sentiment.get('neutral', 0)

        gold_positive, gold_negative, gold_neutral = _scale_percent(gold_positive, gold_negative, gold_neutral)
        
        formatted += f"Positive: {gold_positive:.1f}%\n"
        formatted += f"Negative: {gold_negative:.1f}%\n"
        formatted += f"Neutral: {gold_neutral:.1f}%\n\n"
        
        # Add descriptive analysis
        formatted += "=== Twitter Sentiment Analysis ===\n"
        sentiment_description = self._generate_sentiment_description(
            btc_positive, btc_negative, btc_neutral,
            gold_positive, gold_negative, gold_neutral
        )
        formatted += sentiment_description + "\n"
        
        return formatted
    
    def _generate_sentiment_description(self, btc_pos, btc_neg, btc_neu, gold_pos, gold_neg, gold_neu):
        """Generate a descriptive sentence about Twitter sentiment"""
        # Determine dominant sentiment for Bitcoin
        if btc_neg > btc_pos and btc_neg > btc_neu:
            btc_dominant = "bearish"
            btc_strength = "strong" if btc_neg > 60 else "moderate" if btc_neg > 40 else "weak"
        elif btc_pos > btc_neg and btc_pos > btc_neu:
            btc_dominant = "bullish"
            btc_strength = "strong" if btc_pos > 60 else "moderate" if btc_pos > 40 else "weak"
        else:
            btc_dominant = "neutral"
            btc_strength = "mixed"
        
        # Determine dominant sentiment for Gold
        if gold_neg > gold_pos and gold_neg > gold_neu:
            gold_dominant = "bearish"
            gold_strength = "strong" if gold_neg > 60 else "moderate" if gold_neg > 40 else "weak"
        elif gold_pos > gold_neg and gold_pos > gold_neu:
            gold_dominant = "bullish"
            gold_strength = "strong" if gold_pos > 60 else "moderate" if gold_pos > 40 else "weak"
        else:
            gold_dominant = "neutral"
            gold_strength = "mixed"
        
        # Generate descriptive sentence
        description = f"Twitter sentiment analysis reveals {btc_strength} {btc_dominant} sentiment for Bitcoin "
        description += f"({btc_pos:.1f}% positive, {btc_neg:.1f}% negative) and {gold_strength} {gold_dominant} sentiment for Gold "
        description += f"({gold_pos:.1f}% positive, {gold_neg:.1f}% negative). "
        
        # Add comparative analysis
        if btc_dominant == gold_dominant:
            description += "Both markets show similar sentiment trends, indicating correlated market psychology."
        elif btc_dominant == "bullish" and gold_dominant == "bearish":
            description += "The sentiment divergence suggests a risk-on environment with preference for Bitcoin over traditional safe havens."
        elif btc_dominant == "bearish" and gold_dominant == "bullish":
            description += "The sentiment divergence indicates a risk-off environment with investors favoring traditional safe havens over Bitcoin."
        else:
            description += "The mixed sentiment patterns suggest uncertain market conditions with varying investor preferences."
        
        return description
    
    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Parse the model response to extract prediction and explanation - IMPROVED VERSION"""
        response = response.strip()
        
        # Look for exact matches first
        if response.lower() == "positive":
            return "Positive", response
        elif response.lower() == "negative":
            return "Negative", response
        
        # Look for the standard format: "Bitcoin Price Movement: [Positive/Negative]"
        lines = response.split("\n")
        prediction = "Unknown"
        explanation = response  # Default to full response
        
        # Find prediction line
        prediction_line = None
        for i, line in enumerate(lines):
            if "Bitcoin Price Movement:" in line:
                prediction_line = i
                # Strictly capture a single label after the colon; ignore lines that show both options
                m = _re.search(r"Bitcoin\s*Price\s*Movement\s*:\s*(Positive|Negative)\b", line, flags=_re.IGNORECASE)
                if m:
                    label = m.group(1).strip().lower()
                    prediction = "Positive" if label == "positive" else "Negative"
                else:
                    prediction = "Unknown"
                break
        
        # If prediction found, extract explanation
        if prediction != "Unknown" and prediction_line is not None:
            # Look for explanation after prediction line
            explanation_lines = []
            
            # Check if there's an "Explanation:" line after prediction
            for i in range(prediction_line + 1, len(lines)):
                line = lines[i].strip()
                if line.startswith("Explanation:"):
                    # Extract everything after "Explanation:"
                    explanation_text = line.replace("Explanation:", "").strip()
                    if explanation_text:
                        explanation_lines.append(explanation_text)
                    # Add remaining lines
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip():
                            explanation_lines.append(lines[j].strip())
                    break
                elif line and not line.startswith("Bitcoin Price Movement:"):
                    # If no "Explanation:" found, take all non-empty lines after prediction
                    explanation_lines.append(line)
            
            # If we found explanation lines, use them
            if explanation_lines:
                explanation = " ".join(explanation_lines)
            else:
                # Fallback: use everything after the prediction line
                remaining_lines = []
                for i in range(prediction_line + 1, len(lines)):
                    if lines[i].strip():
                        remaining_lines.append(lines[i].strip())
                if remaining_lines:
                    explanation = " ".join(remaining_lines)
        
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
        
        # Default to Positive if still unknown (to avoid crashes)
        if prediction == "Unknown":
            prediction = "Positive"
        
        # Ensure explanation is not empty
        if not explanation or explanation.strip() == "":
            explanation = response
        
        return prediction, explanation

class NewsSummaryAgent:
    """Agent for analyzing news summary and sentiment"""
    
    def __init__(self, model_manager: DualModelManager):
        self.model_manager = model_manager
        self.llm = None
        
    def predict(self, news_summary: str, sentiment_data: Dict) -> Tuple[str, str]:
        """Predict Bitcoin price movement based on news summary and sentiment data"""
        # Load model
                model_data = self.model_manager.load_news_model()
        self.model_manager.current_model = model_data
        if self.model_manager.test_mode:
            self.llm = FastChatLLM(model=model_data["model"], tokenizer=model_data["tokenizer"], do_sample=False, temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.05, max_new_tokens=512)
        else:
            self.llm = FastChatLLM(model=model_data["model"], tokenizer=model_data["tokenizer"]) 
        
        # Format data
        formatted_sentiment_data = self._format_sentiment_data(sentiment_data)
        
        # Generate prompt with enhanced instruction
        from utils.enhanced_prompts import NEWS_SUMMARY_INSTRUCTION_ENHANCED, NEWS_SUMMARY_INSTRUCTION_TEST_SHORT
        prompt_template = NEWS_SUMMARY_INSTRUCTION_TEST_SHORT if self.model_manager.test_mode else NEWS_SUMMARY_INSTRUCTION_ENHANCED
        prompt = prompt_template.format(
            examples=ENHANCED_NEWS_EXAMPLES,
            news_summary=news_summary,
            sentiment_data=formatted_sentiment_data
        )
        
        print("\n" + "="*100)
        print("📰 NEWS SUMMARY MODEL")
        print("="*100)
        print("📰 NEWS SUMMARY:")
        print("-" * 50)
        print(news_summary)
        print("\n📈 SENTIMENT DATA:")
        print("-" * 50)
        print(formatted_sentiment_data)
        print("\n🤖 PROMPT:")
        print("-" * 50)
        print(prompt)
        print("-" * 50)
        
        # Get prediction
        print("\n🔄 GENERATING PREDICTION...")
        response = self.llm(prompt)
        
        # Parse response
        prediction, explanation = self._parse_response(response)
        
        print("\n📋 MODEL OUTPUT:")
        print("-" * 50)
        print("🔤 RAW RESPONSE:")
        print(response)
        print("-" * 50)
        print(f"🎯 PARSED PREDICTION: {prediction}")
        print(f"📄 PARSED EXPLANATION: {explanation}")
        print("="*100)
        
        return prediction, explanation
    
    def _format_sentiment_data(self, sentiment_data: Dict) -> str:
        """Format sentiment data for the prompt with descriptive analysis"""
        formatted = "=== Bitcoin Sentiment ===\n"
        btc_sentiment = sentiment_data.get('btc_sentiment', {})
        btc_positive = btc_sentiment.get('positive', 0)
        btc_negative = btc_sentiment.get('negative', 0)
        btc_neutral = btc_sentiment.get('neutral', 0)
        
        # Scale to percentages if inputs are 0-1
        def _scale_percent(v1, v2, v3):
            max_val = max(v1, v2, v3)
            if max_val <= 1.0:
                return v1 * 100.0, v2 * 100.0, v3 * 100.0
            return v1, v2, v3
        btc_positive, btc_negative, btc_neutral = _scale_percent(btc_positive, btc_negative, btc_neutral)
        
        formatted += f"Positive: {btc_positive:.1f}%\n"
        formatted += f"Negative: {btc_negative:.1f}%\n"
        formatted += f"Neutral: {btc_neutral:.1f}%\n\n"
        
        formatted += "=== Gold Sentiment ===\n"
        gold_sentiment = sentiment_data.get('gold_sentiment', {})
        gold_positive = gold_sentiment.get('positive', 0)
        gold_negative = gold_sentiment.get('negative', 0)
        gold_neutral = gold_sentiment.get('neutral', 0)
        
        gold_positive, gold_negative, gold_neutral = _scale_percent(gold_positive, gold_negative, gold_neutral)
        
        formatted += f"Positive: {gold_positive:.1f}%\n"
        formatted += f"Negative: {gold_negative:.1f}%\n"
        formatted += f"Neutral: {gold_neutral:.1f}%\n\n"
        
        # Add descriptive analysis
        formatted += "=== Twitter Sentiment Analysis ===\n"
        sentiment_description = self._generate_sentiment_description(
            btc_positive, btc_negative, btc_neutral,
            gold_positive, gold_negative, gold_neutral
        )
        formatted += sentiment_description + "\n"
        
        return formatted
    
    def _generate_sentiment_description(self, btc_pos, btc_neg, btc_neu, gold_pos, gold_neg, gold_neu):
        """Generate a descriptive sentence about Twitter sentiment"""
        # Determine dominant sentiment for Bitcoin
        if btc_neg > btc_pos and btc_neg > btc_neu:
            btc_dominant = "bearish"
            btc_strength = "strong" if btc_neg > 60 else "moderate" if btc_neg > 40 else "weak"
        elif btc_pos > btc_neg and btc_pos > btc_neu:
            btc_dominant = "bullish"
            btc_strength = "strong" if btc_pos > 60 else "moderate" if btc_pos > 40 else "weak"
        else:
            btc_dominant = "neutral"
            btc_strength = "mixed"
        
        # Determine dominant sentiment for Gold
        if gold_neg > gold_pos and gold_neg > gold_neu:
            gold_dominant = "bearish"
            gold_strength = "strong" if gold_neg > 60 else "moderate" if gold_neg > 40 else "weak"
        elif gold_pos > gold_neg and gold_pos > gold_neu:
            gold_dominant = "bullish"
            gold_strength = "strong" if gold_pos > 60 else "moderate" if gold_pos > 40 else "weak"
        else:
            gold_dominant = "neutral"
            gold_strength = "mixed"
        
        # Generate descriptive sentence
        description = f"Twitter sentiment analysis reveals {btc_strength} {btc_dominant} sentiment for Bitcoin "
        description += f"({btc_pos:.1f}% positive, {btc_neg:.1f}% negative) and {gold_strength} {gold_dominant} sentiment for Gold "
        description += f"({gold_pos:.1f}% positive, {gold_neg:.1f}% negative). "
        
        # Add comparative analysis
        if btc_dominant == gold_dominant:
            description += "Both markets show similar sentiment trends, indicating correlated market psychology."
        elif btc_dominant == "bullish" and gold_dominant == "bearish":
            description += "The sentiment divergence suggests a risk-on environment with preference for Bitcoin over traditional safe havens."
        elif btc_dominant == "bearish" and gold_dominant == "bullish":
            description += "The sentiment divergence indicates a risk-off environment with investors favoring traditional safe havens over Bitcoin."
        else:
            description += "The mixed sentiment patterns suggest uncertain market conditions with varying investor preferences."
        
        return description
    
    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Parse the model response to extract prediction and explanation - IMPROVED VERSION"""
        response = response.strip()
        
        # Look for exact matches first
        if response.lower() == "positive":
            return "Positive", response
        elif response.lower() == "negative":
            return "Negative", response
        
        # Look for the standard format: "Bitcoin Price Movement: [Positive/Negative]"
        lines = response.split("\n")
        prediction = "Unknown"
        explanation = response  # Default to full response
        
        # Find prediction line
        prediction_line = None
        for i, line in enumerate(lines):
            if "Bitcoin Price Movement:" in line:
                prediction_line = i
                # Strictly capture a single label after the colon; ignore lines that show both options
                m = _re.search(r"Bitcoin\s*Price\s*Movement\s*:\s*(Positive|Negative)\b", line, flags=_re.IGNORECASE)
                if m:
                    label = m.group(1).strip().lower()
                    prediction = "Positive" if label == "positive" else "Negative"
                else:
                    prediction = "Unknown"
                break
        
        # If prediction found, extract explanation
        if prediction != "Unknown" and prediction_line is not None:
            # Look for explanation after prediction line
            explanation_lines = []
            
            # Check if there's an "Explanation:" line after prediction
            for i in range(prediction_line + 1, len(lines)):
                line = lines[i].strip()
                if line.startswith("Explanation:"):
                    # Extract everything after "Explanation:"
                    explanation_text = line.replace("Explanation:", "").strip()
                    if explanation_text:
                        explanation_lines.append(explanation_text)
                    # Add remaining lines
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip():
                            explanation_lines.append(lines[j].strip())
                    break
                elif line and not line.startswith("Bitcoin Price Movement:"):
                    # If no "Explanation:" found, take all non-empty lines after prediction
                    explanation_lines.append(line)
            
            # If we found explanation lines, use them
            if explanation_lines:
                explanation = " ".join(explanation_lines)
            else:
                # Fallback: use everything after the prediction line
                remaining_lines = []
                for i in range(prediction_line + 1, len(lines)):
                    if lines[i].strip():
                        remaining_lines.append(lines[i].strip())
                if remaining_lines:
                    explanation = " ".join(remaining_lines)
        
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
        
        # Default to Positive if still unknown (to avoid crashes)
        if prediction == "Unknown":
            prediction = "Positive"
        
        # Ensure explanation is not empty
        if not explanation or explanation.strip() == "":
            explanation = response
        
        return prediction, explanation
    
    def _format_sentiment_data(self, sentiment_data: Dict) -> str:
        """Format sentiment data for the prompt with descriptive analysis"""
        formatted = "=== Bitcoin Sentiment ===\n"
        btc_sentiment = sentiment_data.get('btc_sentiment', {})
        btc_positive = btc_sentiment.get('positive', 0)
        btc_negative = btc_sentiment.get('negative', 0)
        btc_neutral = btc_sentiment.get('neutral', 0)
        
        formatted += f"Positive: {btc_positive:.1f}%\n"
        formatted += f"Negative: {btc_negative:.1f}%\n"
        formatted += f"Neutral: {btc_neutral:.1f}%\n\n"
        
        formatted += "=== Gold Sentiment ===\n"
        gold_sentiment = sentiment_data.get('gold_sentiment', {})
        gold_positive = gold_sentiment.get('positive', 0)
        gold_negative = gold_sentiment.get('negative', 0)
        gold_neutral = gold_sentiment.get('neutral', 0)
        
        formatted += f"Positive: {gold_positive:.1f}%\n"
        formatted += f"Negative: {gold_negative:.1f}%\n"
        formatted += f"Neutral: {gold_neutral:.1f}%\n\n"
        
        # Add descriptive analysis
        formatted += "=== Twitter Sentiment Analysis ===\n"
        sentiment_description = self._generate_sentiment_description(
            btc_positive, btc_negative, btc_neutral,
            gold_positive, gold_negative, gold_neutral
        )
        formatted += sentiment_description + "\n"
        
        return formatted
    
    def _generate_sentiment_description(self, btc_pos, btc_neg, btc_neu, gold_pos, gold_neg, gold_neu):
        """Generate a descriptive sentence about Twitter sentiment"""
        # Determine dominant sentiment for Bitcoin
        if btc_neg > btc_pos and btc_neg > btc_neu:
            btc_dominant = "bearish"
            btc_strength = "strong" if btc_neg > 60 else "moderate" if btc_neg > 40 else "weak"
        elif btc_pos > btc_neg and btc_pos > btc_neu:
            btc_dominant = "bullish"
            btc_strength = "strong" if btc_pos > 60 else "moderate" if btc_pos > 40 else "weak"
        else:
            btc_dominant = "neutral"
            btc_strength = "mixed"
        
        # Determine dominant sentiment for Gold
        if gold_neg > gold_pos and gold_neg > gold_neu:
            gold_dominant = "bearish"
            gold_strength = "strong" if gold_neg > 60 else "moderate" if gold_neg > 40 else "weak"
        elif gold_pos > gold_neg and gold_pos > gold_neu:
            gold_dominant = "bullish"
            gold_strength = "strong" if gold_pos > 60 else "moderate" if gold_pos > 40 else "weak"
        else:
            gold_dominant = "neutral"
            gold_strength = "mixed"
        
        # Generate descriptive sentence
        description = f"Twitter sentiment analysis reveals {btc_strength} {btc_dominant} sentiment for Bitcoin "
        description += f"({btc_pos:.1f}% positive, {btc_neg:.1f}% negative) and {gold_strength} {gold_dominant} sentiment for Gold "
        description += f"({gold_pos:.1f}% positive, {gold_neg:.1f}% negative). "
        
        # Add comparative analysis
        if btc_dominant == gold_dominant:
            description += "Both markets show similar sentiment trends, indicating correlated market psychology."
        elif btc_dominant == "bullish" and gold_dominant == "bearish":
            description += "The sentiment divergence suggests a risk-on environment with preference for Bitcoin over traditional safe havens."
        elif btc_dominant == "bearish" and gold_dominant == "bullish":
            description += "The sentiment divergence indicates a risk-off environment with investors favoring traditional safe havens over Bitcoin."
        else:
            description += "The mixed sentiment patterns suggest uncertain market conditions with varying investor preferences."
        
        return description
    
    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Parse the model response to extract prediction and explanation - IMPROVED VERSION"""
        response = response.strip()
        
        # Look for exact matches first
        if response.lower() == "positive":
            return "Positive", response
        elif response.lower() == "negative":
            return "Negative", response
        
        # Look for the standard format: "Bitcoin Price Movement: [Positive/Negative]"
        lines = response.split("\n")
        prediction = "Unknown"
        explanation = response  # Default to full response
        
        # Find prediction line
        prediction_line = None
        for i, line in enumerate(lines):
            if "Bitcoin Price Movement:" in line:
                prediction_line = i
                # Strictly capture a single label after the colon; ignore lines that show both options
                m = _re.search(r"Bitcoin\s*Price\s*Movement\s*:\s*(Positive|Negative)\b", line, flags=_re.IGNORECASE)
                if m:
                    label = m.group(1).strip().lower()
                    prediction = "Positive" if label == "positive" else "Negative"
                else:
                    prediction = "Unknown"
                break
        
        # If prediction found, extract explanation
        if prediction != "Unknown" and prediction_line is not None:
            # Look for explanation after prediction line
            explanation_lines = []
            
            # Check if there's an "Explanation:" line after prediction
            for i in range(prediction_line + 1, len(lines)):
                line = lines[i].strip()
                if line.startswith("Explanation:"):
                    # Extract everything after "Explanation:"
                    explanation_text = line.replace("Explanation:", "").strip()
                    if explanation_text:
                        explanation_lines.append(explanation_text)
                    # Add remaining lines
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip():
                            explanation_lines.append(lines[j].strip())
                    break
                elif line and not line.startswith("Bitcoin Price Movement:"):
                    # If no "Explanation:" found, take all non-empty lines after prediction
                    explanation_lines.append(line)
            
            # If we found explanation lines, use them
            if explanation_lines:
                explanation = " ".join(explanation_lines)
            else:
                # Fallback: use everything after the prediction line
                remaining_lines = []
                for i in range(prediction_line + 1, len(lines)):
                    if lines[i].strip():
                        remaining_lines.append(lines[i].strip())
                if remaining_lines:
                    explanation = " ".join(remaining_lines)
        
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
        
        # Default to Positive if still unknown (to avoid crashes)
        if prediction == "Unknown":
            prediction = "Positive"
        
        # Ensure explanation is not empty
        if not explanation or explanation.strip() == "":
            explanation = response
        
        return prediction, explanation

class DualPredictionAgent:
    """Agent that coordinates predictions from both models and manages reflection process"""
    
    def __init__(self, model_manager: DualModelManager):
        self.model_manager = model_manager
        self.candle_agent = CandleSentimentAgent(model_manager)
        self.news_agent = NewsSummaryAgent(model_manager)
        self.reflection_agent = ReflectionAgent(model_manager)
        
    def predict(self, market_data: Dict, sentiment_data: Dict, news_summary: str, target: str) -> Dict:
        """Make predictions using both models and coordinate the results"""
        
        # Get predictions from both models
        candle_prediction, candle_explanation = self.candle_agent.predict(market_data, sentiment_data)
        news_prediction, news_explanation = self.news_agent.predict(news_summary, sentiment_data)
        
        # Check if models agree
        models_agree = candle_prediction == news_prediction
        
        # Check if predictions are correct
        candle_correct = candle_prediction == target
        news_correct = news_prediction == target
        
        print(f"\n{'='*100}")
        print("🔄 DUAL PREDICTION RESULTS")
        print(f"{'='*100}")
        print(f"🕯️  Candle + Sentiment: {candle_prediction} ({'✅' if candle_correct else '❌'})")
        print(f"📰 News Summary: {news_prediction} ({'✅' if news_correct else '❌'})")
        print(f"🎯 Target: {target}")
        print(f"🤝 Models Agree: {'✅' if models_agree else '❌'}")
        print(f"{'='*100}")
        
        # Test mode: no reflection, just return candle prediction
        if self.model_manager.test_mode:
            print(f"🧪 TEST MODE: No reflection process")
            result = {
                "candle_prediction": candle_prediction,
                "candle_explanation": candle_explanation,
                "news_prediction": news_prediction,
                "news_explanation": news_explanation,
                "models_agree": models_agree,
                "candle_correct": candle_correct,
                "news_correct": news_correct,
                "final_prediction": candle_prediction,
                "final_explanation": candle_explanation,
                "dataset_type": "test"
            }
            print(f"📝 Final Prediction: {result['final_prediction']}")
            print(f"📄 Final Explanation: {result['final_explanation']}")
            print("="*100)
            return result
        
        # If both models agree and are correct, add to supervised dataset
        if models_agree and candle_correct:
            print(f"✅ SUPERVISED: Both models agree and are correct")
            result = {
                "candle_prediction": candle_prediction,
                "candle_explanation": candle_explanation,
                "news_prediction": news_prediction,
                "news_explanation": news_explanation,
                "models_agree": models_agree,
                "candle_correct": candle_correct,
                "news_correct": news_correct,
                "final_prediction": candle_prediction,
                "final_explanation": candle_explanation,
                "dataset_type": "supervised"
            }
            return result
        
        # Otherwise, trigger reflection process for DPO dataset
        print(f"🔄 REFLECTION: Models disagree or are incorrect, triggering reflection process")
        
        # Determine which prediction to use for reflection
        if candle_correct and not news_correct:
            incorrect_prediction = news_prediction
            incorrect_explanation = news_explanation
            correct_prediction = candle_prediction
            correct_explanation = candle_explanation
        elif news_correct and not candle_correct:
            incorrect_prediction = candle_prediction
            incorrect_explanation = candle_explanation
            correct_prediction = news_prediction
            correct_explanation = news_explanation
        else:
            # Both are incorrect, use candle prediction as base
            incorrect_prediction = candle_prediction
            incorrect_explanation = candle_explanation
            correct_prediction = target
            correct_explanation = f"Correct prediction should be {target} based on market analysis"
        
        # Generate DPO pairs using reflection
        dpo_pairs = self.reflection_agent.generate_dpo_pairs(
            incorrect_prediction=incorrect_prediction,
            incorrect_explanation=incorrect_explanation,
            correct_prediction=correct_prediction,
            correct_explanation=correct_explanation,
            market_data=market_data,
            sentiment_data=sentiment_data,
            news_summary=news_summary
        )
        
        result = {
            "candle_prediction": candle_prediction,
            "candle_explanation": candle_explanation,
            "news_prediction": news_prediction,
            "news_explanation": news_explanation,
            "models_agree": models_agree,
            "candle_correct": candle_correct,
            "news_correct": news_correct,
            "final_prediction": candle_prediction,
            "final_explanation": candle_explanation,
            "dataset_type": "dpo",
            "dpo_pairs": dpo_pairs
        }
        
        return result

    def create_dpo_sample(self, result: Dict) -> List[Dict]:
        """Create DPO samples from prediction result"""
        if result.get("dataset_type") == "dpo" and "dpo_pairs" in result:
            return result["dpo_pairs"]
        else:
            return []

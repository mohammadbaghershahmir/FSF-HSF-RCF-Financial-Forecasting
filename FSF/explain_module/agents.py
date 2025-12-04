from typing import List, Union, Literal, Dict
from utils.llm import OpenAILLM, NShotLLM, FastChatLLM
from utils.prompts import REFLECT_INSTRUCTION, PREDICT_INSTRUCTION, PREDICT_REFLECT_INSTRUCTION, REFLECTION_HEADER
from utils.fewshots import PREDICT_EXAMPLES
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_loader import VicunaModel
model_data = VicunaModel.get_model(
    model_name="lmsys/vicuna-7b-v1.5",
    quantization=True,
    device_map="auto"
)

class PredictAgent:
    THRESHOLD_DEFAULT = 0.7
    def __init__(self,
                 market: str,
                 market_data: Dict[str, Dict],
                 target: str,
                 price_change: float,
                 predict_llm = FastChatLLM(model=model_data["model"], tokenizer=model_data["tokenizer"]),
                 threshold: float = THRESHOLD_DEFAULT
                 ) -> None:

        self.market = market
        self.market_data = market_data
        self.target = target
        self.price_change = price_change
        self.predict_llm = predict_llm
        self.probability: float = 1.0
        self.threshold: float = threshold
        self.is_main_market = market == "BTC"  # آیا این بازار اصلی است؟
        self.related_market = "GOLD" if self.is_main_market else "BTC"
        self.prediction = ''
        self.predict_prompt = PREDICT_INSTRUCTION
        self.predict_examples = PREDICT_EXAMPLES
        self.__reset_agent()

    def format_market_data(self) -> str:
        """فرمت‌بندی داده‌های بازار برای استفاده در پرامپت با تاکید بر بازار اصلی"""
        # اگر بازار اصلی است، اول داده‌های آن را نمایش می‌دهیم
        if self.is_main_market:
            formatted_data = f"=== Main Market ({self.market}) Data ===\n"
        else:
            formatted_data = f"=== Secondary Market ({self.market}) Data ===\n"

        # Basic price data
        main_data = self.market_data[self.market.lower() + '_data']
        formatted_data += f"High: {main_data['high']:.6f}\n"
        formatted_data += f"Low: {main_data['low']:.6f}\n"
        formatted_data += f"Open: {main_data['open']:.6f}\n"
        #formatted_data += f"Close: {main_data['close']:.6f}\n"
        formatted_data += f"Volume: {main_data['volume']:.2f}\n"
        
        # Technical features
        if 'price_range' in main_data:
            formatted_data += f"Price Range: {main_data['price_range']:.4f}\n"
        if 'close_open_ratio' in main_data:
            formatted_data += f"Close/Open Ratio: {main_data['close_open_ratio']:.4f}\n"
        if 'volume_log' in main_data:
            formatted_data += f"Log Volume: {main_data['volume_log']:.2f}\n"
        formatted_data += "\n"

        # داده‌های بازار مرتبط
        if self.is_main_market:
            formatted_data += f"=== Secondary Market ({self.related_market}) Data ===\n"
        else:
            formatted_data += f"=== Main Market ({self.related_market}) Data ===\n"

        # Related market data
        related_data = self.market_data[self.related_market.lower() + '_data']
        formatted_data += f"High: {related_data['high']:.6f}\n"
        formatted_data += f"Low: {related_data['low']:.6f}\n"
        formatted_data += f"Open: {related_data['open']:.6f}\n"
        #formatted_data += f"Close: {related_data['close']:.6f}\n"
        formatted_data += f"Volume: {related_data['volume']:.2f}\n"
        
        # Technical features for related market
        if 'price_range' in related_data:
            formatted_data += f"Price Range: {related_data['price_range']:.4f}\n"
        if 'close_open_ratio' in related_data:
            formatted_data += f"Close/Open Ratio: {related_data['close_open_ratio']:.4f}\n"
        if 'volume_log' in related_data:
            formatted_data += f"Log Volume: {related_data['volume_log']:.2f}\n"

        return formatted_data

    def format_sentiment_data(self) -> str:
        """فرمت‌بندی داده‌های احساسات برای استفاده در پرامپت با تاکید بر بازار اصلی"""
        if self.is_main_market:
            formatted_data = f"=== Main Market ({self.market}) Sentiment ===\n"
        else:
            formatted_data = f"=== Secondary Market ({self.market}) Sentiment ===\n"

        for key, value in self.market_data[self.market.lower() + '_sentiment'].items():
            formatted_data += f"{key}: {value*100:.1f}%\n"
        
        if self.is_main_market:
            formatted_data += f"\n=== Secondary Market ({self.related_market}) Sentiment ===\n"
        else:
            formatted_data += f"\n=== Main Market ({self.related_market}) Sentiment ===\n"

        for key, value in self.market_data[self.related_market.lower() + '_sentiment'].items():
            formatted_data += f"{key}: {value*100:.1f}%\n"

        return formatted_data

    def predict(self) -> str:
        """پیش‌بینی حرکت قیمت با در نظر گرفتن تاثیر متقابل بازارها"""
        market_data = self.format_market_data()
        sentiment_data = self.format_sentiment_data()

        prompt = PREDICT_INSTRUCTION.format(
            market=self.market,
            related_market=self.related_market,
            market_data=market_data,
            sentiment_data=sentiment_data
        )

        prediction = self.predict_llm.generate(prompt)
        return prediction

    def reflect(self, prediction: str) -> str:
        """تحلیل و بازخورد از پیش‌بینی قبلی"""
        # Don't reveal the actual price change to avoid data leakage
        # Instead, provide general feedback based on prediction correctness
        if self.is_correct():
            outcome = "Your prediction was correct"
        else:
            outcome = "Your prediction was incorrect"
            
        prompt = PREDICT_REFLECT_INSTRUCTION.format(
            prediction=prediction,
            outcome=outcome
        )

        reflection = self.predict_llm.generate(prompt)
        return reflection

    def run(self, reset=True) -> None:
        if reset:
            self.__reset_agent()

        market_data = self.format_market_data()
        sentiment_data = self.format_sentiment_data()
        
        prompt = f"Market Data:\n{market_data}\n\nSentiment Analysis:\n{sentiment_data}\n\nPrice Movement: "
        self.scratchpad += prompt
        print(prompt, end="")

        self.scratchpad += self.prompt_agent()
        response = self.scratchpad.split('Price Movement: ')[-1]
        response_block = response.strip()
        lines = response_block.splitlines()
        
        # Improved prediction extraction and validation
        prediction = None
        probability = None
        
        # Look for prediction in first few lines
        for line in lines[:3]:  # Check first 3 lines
            line = line.lower().strip()
            if "positive" in line:
                prediction = "positive"
                break
            elif "negative" in line:
                prediction = "negative"
                break
        
        # If no valid prediction found, default to negative to avoid positive bias
        if prediction is None:
            prediction = "negative"
            probability = 0.5  # Default probability
        else:
            # Look for probability/confidence in next few lines
            for line in lines[1:5]:  # Check next 4 lines
                # Look for "Confidence Probability: 0.75" format
                conf_prob_match = re.search(r'confidence\s+probability\s*[:\-]?\s*([\d\.]+)', line, re.I)
                # Look for "Confidence Level: 0.75" format
                conf_level_match = re.search(r'confidence\s+level\s*[:\-]?\s*([\d\.]+)', line, re.I)
                # Look for "Probability: 0.75" format
                prob_match = re.search(r'probability\s*[:\-]?\s*([\d\.]+)', line, re.I)
                # Look for "Confidence: 0.75" format
                conf_match = re.search(r'confidence\s*[:\-]?\s*([\d\.]+)', line, re.I)
                
                if conf_prob_match:
                    p = float(conf_prob_match.group(1))
                    probability = p / 100 if p > 1 else p
                    break
                elif conf_level_match:
                    p = float(conf_level_match.group(1))
                    probability = p / 100 if p > 1 else p
                    break
                elif prob_match:
                    p = float(prob_match.group(1))
                    probability = p / 100 if p > 1 else p
                    break
                elif conf_match:
                    p = float(conf_match.group(1))
                    probability = p / 100 if p > 1 else p
                    break
        
        # If no probability found, use default
        if probability is None:
            probability = 0.5
        
        # Validate and clamp probability to [0, 1]
        probability = max(0.0, min(1.0, probability))

        self.prediction = prediction
        self.probability = probability

        print(response, end="\n\n\n\n")
        self.finished = True

    def prompt_agent(self) -> str:
        return self.predict_llm(self._build_agent_prompt())

    def _build_agent_prompt(self) -> str:
        return self.predict_prompt.format(
                            market=self.market,
                            related_market=self.related_market,
                            market_data=self.format_market_data(),
                            sentiment_data=self.format_sentiment_data())

    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        # Validate prediction format
        if self.prediction not in ["positive", "negative"]:
            return False
        
        # Validate probability is between 0 and 1
        #if not (0 <= self.probability <= 1):
         #   return False
        
        # Validate target format
        if self.target.lower() not in ["positive", "negative"]:
            return False
        
        # Check if prediction matches target and probability exceeds threshold
        return (self.target.lower() == self.prediction.lower()) #and (self.probability >= self.threshold)
    
    def get_threshold_status(self) -> str:
        """Get detailed status about threshold decision"""
        if self.probability >= self.threshold:
            if self.target.lower() == self.prediction.lower():
                return "ACCEPTED_CORRECT"
            else:
                return "ACCEPTED_INCORRECT"
        else:
            return "REJECTED_LOW_CONFIDENCE"
    
    def get_confidence_level(self) -> str:
        """Get confidence level description"""
        if self.probability >= 0.71:
            return "High"
        elif self.probability >= 0.31:
            return "Medium"
        else:
            return "Low"

    def __reset_agent(self) -> None:
        self.finished = False
        self.scratchpad: str = ''


class PredictReflectAgent(PredictAgent):
    def __init__(self,
                 ticker: str,
                 market_data: Dict[str, Dict],
                 target: str,
                 price_change: float,
                 predict_llm = FastChatLLM(model=model_data["model"], tokenizer=model_data["tokenizer"]),
                 reflect_llm = FastChatLLM(model=model_data["model"], tokenizer=model_data["tokenizer"]),
                 threshold: float = 0.7
                 ) -> None:

        super().__init__(ticker, market_data, target, price_change, predict_llm, threshold)
        self.predict_llm = predict_llm
        self.reflect_llm = reflect_llm
        self.reflect_prompt = REFLECT_INSTRUCTION
        self.agent_prompt = PREDICT_REFLECT_INSTRUCTION
        self.reflections = []
        self.reflections_str: str = ''

    def run(self, reset=True) -> None:
        if self.is_finished() and not self.is_correct():
            self.reflect()

        PredictAgent.run(self, reset=reset)

    def reflect(self) -> None:
        print('Reflecting...\n')
        reflection = self.prompt_reflection()
        self.reflections += [reflection]
        self.reflections_str = format_reflections(self.reflections)
        print(self.reflections_str, end="\n\n\n\n")

    def prompt_reflection(self) -> str:
        return self.reflect_llm(self._build_reflection_prompt())

    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
                            ticker=self.market,
                            scratchpad=self.scratchpad)

    def _build_agent_prompt(self) -> str:
        # اگر هیچ بازتابی نداریم، از پرامپت پیش‌بینی استفاده می‌کنیم
        if not self.reflections:
            return self.predict_prompt.format(
                market=self.market,
                related_market=self.related_market,
                market_data=self.format_market_data(),
                sentiment_data=self.format_sentiment_data()
            )
        
        # اگر بازتاب داریم، از پرامپت بازتاب استفاده می‌کنیم
        return self.predict_prompt.format(
            market=self.market,
            related_market=self.related_market,
            examples=self.predict_examples,
            reflections=self.reflections_str,
            market_data=self.format_market_data(),
            sentiment_data=self.format_sentiment_data()
        )

    def run_n_shots(self, model, tokenizer, num_shots=4, reset=True) -> None:
        self.llm = NShotLLM(model, tokenizer, num_shots)
        PredictAgent.run(self, reset=reset)


def format_reflections(reflections: List[str], header: str = REFLECTION_HEADER) -> str:
    if reflections == []:
        return ''
    else:
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])



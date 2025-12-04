from typing import List, Union, Literal, Dict
from utils.llm import OpenAILLM, NShotLLM, FastChatLLM
from utils.prompts import REFLECT_INSTRUCTION, PREDICT_INSTRUCTION, PREDICT_REFLECT_INSTRUCTION, REFLECTION_HEADER
from utils.fewshots import PREDICT_EXAMPLES

from transformers import AutoModelForCausalLM, AutoTokenizer
from model_loader import VicunaModel
model_data = VicunaModel.get_model(
    model_name="lmsys/vicuna-7b-v1.5",
    quantization=True,
    device_map="auto"
)

class PredictAgent:
    def __init__(self,
                 market: str,
                 market_data: Dict[str, Dict],
                 target: str,
                 price_change: float,
                 predict_llm = FastChatLLM(model=model_data["model"], tokenizer=model_data["tokenizer"])
                 ) -> None:

        self.market = market
        self.market_data = market_data
        self.target = target
        self.price_change = price_change
        self.predict_llm = predict_llm
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

        formatted_data += f"High: {self.market_data[self.market.lower() + '_data']['high']:}\n"
        formatted_data += f"Low: {self.market_data[self.market.lower() + '_data']['low']:}\n"
        formatted_data += f"Volume: {self.market_data[self.market.lower() + '_data']['volume']:}\n\n"

        # داده‌های بازار مرتبط
        if self.is_main_market:
            formatted_data += f"=== Secondary Market ({self.related_market}) Data ===\n"
        else:
            formatted_data += f"=== Main Market ({self.related_market}) Data ===\n"

        formatted_data += f"High: {self.market_data[self.related_market.lower() + '_data']['high']:}\n"
        formatted_data += f"Low: {self.market_data[self.related_market.lower() + '_data']['low']:}\n"
        formatted_data += f"Volume: {self.market_data[self.related_market.lower() + '_data']['volume']:}\n"

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
        prompt = PREDICT_REFLECT_INSTRUCTION.format(
            prediction=prediction,
            outcome=f"Actual {self.market} price change: {self.price_change:}%"
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
        self.prediction = response.split()[0]
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
        return (self.target.lower() == self.prediction.lower())
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
                 reflect_llm = FastChatLLM(model=model_data["model"], tokenizer=model_data["tokenizer"])
                 ) -> None:

        super().__init__(ticker, market_data, target, price_change, predict_llm)
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



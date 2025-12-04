from typing import Dict, Tuple
from utils.llm import FastChatLLM
from utils.enhanced_prompts import ENHANCED_CANDLE_EXAMPLES
from model_loader import VicunaModel
from explain_module.reflection_agent import ReflectionAgent
import torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer


class MultiModelManager:
	"""Manages k Vicuna models for ensemble prediction with voting mechanism"""
	def __init__(self, num_models: int = 3):
		if num_models % 2 == 0:
			raise ValueError("Number of models must be odd for voting mechanism")
		self.num_models = num_models
		self.models = [None] * num_models
		self.current_model_index = None
		self.test_mode = False
		self.dpo_model_dir = "./saved_models/tuning_tiny_vicuna_dpo_checkpoints/dpo_model"
		self.dpo_tokenizer_dir = "./saved_models/tuning_tiny_vicuna_dpo_checkpoints/dpo_tokenizer"

	def load_model(self, model_index: int):
		if model_index < 0 or model_index >= self.num_models:
			raise ValueError(f"Model index {model_index} out of range [0, {self.num_models-1}]")
		if self.models[model_index] is None:
			print(f"Loading Model {model_index + 1}/{self.num_models}...")
			model_data = None
			if self.test_mode:
				print("Loading DPO model for TEST mode...")
				model = AutoModelForCausalLM.from_pretrained(self.dpo_model_dir, device_map="auto", torch_dtype="auto", low_cpu_mem_usage=True)
				tokenizer = AutoTokenizer.from_pretrained(self.dpo_tokenizer_dir, use_fast=False, padding_side="left")
				model_data = {"model": model, "tokenizer": tokenizer}
				model_name="DPO Model"
			else:
				model_data = VicunaModel.get_model(
				model_name="lmsys/vicuna-7b-v1.5",
				quantization=True,
				device_map="auto"
			)
			self.models[model_index] = model_data
			#print(f"Model {model_name} Loaded.")
		self.current_model_index = model_index
		return self.models[model_index]

	def unload_model(self, model_index: int):
		if model_index < 0 or model_index >= self.num_models:
			return
		if self.models[model_index] is not None:
			self.models[model_index] = None
			torch.cuda.empty_cache()
			print(f"Model {model_index + 1} unloaded from GPU memory.")

	def unload_all_models(self):
		for i in range(self.num_models):
			self.unload_model(i)
		self.current_model_index = None


class UnifiedPredictionAgent:
	"""Agent that makes predictions using CANDLE + SENTIMENT + News Summary for a single model"""
	def __init__(self, model_manager: "MultiModelManager", model_index: int):
		self.model_manager = model_manager
		self.model_index = model_index
		self.llm = None

	def predict(self, market_data: Dict, sentiment_data: Dict, news_summary: str) -> Tuple[str, str]:
		model_data = self.model_manager.load_model(self.model_index)
		self.llm = FastChatLLM(model=model_data["model"], tokenizer=model_data["tokenizer"])
		formatted_market_data = self._format_market_data(market_data)
		formatted_sentiment_data = self._format_sentiment_data(sentiment_data)
		formatted_news_summary = self._format_news_summary(news_summary)

		print("\n" + "="*100)
		print(f"MODEL {self.model_index + 1} - INPUTS")
		print("="*100)
		print("Market Data (formatted):\n" + formatted_market_data)
		print("Market Data (raw): ", market_data)
		print("Sentiment Data:" + formatted_sentiment_data)
		print("News Summary:\n" + formatted_news_summary)

		base_prompt = (
			"You are a Bitcoin market analyst. Analyze the data and predict Bitcoin's next price movement.\n\n"
			"CRITICAL: You MUST predict ONLY \"Positive\" or \"Negative\". NO other predictions are allowed.\n\n"
			"OUTPUT FORMAT (STRICT):\n"
			"Bitcoin Price Movement: [Positive/Negative]\n\n"
			"Explanation: [Provide a clear analysis in one paragraph of 100-150 words explaining your reasoning based on the data provided.]\n\n"
			"{examples}\n\n{market_data}"
		)
		unified_prompt = self._create_unified_prompt(
			base_prompt,
			formatted_market_data,
			formatted_sentiment_data,
			formatted_news_summary
		)
		print("\nPROMPT:\n" + unified_prompt[:1200])
		response = self.llm(unified_prompt)
		print("\n" + "-"*100)
		print(f"RAW RESPONSE (Model {self.model_index + 1}):\n" + str(response))
		prediction, explanation = self._parse_response(response)
		print("\nParsed Prediction:", prediction)
		print("Parsed Explanation (full):", explanation or "")
		return prediction, explanation

	def _create_unified_prompt(self, base_prompt: str, market_data: str, sentiment_data: str, news_summary: str) -> str:
		unified_content = (
			"Market Data (Candlestick Analysis):\n" + market_data + "\n\n"
			+ "Sentiment Data (Twitter Analysis):\n" + sentiment_data + "\n\n"
			+ "News Summary (Market News):\n" + news_summary + "\n\n"
			+ "Based on the above comprehensive market data, sentiment analysis, and news summary, predict Bitcoin's price movement."
		)
		unified_prompt = base_prompt.replace("{examples}", ENHANCED_CANDLE_EXAMPLES)
		unified_prompt = unified_prompt.replace("{market_data}", unified_content)
		return unified_prompt

	def _format_market_data(self, market_data: Dict) -> str:
		if not market_data:
			return "No market data available"
		def extract(md, prefix):
			d = md.get(f"{prefix}_data") or md.get(prefix) or {}
			return {
				"high": d.get("high", d.get("High", "0.0")),
				"low": d.get("low", d.get("Low", "0.0")),
				"open": d.get("open", d.get("Open", "0.0")),
				"volume": d.get("volume", d.get("Volume", "0.0")),
			}
		btc = extract(market_data, "btc")
		gold = extract(market_data, "gold")

		# In test mode, include only BTC candles
		if self.model_manager.test_mode:
			formatted = "=== Bitcoin Market Data ===\n"
			formatted += f"Open: {btc['open']}\n"
			formatted += f"High: {btc['high']}\n"
			formatted += f"Low: {btc['low']}\n"
			formatted += f"Volume: {btc['volume']}\n\n"
			return formatted
		formatted = "=== Bitcoin Market Data ===\n"
		formatted += f"Open: {btc['open']}\n"
		formatted += f"High: {btc['high']}\n"
		formatted += f"Low: {btc['low']}\n"
		formatted += f"Volume: {btc['volume']}\n\n"
		formatted += "=== Gold Market Data ===\n"
		formatted += f"Open: {gold['open']}\n"
		formatted += f"High: {gold['high']}\n"
		formatted += f"Low: {gold['low']}\n"
		formatted += f"Volume: {gold['volume']}\n\n"
		return formatted

	def _format_sentiment_data(self, sentiment_data: Dict) -> str:
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
		return formatted

	def _format_news_summary(self, news_summary: str) -> str:
		if not news_summary or news_summary.strip() == "":
			return "No news summary available"
		return news_summary.strip()

	def _parse_response(self, response: str) -> Tuple[str, str]:
		response = (response or '').strip()
		if not response:
			return 'Unknown', ''
		low = response.lower()
		if low == 'positive':
			return 'Positive', response
		if low == 'negative':
			return 'Negative', response
		prediction = 'Unknown'
		explanation = response
		lines = response.splitlines()
		prediction_line = None
		for i, line in enumerate(lines):
			if 'Bitcoin Price Movement:' in line:
				prediction_line = i
				if 'Positive' in line:
					prediction = 'Positive'
				elif 'Negative' in line:
					prediction = 'Negative'
				break
		if prediction != 'Unknown' and prediction_line is not None:
			explanation_lines = []
			for j in range(prediction_line + 1, len(lines)):
				line = lines[j].strip()
				if line.startswith('Explanation:'):
					expl_text = line.replace('Explanation:', '').strip()
					if expl_text:
						explanation_lines.append(expl_text)
					for k in range(j + 1, len(lines)):
						if lines[k].strip():
							explanation_lines.append(lines[k].strip())
					break
				elif line and not line.startswith('Bitcoin Price Movement:'):
					explanation_lines.append(line)
			if explanation_lines:
				explanation = ' '.join(explanation_lines)
			else:
				remaining = [ln.strip() for ln in lines[prediction_line+1:] if ln.strip()]
				if remaining:
					explanation = ' '.join(remaining)
		if prediction == 'Unknown':
			for line in lines:
				if 'Price Movement:' in line:
					if 'Positive' in line:
						prediction = 'Positive'
					elif 'Negative' in line:
						prediction = 'Negative'
					break
		if prediction == 'Unknown':
			if 'positive' in low:
				prediction = 'Positive'
			elif 'negative' in low:
				prediction = 'Negative'
		if not explanation or explanation.strip() == '':
			explanation = response
		return prediction, explanation


class MultiModelPredictionAgent:
	"""Main agent that coordinates k models and implements voting mechanism"""
	def __init__(self, num_models: int = 3):
		if num_models % 2 == 0:
			raise ValueError("Number of models must be odd for voting mechanism")
		self.num_models = num_models
		self.model_manager = MultiModelManager(num_models)
		self.agents = [UnifiedPredictionAgent(self.model_manager, i) for i in range(num_models)]
		self.reflection_agent = ReflectionAgent(self.model_manager)

	def predict(self, market_data: Dict, sentiment_data: Dict, news_summary: str, target: str) -> Dict:
		model_predictions = []
		model_explanations = []
		for i, agent in enumerate(self.agents):
			pred, expl = agent.predict(market_data, sentiment_data, news_summary)
			model_predictions.append(pred)
			model_explanations.append(expl)
			self.model_manager.unload_model(i)
		vote_counts = Counter(model_predictions)
		majority_prediction = vote_counts.most_common(1)[0][0]
		majority_count = vote_counts[majority_prediction]
		correct_predictions = sum(1 for p in model_predictions if p == target)
		incorrect_predictions = self.num_models - correct_predictions
		majority_correct = correct_predictions > (self.num_models // 2)
		if self.model_manager.test_mode:
			return {
				"model_predictions": model_predictions,
				"model_explanations": model_explanations,
				"vote_counts": dict(vote_counts),
				"majority_prediction": majority_prediction,
				"majority_count": majority_count,
				"majority_correct": majority_correct,
				"correct_predictions": correct_predictions,
				"incorrect_predictions": incorrect_predictions,
				"correct_model_indices": [i for i,p in enumerate(model_predictions) if p == target],
				"incorrect_model_indices": [i for i,p in enumerate(model_predictions) if p != target],
				"final_prediction": majority_prediction,
				"final_explanation": model_explanations[model_predictions.index(majority_prediction)],
				"dataset_type": "test",
			}
		if majority_correct:
			return {
				"model_predictions": model_predictions,
				"model_explanations": model_explanations,
				"vote_counts": dict(vote_counts),
				"majority_prediction": majority_prediction,
				"majority_count": majority_count,
				"majority_correct": majority_correct,
				"correct_predictions": correct_predictions,
				"incorrect_predictions": incorrect_predictions,
				"correct_model_indices": [i for i,p in enumerate(model_predictions) if p == target],
				"incorrect_model_indices": [i for i,p in enumerate(model_predictions) if p != target],
				"final_prediction": majority_prediction,
				"final_explanation": model_explanations[model_predictions.index(majority_prediction)],
				"dataset_type": "supervised",
			}
		incorrect_model_indices = [i for i,p in enumerate(model_predictions) if p != target]
		correct_model_indices = [i for i,p in enumerate(model_predictions) if p == target]
		reflection_results = []
		dpo_pairs = []
		for idx in incorrect_model_indices:
			for rtype in ["solution","composite"]:
				try:
					ref = self.reflection_agent.reflect(
						original_prediction=model_predictions[idx],
						original_explanation=model_explanations[idx],
						correct_prediction=target,
						correct_explanation=(model_explanations[correct_model_indices[0]] if correct_model_indices else "Correct prediction based on market analysis"),
						reflection_type=rtype,
						market_data=market_data,
						sentiment_data=sentiment_data,
						news_summary=news_summary,
					)
					reflection_results.append({"model_index": idx, "reflection_type": rtype, **ref})
					dpo_pairs.append({
						"prompt": "Analyze Bitcoin and Gold market data to predict Bitcoin's price movement.",
						"chosen": (model_explanations[correct_model_indices[0]] if correct_model_indices else "Correct prediction based on market analysis"),
						"rejected": ref["reflection_explanation"],
						"preferred": "chosen",
						"reflection_type": rtype,
						"model_index": idx,
						"original_prediction": model_predictions[idx],
						"correct_prediction": target,
					})
				except Exception as e:
					print(f"Error generating {rtype} reflection for Model {idx + 1}: {e}")
		try:
			self.reflection_agent.unload_model()
		except Exception:
			pass
		return {
			"model_predictions": model_predictions,
			"model_explanations": model_explanations,
			"vote_counts": dict(vote_counts),
			"majority_prediction": majority_prediction,
			"majority_count": majority_count,
			"majority_correct": majority_correct,
			"correct_predictions": correct_predictions,
			"incorrect_predictions": incorrect_predictions,
			"incorrect_model_indices": incorrect_model_indices,
			"correct_model_indices": correct_model_indices,
			"reflection_results": reflection_results,
			"dpo_pairs": dpo_pairs,
			"final_prediction": majority_prediction,
			"final_explanation": model_explanations[model_predictions.index(majority_prediction)],
			"dataset_type": "dpo",
		}

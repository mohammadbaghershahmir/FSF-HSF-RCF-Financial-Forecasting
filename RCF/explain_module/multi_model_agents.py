from typing import Dict, Tuple
from utils.llm import FastChatLLM
from utils.enhanced_prompts import ENHANCED_CANDLE_EXAMPLES
from utils.enhanced_prompts import PROMPT_TECHNICAL_STRUCTURE, PROMPT_SENTIMENT_CONTRARIAN, PROMPT_NEWS_MACRO
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
				# Avoid accelerate offload conflicts by not using device_map='auto'
				import torch as _torch
				device = _torch.device("cuda") if _torch.cuda.is_available() else _torch.device("cpu")
				try:
					model = AutoModelForCausalLM.from_pretrained(
						self.dpo_model_dir,
						device_map=None,
						torch_dtype=_torch.float16 if device.type == "cuda" else _torch.float32,
						low_cpu_mem_usage=False,
					)
					model.to(device)
				except Exception as _e:
					print(f"⚠️ Failed to load DPO model on {device}: {_e} → falling back to CPU float32")
					model = AutoModelForCausalLM.from_pretrained(
						self.dpo_model_dir,
						device_map=None,
						torch_dtype=_torch.float32,
						low_cpu_mem_usage=False,
					)
					device = _torch.device("cpu")
					tmodel = model.to(device)
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

	def _score_label(self, base_prompt: str, label_prefix: str, option: str, model, tokenizer) -> float:
		"""Compute average log-probability of `option` tokens conditioned on (base_prompt + label_prefix).
		Evaluation is done token-by-token to avoid slicing errors and bias. Returns de-biased score by subtracting a baseline prior measured on a neutral prefix.
		"""
		def avg_logprob(prefix_text: str, opt_text: str) -> float:
			model.eval()
			device = next(model.parameters()).device
			with torch.no_grad():
				prefix_ids = tokenizer(prefix_text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
				option_ids = tokenizer(opt_text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
				input_ids = torch.cat([prefix_ids, option_ids], dim=1)
				attn_mask = torch.ones_like(input_ids, device=device)
				outputs = model(input_ids, attention_mask=attn_mask)
				logits = outputs.logits
				log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
				target_ids = input_ids[:, 1:]
				prefix_len = prefix_ids.shape[1]
				opt_len = option_ids.shape[1]
				start = prefix_len - 1 if prefix_len > 0 else 0
				end = start + opt_len
				end = min(end, target_ids.shape[1])
				if end <= start:
					return float('-inf')
				sel_lp = log_probs[:, start:end, :].gather(-1, target_ids[:, start:end].unsqueeze(-1)).squeeze(-1)
				return sel_lp.mean().item()
		# Data-conditioned score
		prefix_text = base_prompt + "\n" + label_prefix
		cond = avg_logprob(prefix_text, option)
		# Baseline prior score with neutral prefix (removes inherent token bias)
		neutral_prefix = "Instruction: choose a label.\n" + label_prefix
		prior = avg_logprob(neutral_prefix, option)
		return cond - prior

	def predict(self, market_data: Dict, sentiment_data: Dict, news_summary: str) -> Tuple[str, str]:
		model_data = self.model_manager.load_model(self.model_index)
		if self.model_manager.test_mode:
			# Add variety to break deterministic patterns
			import random
			temp_variations = [0.7, 0.9, 1.1, 1.3]
			top_p_variations = [0.85, 0.9, 0.95]
			top_k_variations = [30, 40, 50]
			
			random_temp = temp_variations[random.randint(0, len(temp_variations)-1)]
			random_top_p = top_p_variations[random.randint(0, len(top_p_variations)-1)]
			random_top_k = top_k_variations[random.randint(0, len(top_k_variations)-1)]
			
			self.llm = FastChatLLM(
				model=model_data["model"],
				tokenizer=model_data["tokenizer"],
				do_sample=True,
				temperature=random_temp,
				top_p=random_top_p,
				top_k=random_top_k,
				repetition_penalty=1.1,
				max_new_tokens=256,
				use_raw_prompt=True
			)
		else:
			# تنوع‌سازی تنظیمات دیکدینگ برای جلوگیری از همگرایی خروجی‌ها
			per_model_temperature = [0.7, 0.9, 1.1]
			per_model_top_p = [0.9, 0.92, 0.95]
			temp = per_model_temperature[self.model_index % len(per_model_temperature)]
			top_p = per_model_top_p[self.model_index % len(per_model_top_p)]
			self.llm = FastChatLLM(
				model=model_data["model"],
				tokenizer=model_data["tokenizer"],
				do_sample=True,
				temperature=temp,
				top_p=top_p,
				top_k=50,
				repetition_penalty=1.1,
				max_new_tokens=768
			)
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

		# Build prompt
		# Select prompt template (training: per-model; test: combined best-of-three)
		if self.model_manager.test_mode:
			# Add randomization to break deterministic patterns
			import random
			random_seed = random.randint(1, 1000)
			random_phrases = [
				'Analyze the data objectively', 'Consider all factors carefully', 
				'Evaluate the market conditions', 'Assess the sentiment indicators',
				'Review the price movements'
			]
			random_phrase = random_phrases[random_seed % len(random_phrases)]
			
			combined_prompt = ("""
			You are a financial analyst specializing in cryptocurrency markets. {random_phrase}.
			
			Task: Analyze the provided Bitcoin market data and sentiment information to predict whether Bitcoin's price will move up (Positive) or down (Negative) in the next period.
			
			Instructions:
			- Examine the Bitcoin price change, trading volume, and market sentiment data
			- Consider both positive and negative indicators equally
			- Look for conflicting signals and weigh them appropriately
			- Make an objective prediction based on the data
			
			Required Output Format:
			Bitcoin Price Movement: [Positive or Negative]
			
			Explanation: [Write a detailed analysis explaining your prediction. Mention specific data points that support your conclusion. Be objective and consider both bullish and bearish factors.]
			
			Data to analyze:
			{market_data}
			
			Sentiment Analysis:
			{sentiment_data}
			"""
			)

			unified_prompt = (
				combined_prompt
					.replace("{random_phrase}", random_phrase)
					.replace("{market_data}", formatted_market_data)
					.replace("{sentiment_data}", formatted_sentiment_data)
			)
		else:
			# Training: per-model enhanced prompts
			templates = [PROMPT_TECHNICAL_STRUCTURE, PROMPT_SENTIMENT_CONTRARIAN, PROMPT_NEWS_MACRO]
			prompt_template = templates[self.model_index % len(templates)]
			unified_prompt = (
				prompt_template
					.replace("{market_data}", formatted_market_data)
					.replace("{sentiment_data}", formatted_sentiment_data)
					.replace("{news_summary}", formatted_news_summary)
					.replace("{examples}", ENHANCED_CANDLE_EXAMPLES)
			)
		print("\nPROMPT:\n" + unified_prompt)
		
		# Log input length and truncation risk
		tokenizer = model_data["tokenizer"]
		try:
			prompt_tokens = len(tokenizer.encode(unified_prompt)) if hasattr(tokenizer, 'encode') else None
		except Exception:
			prompt_tokens = None
		# Try to infer max context
		model_max_ctx = None
		try:
			model_max_ctx = getattr(getattr(model_data['model'], 'config', None), 'max_position_embeddings', None)
		except Exception:
			model_max_ctx = None
		if not model_max_ctx and hasattr(tokenizer, 'model_max_length'):
			model_max_ctx = tokenizer.model_max_length
		if not model_max_ctx:
			model_max_ctx = 4096
		gen_budget = 512
		safety_margin = 64
		allowed_prompt_tokens = max(0, model_max_ctx - gen_budget - safety_margin)
		print(f"\n📊 INPUT LENGTH DEBUG (Model {self.model_index + 1}):")
		print(f"   Prompt tokens: {prompt_tokens if prompt_tokens is not None else 'N/A'}")
		print(f"   Model max context: {model_max_ctx}")
		print(f"   Max new tokens (gen budget): {gen_budget}")
		print(f"   Safety margin: {safety_margin}")
		print(f"   Allowed prompt tokens: {allowed_prompt_tokens}")
		if prompt_tokens is not None and prompt_tokens > allowed_prompt_tokens:
			print("   ⚠️ Prompt likely exceeds safe limit; truncation may occur.")
		# Show head/tail preview of prompt
		head = unified_prompt[:400]
		tail = unified_prompt[-400:] if len(unified_prompt) > 800 else ''
		print("\n🔎 PROMPT PREVIEW (head):\n" + head)
		if tail:
			print("\n🔎 PROMPT PREVIEW (tail):\n" + tail)
		
		if self.model_manager.test_mode:
			# Phase 1: score-based label selection to avoid bias
			label_prefix = "Bitcoin Price Movement: "
			pos_score = self._score_label(unified_prompt, label_prefix, "Positive", model_data["model"], model_data["tokenizer"])
			neg_score = self._score_label(unified_prompt, label_prefix, "Negative", model_data["model"], model_data["tokenizer"])
			label_line = (label_prefix + ("Positive" if pos_score >= neg_score else "Negative"))
			# Phase 2: generate explanation only
			expl_llm = FastChatLLM(
				model=model_data["model"], tokenizer=model_data["tokenizer"],
				do_sample=False, temperature=0.0, top_p=1.0, top_k=1,
				repetition_penalty=1.15, max_new_tokens=180, use_raw_prompt=True
			)
			expl_prompt = unified_prompt + f"\n{label_line}\nExplanation: "
			expl_resp = expl_llm(expl_prompt)
			expl_first = (expl_resp or '').splitlines()[0].strip()
			explanation = ("Explanation: " + expl_first) if not expl_first.lower().startswith('explanation:') else expl_first
			response = label_line + "\n" + explanation
		else:
			response = self.llm(unified_prompt)
		print("\n" + "-"*100)
		print(f"RAW RESPONSE (Model {self.model_index + 1}):\n" + str(response))
		
		# Parse strict two-line output for test mode
		explanation = (response or '').strip()
		if self.model_manager.test_mode and response:
			lines = [l.rstrip() for l in response.splitlines() if l.strip()]
			label_line2 = lines[0] if len(lines) >= 1 else ''
			explain_line = lines[1] if len(lines) >= 2 else ''
			# Normalize and validate label
			low2 = label_line2.strip().lower()
			if low2 == 'bitcoin price movement: positive':
				prediction = 'Positive'
			elif low2 == 'bitcoin price movement: negative':
				prediction = 'Negative'
			else:
				prediction = 'Unknown'
			# Normalize explanation: keep only the second line, enforce prefix
			if explain_line.lower().startswith('explanation:'):
				explanation = explain_line.strip()
			else:
				explanation = ('Explanation: ' + explain_line.strip()) if explain_line else ''
			# Keep exactly two lines
			response = label_line2 + ('\n' + explanation if explanation else '')
		
		# Validate response and parse only prediction label
		if not response or len(response.strip()) < 5:
			print(f"⚠️  WARNING: Model {self.model_index + 1} generated very short response")
			response = ""
		prediction, _ = self._parse_response(response)
		# In test mode, allow one self-repair retry to obtain a valid label
		max_retries = 1 if self.model_manager.test_mode else 2
		retry_count = 0
		while (prediction not in ("Positive", "Negative")) and retry_count < max_retries:
			retry_count += 1
			# Reuse the exact same data-only prompt (news removed) to avoid drift
			repair_prompt = (
				"Your previous response did not follow the required two-line format. Output EXACTLY TWO LINES and NOTHING ELSE.\n"
				"Line 1: Bitcoin Price Movement: Positive   OR   Bitcoin Price Movement: Negative\n"
				"Line 2: Explanation: <one-paragraph, objective and balanced reasoning; no extra lines>\n\n"
				"DATA (use only to determine the two lines; do not echo):\n" + unified_prompt
			)
			repair_response = self.llm(repair_prompt)
			print("\n🛠️ SELF-RETRY RESPONSE:\n" + str(repair_response))
			prediction, _ = self._parse_response(repair_response)
		
		# Post-prediction fallback in TEST mode: fill only when model failed to produce a label
		if self.model_manager.test_mode:
			try:
				btc_pc = float(market_data.get('btc_data', {}).get('price_change', 0.0))
				gold_pc = float(market_data.get('gold_data', {}).get('price_change', 0.0))
				btc_sent = sentiment_data.get('btc_sentiment', {})
				p = float(btc_sent.get('positive', 0.0))
				n = float(btc_sent.get('negative', 0.0))
				# sentiment gap normalized to 0..1 scale if inputs are percent already
				if max(p, n, float(btc_sent.get('neutral',0.0))) > 1.0:
					p /= 100.0; n /= 100.0
				gap = p - n
				# Simple suggestion (no flipping):
				suggested = 'Positive' if (btc_pc > 0 and gap > 0.05) else 'Negative' if (btc_pc < 0 and gap < -0.05) else 'Unknown'
				if prediction not in ("Positive","Negative") and suggested != 'Unknown':
					prediction = suggested
			except Exception:
				pass

		if prediction not in ("Positive", "Negative"):
			print("❌ Model failed to produce required binary prediction.")
			return "Unknown", explanation
		print("\n" + "="*50)
		print(f"PARSED RESULTS (Model {self.model_index + 1}):")
		print("="*50)
		print(f"🎯 Prediction: {prediction}")
		print(f"📝 Explanation (head):\n{explanation[:600]}")
		if len(explanation) > 1200:
			print(f"\n📝 Explanation (tail):\n{explanation[-600:]}")
		print("="*50)
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
				"high": d.get("high", d.get("High", 0.0)),
				"low": d.get("low", d.get("Low", 0.0)),
				"open": d.get("open", d.get("Open", 0.0)),
				"volume": d.get("volume", d.get("Volume", 0.0)),
				"price_change": d.get("price_change", d.get("close_change", d.get("Price_Change", 0.0))),
			}
		btc = extract(market_data, "btc")
		gold = extract(market_data, "gold")

		def fmt_num(x: float) -> str:
			try:
				return f"{float(x):.4f}"
			except Exception:
				return str(x)
		def fmt_vol(x: float) -> str:
			try:
				return f"{int(float(x)):,}"
			except Exception:
				return str(x)
		def fmt_pct(x: float) -> str:
			try:
				return f"{float(x)*100:.2f}%"
			except Exception:
				return str(x)

		formatted = "=== Bitcoin Market Data ===\n"
		formatted += f"Open: {fmt_num(btc['open'])}\n"
		formatted += f"High: {fmt_num(btc['high'])}\n"
		formatted += f"Low: {fmt_num(btc['low'])}\n"
		formatted += f"Volume: {fmt_vol(btc['volume'])}\n"
		formatted += f"Price Change: {fmt_num(btc['price_change'])}\n\n"
		formatted += "=== Gold Market Data ===\n"
		formatted += f"Open: {fmt_num(gold['open'])}\n"
		formatted += f"High: {fmt_num(gold['high'])}\n"
		formatted += f"Low: {fmt_num(gold['low'])}\n"
		formatted += f"Volume: {fmt_vol(gold['volume'])}\n"
		formatted += f"Price Change: {fmt_num(gold['price_change'])}\n\n"
		return formatted

	def _format_sentiment_data(self, sentiment_data: Dict) -> str:
		formatted = "=== Bitcoin Sentiment ===\n"
		btc_sentiment = sentiment_data.get('btc_sentiment', {})
		btc_positive = btc_sentiment.get('positive', 0)
		btc_negative = btc_sentiment.get('negative', 0)
		btc_neutral = btc_sentiment.get('neutral', 0)
		# Scale to percentages if values are in 0-1 range
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
		return formatted

	def _format_news_summary(self, news_summary: str) -> str:
		if not news_summary or news_summary.strip() == "":
			return "No news summary available"
		return news_summary.strip()

	def _parse_response(self, response: str) -> Tuple[str, str]:
		"""Parse only the binary label from model response; do not parse or fabricate explanation."""
		resp = (response or '').strip()
		if not resp:
			return 'Unknown', ''
		low = resp.lower()
		# Exact label only
		if low == 'positive':
			return 'Positive', ''
		if low == 'negative':
			return 'Negative', ''
		# Look for required header line (strict single-label capture only)
		import re as _re
		for line in resp.splitlines():
			l = line.strip()
			m = _re.search(r"^\s*Bitcoin\s*Price\s*Movement\s*:\s*(Positive|Negative)\b\s*$", l, flags=_re.IGNORECASE)
			if m:
				label = m.group(1).strip().lower()
				return ('Positive' if label == 'positive' else 'Negative'), ''
		# Pattern-based fallback (keep strict)
		m2 = _re.search(r'^\s*bitcoin\s*price\s*movement\s*:\s*(positive|negative)\s*$', low, flags=_re.IGNORECASE | _re.MULTILINE)
		if m2:
			label = m2.group(1).strip().lower()
			return ('Positive' if label == 'positive' else 'Negative'), ''
		# Do NOT infer from lone keywords to avoid bias
		return 'Unknown', ''


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
		# Exclude Unknowns from voting (explanations kept as full responses)
		valid_indices = [idx for idx,p in enumerate(model_predictions) if p in ("Positive","Negative")]
		valid_votes = [model_predictions[i] for i in valid_indices]
		vote_counts = Counter(valid_votes) if valid_votes else Counter()
		if vote_counts:
			majority_prediction = vote_counts.most_common(1)[0][0]
			majority_count = vote_counts[majority_prediction]
		else:
			# No valid model outputs → skip this sample
			return {
				"model_predictions": model_predictions,
				"model_explanations": model_explanations,
				"vote_counts": {},
				"majority_prediction": "Unknown",
				"majority_count": 0,
				"majority_correct": False,
				"correct_predictions": 0,
				"incorrect_predictions": self.num_models,
				"correct_model_indices": [],
				"incorrect_model_indices": list(range(self.num_models)),
				"final_prediction": "Unknown",
				"final_explanation": "",
				"dataset_type": "skip",
			}

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
				"final_explanation": model_explanations[model_predictions.index(majority_prediction)] if majority_prediction in model_predictions else (model_explanations[0] if model_explanations else ""),
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
				"final_explanation": model_explanations[model_predictions.index(majority_prediction)] if majority_prediction in model_predictions else (model_explanations[0] if model_explanations else ""),
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
			"final_explanation": model_explanations[model_predictions.index(majority_prediction)] if majority_prediction in model_predictions else (model_explanations[0] if model_explanations else ""),
			"dataset_type": "dpo",
		}

# ===========================
# Candle + Sentiment Prompt
# ===========================

CANDLE_SENTIMENT_INSTRUCTION_ENHANCED = """
You are a Bitcoin market analyst. Analyze the data and predict Bitcoin's next price movement.

CRITICAL: The prediction label MUST be either "Positive" or "Negative". After the label, you MUST also include an Explanation paragraph as instructed.

OUTPUT FORMAT (STRICT):
Bitcoin Price Movement: [Positive/Negative]

Explanation: [Provide a clear analysis in one paragraph of 100-150 words explaining your reasoning based on the data provided.]

Be objective. If the data suggests negative movement, predict "Negative". If it suggests positive movement, predict "Positive".

Market Data:
{market_data}

Sentiment Data:
{sentiment_data}
"""

# ===========================
# News + Sentiment Prompt
# ===========================

NEWS_SUMMARY_INSTRUCTION_ENHANCED = """
You are a Bitcoin market analyst. Analyze the news summary and sentiment to predict Bitcoin's next price movement.

CRITICAL: The prediction label MUST be either "Positive" or "Negative". After the label, you MUST also include an Explanation paragraph as instructed.

OUTPUT FORMAT (STRICT):
Bitcoin Price Movement: [Positive/Negative]

Explanation: [Provide a clear analysis in one paragraph of 100-150 words explaining your reasoning based on the news and sentiment provided.]

Be objective. If the news and sentiment suggest negative movement, predict "Negative". If they suggest positive movement, predict "Positive".

News Summary:
{news_summary}

Sentiment Data:
{sentiment_data}
"""

# ===========================
# Reflection Prompts - Enhanced for Diversity
# ===========================

REFLECTION_RETRY_INSTRUCTION = """
You made an incorrect prediction. Try again with a fresh perspective.

CRITICAL: You MUST predict ONLY "Positive" or "Negative". NO other predictions are allowed.

OUTPUT FORMAT (STRICT):
Bitcoin Price Movement: [Positive/Negative]

Explanation: [Write a different analysis focusing on alternative factors. 100-150 words.]

Your Previous Prediction: {original_prediction}
Your Previous Explanation: {original_explanation}

Correct Prediction: {correct_prediction}
Correct Explanation: {correct_explanation}

Think differently this time.
"""

REFLECTION_KEYWORDS_INSTRUCTION = """
You made an incorrect prediction. Focus on these technical keywords:

- Volume patterns and participation levels
- Price structure and trend analysis  
- Sentiment extremes and market psychology
- Cross-market correlations and macro factors

CRITICAL: You MUST predict ONLY "Positive" or "Negative". NO other predictions are allowed.

OUTPUT FORMAT (STRICT):
Bitcoin Price Movement: [Positive/Negative]

Explanation: [Write a technical analysis focusing on the keywords above. 100-150 words.]

Your Previous Prediction: {original_prediction}
Your Previous Explanation: {original_explanation}

Correct Prediction: {correct_prediction}
Correct Explanation: {correct_explanation}

Focus on technical analysis.
"""

REFLECTION_ADVICE_INSTRUCTION = """
You made an incorrect prediction. Here's advice for better analysis:

1. Don't anchor on initial impressions - look at the complete picture
2. Consider both bullish and bearish factors equally
3. Pay attention to volume and participation levels
4. Analyze sentiment extremes and contrarian signals

CRITICAL: You MUST predict ONLY "Positive" or "Negative". NO other predictions are allowed.

OUTPUT FORMAT (STRICT):
Bitcoin Price Movement: [Positive/Negative]

Explanation: [Write a strategic analysis following the advice above. 100-150 words.]

Your Previous Prediction: {original_prediction}
Your Previous Explanation: {original_explanation}

Correct Prediction: {correct_prediction}
Correct Explanation: {correct_explanation}

Apply the strategic advice above.
"""

REFLECTION_EXPLANATION_INSTRUCTION = """
You made an incorrect prediction. Consider these analytical frameworks:

- Market structure and trend analysis
- Volume patterns and participation levels
- Sentiment extremes and contrarian signals
- Cross-market correlations and macro factors

CRITICAL: You MUST predict ONLY "Positive" or "Negative". NO other predictions are allowed.

OUTPUT FORMAT (STRICT):
Bitcoin Price Movement: [Positive/Negative]

Explanation: [Write an analytical framework-based analysis. 100-150 words.]

Your Previous Prediction: {original_prediction}
Your Previous Explanation: {original_explanation}

Correct Prediction: {correct_prediction}
Correct Explanation: {correct_explanation}

Use the analytical frameworks provided above.
"""

REFLECTION_INSTRUCTIONS_INSTRUCTION = """
You made an incorrect prediction. Follow these step-by-step instructions:

Step 1: Identify the dominant trend and market structure
Step 2: Analyze volume patterns and participation levels
Step 3: Evaluate sentiment distribution and extremes
Step 4: Consider cross-market correlations and macro factors
Step 5: Balance all evidence objectively before concluding

CRITICAL: You MUST predict ONLY "Positive" or "Negative". NO other predictions are allowed.

OUTPUT FORMAT (STRICT):
Bitcoin Price Movement: [Positive/Negative]

Explanation: [Write a step-by-step analysis following the instructions above. 100-150 words.]

Your Previous Prediction: {original_prediction}
Your Previous Explanation: {original_explanation}

Correct Prediction: {correct_prediction}
Correct Explanation: {correct_explanation}

Follow the step-by-step instructions.
"""

REFLECTION_SOLUTION_INSTRUCTION = """
You made an incorrect prediction. Consider the overall market narrative and how different elements interact.

CRITICAL: You MUST predict ONLY "Positive" or "Negative". NO other predictions are allowed.

OUTPUT FORMAT (STRICT):
Bitcoin Price Movement: [Positive/Negative]

Explanation: [Write a holistic analysis focusing on market narrative. 100-150 words.]

Your Previous Prediction: {original_prediction}
Your Previous Explanation: {original_explanation}

Correct Prediction: {correct_prediction}
Correct Explanation: {correct_explanation}

Focus on holistic market narrative.
"""

REFLECTION_COMPOSITE_INSTRUCTION = """
You made an incorrect prediction. Use a comprehensive multi-dimensional approach:

1. Technical Analysis: Examine price structure, volume, and momentum
2. Sentiment Analysis: Consider market psychology and social signals
3. Fundamental Analysis: Evaluate macro environment and regulatory factors
4. Cross-Market Analysis: Assess Bitcoin-Gold correlations and risk flows

CRITICAL: You MUST predict ONLY "Positive" or "Negative". NO other predictions are allowed.

OUTPUT FORMAT (STRICT):
Bitcoin Price Movement: [Positive/Negative]

Explanation: [Write a comprehensive multi-dimensional analysis. 100-150 words.]

Your Previous Prediction: {original_prediction}
Your Previous Explanation: {original_explanation}

Correct Prediction: {correct_prediction}
Correct Explanation: {correct_explanation}

Cover all analytical dimensions.
"""

# ===========================
# Reflection Types Dictionary
# ===========================

REFLECTION_TYPES = {
    "retry": REFLECTION_RETRY_INSTRUCTION,
    "keywords": REFLECTION_KEYWORDS_INSTRUCTION,
    "advice": REFLECTION_ADVICE_INSTRUCTION,
    "explanation": REFLECTION_EXPLANATION_INSTRUCTION,
    "instructions": REFLECTION_INSTRUCTIONS_INSTRUCTION,
    "solution": REFLECTION_SOLUTION_INSTRUCTION,
    "composite": REFLECTION_COMPOSITE_INSTRUCTION
}

# ===========================
# Examples
# ===========================

ENHANCED_CANDLE_EXAMPLES = """
Example 1:
Market Data: Bitcoin High: 0.1107, Low: 0.0733, Volume: 78,070
Sentiment: Bitcoin Positive: 65%, Negative: 25%, Neutral: 10%
Analysis: Bitcoin Price Movement: Positive
Explanation: The market shows strong bullish momentum with Bitcoin reaching new highs at $45,000 and maintaining support above $42,000. High volume confirms strong participation and buying interest. The sentiment distribution shows 65% positive sentiment, indicating strong market confidence. The technical structure suggests continuation of the uptrend with higher highs and higher lows pattern. Cross-market analysis reveals positive correlation with risk assets, supporting the bullish outlook.

Example 2:
Market Data: Bitcoin High: 0.0533, Low: -0.0860, Volume: 47,480
Sentiment: Bitcoin Positive: 30%, Negative: 60%, Neutral: 10%
Analysis: Bitcoin Price Movement: Negative
Explanation: The market shows bearish momentum with Bitcoin failing to hold above $38,000 and testing lower support levels. Low volume indicates weak participation and lack of buying interest. The sentiment distribution shows 60% negative sentiment, reflecting market concerns and risk aversion. The technical structure suggests potential breakdown with lower highs and lower lows pattern. Cross-market analysis reveals negative correlation with risk assets, supporting the bearish outlook.
"""

ENHANCED_NEWS_EXAMPLES = """
Example 1:
News: Major institutional adoption announcement, positive regulatory developments
Sentiment: Bitcoin Positive: 70%, Negative: 20%, Neutral: 10%
Analysis: Bitcoin Price Movement: Positive
Explanation: The news flow is overwhelmingly positive with major institutional adoption announcements driving market confidence. Positive regulatory developments provide clarity and reduce uncertainty, supporting institutional participation. The sentiment distribution shows 70% positive sentiment, indicating strong market optimism. Cross-market analysis reveals positive correlation with risk assets, supporting the bullish outlook. The combination of fundamental catalysts and positive sentiment creates a strong case for continued upward momentum.

Example 2:
News: Regulatory concerns, negative market sentiment, risk-off environment
Sentiment: Bitcoin Positive: 25%, Negative: 65%, Neutral: 10%
Analysis: Bitcoin Price Movement: Negative
Explanation: The news flow is predominantly negative with regulatory concerns creating uncertainty and risk-off sentiment dominating the market. The sentiment distribution shows 65% negative sentiment, reflecting market concerns and risk aversion. Cross-market analysis reveals negative correlation with risk assets, supporting the bearish outlook. The combination of negative catalysts and pessimistic sentiment creates a strong case for continued downward pressure.
"""

# ===========================
# Test Mode Prompts - Short and Focused
# ===========================

CANDLE_SENTIMENT_INSTRUCTION_TEST_SHORT = """
Analyze Bitcoin market data and predict price movement.

CRITICAL: The prediction label MUST be either "Positive" or "Negative". You MUST also include an Explanation paragraph as instructed.

Decision Rubric (balanced):
- Favor Positive if BTC technicals lean up (higher highs/lows) OR BTC positive sentiment > negative by a clear margin.
- Favor Negative if BTC technicals lean down OR BTC negative sentiment > positive by a clear margin.
- Use Gold as risk-on/off context: strong Gold risk-on (bearish Gold) slightly supports Positive for BTC; strong Gold risk-off (bullish Gold) slightly supports Negative for BTC.
- If signals conflict, state both, then pick the stronger side decisively. Do NOT default to Negative.

OUTPUT FORMAT:
Bitcoin Price Movement: [Positive/Negative]

Explanation: [Provide a clear analysis in one paragraph of 80-140 words explaining your reasoning based on the data provided.]

Examples:
{examples}

Market Data:
{market_data}

Sentiment Data:
{sentiment_data}

Provide your prediction and explanation.
"""
NEWS_SUMMARY_INSTRUCTION_TEST_SHORT = """
Analyze Bitcoin news and sentiment to predict price movement.

CRITICAL: The prediction label MUST be either "Positive" or "Negative". You MUST also include an Explanation paragraph as instructed.

Decision Rubric (balanced):
- Favor Positive if news catalysts are constructive (adoption, regulation clarity) and sentiment skews positive.
- Favor Negative if news is risk-off (regulatory crackdowns, macro stress) and sentiment skews negative.
- When mixed, explain both sides briefly and choose the stronger. Do NOT default to Negative.

OUTPUT FORMAT:
Bitcoin Price Movement: [Positive/Negative]

Explanation: [Brief analysis in 70-120 words based on news and sentiment.]

Examples:
{examples}

News Summary:
{news_summary}

Sentiment Data:
{sentiment_data}

Provide your prediction and explanation.
"""

# ===========================
# Diverse Prompt Templates for Three Base Models
# ===========================

PROMPT_TECHNICAL_STRUCTURE = """
You are a Bitcoin market analyst focused on market structure and order-flow. Prioritize structural signals over headlines. Weigh:
- Trend regime (higher-highs/lows vs lower-highs/lows)
- Key levels (break/retest), momentum divergences
- Volume and participation (expansion/contraction)
- BTC–Gold cross-market alignment

CRITICAL: The prediction label MUST be either "Positive" or "Negative". After the label, you MUST also include an Explanation paragraph as instructed.

OUTPUT FORMAT (STRICT):
Bitcoin Price Movement: [Positive/Negative]

Explanation: [A single cohesive paragraph (100–150 words) summarizing structure, momentum, levels, volume, and BTC–Gold interaction. Avoid bullet points and hedging.]

Inputs:
- Market Data (BTC/Gold OHLC/Volume)
- Twitter Sentiment (BTC/Gold distributions)
- News Summary (if provided)

Market Data:
{market_data}

Sentiment Data:
{sentiment_data}

News Summary:
{news_summary}
"""

PROMPT_SENTIMENT_CONTRARIAN = """
You are a Bitcoin market psychologist. Prioritize sentiment distribution and contrarian signals. Weigh:
- Skew and extremes in BTC/Gold sentiment (herding vs dispersion)
- Sentiment–price divergences (euphoria on lower highs; fear on higher lows)
- Participation asymmetry and positioning risk
- Macro risk-on/off tone inferred from BTC–Gold sentiment contrast

CRITICAL: The prediction label MUST be either "Positive" or "Negative". After the label, you MUST also include an Explanation paragraph as instructed.

OUTPUT FORMAT (STRICT):
Bitcoin Price Movement: [Positive/Negative]

Explanation: [100–150 words explaining how sentiment skew, divergences, and risk regime drive the call. Be decisive; no lists.]

Market Data:
{market_data}

Sentiment Data:
{sentiment_data}

News Summary:
{news_summary}
"""

PROMPT_NEWS_MACRO = """
You are a macro-driven Bitcoin analyst. Prioritize news flow and cross-asset context. Weigh:
- Regulatory/policy headlines and institutional flows
- Macro risk regime (risk-on/off) and liquidity impulse
- BTC–Gold correlation shifts (hedge vs risk asset)
- Whether news aligns or conflicts with price and sentiment

CRITICAL: The prediction label MUST be either "Positive" or "Negative". After the label, you MUST also include an Explanation paragraph as instructed.

OUTPUT FORMAT (STRICT):
Bitcoin Price Movement: [Positive/Negative]

Explanation: [100–150 words linking the news narrative to price action and sentiment confirmation/contradiction, ending with a clear directional rationale.]

News Summary:
{news_summary}

Market Data:
{market_data}

Sentiment Data:
{sentiment_data}
"""

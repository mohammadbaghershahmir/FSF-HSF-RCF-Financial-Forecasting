SUMMARIZE_INSTRUCTION = """You are given a collection of tweets and news items. Summarize all essential facts and signals about the {market} market.
Base all points strictly on the provided items; avoid speculation. If evidence is missing for a section, say 'None found'.

Your summary must highlight:
- Market signals and short-term indicators
- Key macroeconomic news and trends
- Activities of major players and institutions
- Sentiment analysis (positive/negative/neutral)
- Regulatory updates or policy shifts
- At least one bullish factor AND one bearish factor if available

Additionally, analyze potential correlations or spillover effects from other markets, especially {related_market}. Always mention both supportive and opposing signals before concluding.

Here are some reference examples:
{examples}
(END OF EXAMPLES)

Tweets and News:
{tweets}

Key Facts:"""

PREDICT_INSTRUCTION = """You are tasked with analyzing the relationship between Bitcoin (main market) and Gold (secondary market), with primary emphasis on Bitcoin’s price movement.
Use only the provided data.

STRICT OUTPUT FORMAT (REQUIRED):
Bitcoin Price Movement: [Positive/Negative]

Explanation: [Write a cohesive paragraph]

MANDATORY RULES:
- You MUST output ONLY "Positive" or "Negative". Do NOT output Neutral, Unknown, or any third option.
- Base your decision strictly on the provided inputs.
- Mention at least one bullish and one bearish factor before deciding.
- Explain briefly how Gold conditions influence Bitcoin flows.

Input Data:
{market_data}

Sentiment Data:
{sentiment_data}
"""

PREDICT_REFLECT_INSTRUCTION = """You are reviewing the accuracy of a previous Bitcoin price prediction by comparing it against the actual outcome.
Be concise and evidence-driven.

Previous Prediction:
{prediction}

Actual Outcome:
{outcome}

In your reflection, address the following:
1. Were Bitcoin-specific indicators (both bullish and bearish) properly analyzed?
2. How accurate was the assessment of Gold’s influence on Bitcoin?
3. What lessons can be applied to improve future Bitcoin predictions?

Now provide a corrected prediction in this exact format (binary only):

Bitcoin Price Movement: [Positive/Negative]

Explanation: [Write a cohesive paragraph justifying the corrected binary label]
"""

REFLECTION_HEADER = "You have attempted this Bitcoin price prediction task before and failed. Below are reflections that suggest strategies to improve prediction accuracy. Use them to adjust and strengthen your reasoning approach.\n"

REFLECT_INSTRUCTION = """You are an advanced market reasoning agent capable of learning from past errors.
You will review a previous attempt at predicting Bitcoin’s price movement that turned out incorrect.

In a few sentences, reflect on:
1. Whether Bitcoin-specific indicators (both bullish and bearish) were properly analyzed
2. Whether Gold’s influence on Bitcoin was correctly assessed
3. Whether important signals were overlooked or undervalued

Then, propose a clear, concise strategy to improve accuracy in future predictions by balancing bullish and bearish factors more carefully.

Previous Trial:
{scratchpad}

Reflection:"""

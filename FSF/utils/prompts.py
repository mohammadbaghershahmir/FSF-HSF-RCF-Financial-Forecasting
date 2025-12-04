SUMMARIZE_INSTRUCTION = """You are given a list of tweets and news items. Summarize all essential facts and indicators about the {market} market.
Base your summary only on the provided content. Do not speculate or invent facts. If no evidence is available for a section, state "None found".

Your summary must cover:
- Market signals and short-term indicators
- Macroeconomic news and trends
- Activities of major players and institutions
- Sentiment analysis (positive/negative/neutral)
- Regulatory updates or policy changes
- Correlations or spillover effects from other markets, especially {related_market}

Here are some reference examples:
{examples}
(END OF EXAMPLES)

Tweets and News:
{tweets}

Facts:"""

PREDICT_INSTRUCTION = """You are tasked with analyzing the relationship between Bitcoin (main market) and Gold (secondary market), with primary emphasis on Bitcoin's price movement.
Use only the provided data. If information is missing or contradictory, lower the confidence level and explain the uncertainty in the rationale.

For Bitcoin (Main Market), consider:
1. Technical indicators and price action patterns
2. Market sentiment and social media signals
3. On-chain metrics and network health
4. Institutional activity and whale movements

For Cross-Market Analysis, consider:
1. Gold's role as a competing safe-haven asset
2. Shared macroeconomic factors influencing both Bitcoin and Gold
3. Risk-on vs. Risk-off market sentiment dynamics
4. Historical correlation patterns between BTC and Gold
5. The impact of volatility on both assets

Market Data:
{market_data}

Sentiment Analysis:
{sentiment_data}

Give your response in this exact format:
(1) Bitcoin Price Movement: [Positive/Negative]
(2) Confidence Probability: [0.00-1.00] (e.g., 0.75 for 75% confidence)
(3) Primary Bitcoin Factors:
    - List 2–3 key Bitcoin-specific factors grounded in the data
(4) Cross-Market Impact:
    - Explain clearly how Gold's conditions are affecting Bitcoin
(5) Rationale:
    - Evidence-based explanation of Bitcoin's outlook, noting uncertainties or missing data

IMPORTANT: The confidence probability must be a number between 0.00 and 1.00. 
- 0.00-0.30: Low confidence
- 0.31-0.70: Medium confidence  
- 0.71-1.00: High confidence"""

PREDICT_REFLECT_INSTRUCTION = """Based on the previous prediction and actual outcome, reflect on the accuracy of the Bitcoin price prediction and identify areas for improvement.

Previous Prediction:
{prediction}

Actual Outcome:
{outcome}

Consider:
1. Were the Bitcoin-specific indicators interpreted correctly?
2. How accurate was the assessment of Gold's influence on Bitcoin?
3. Did you assign the right weight to different market drivers?
4. What lessons can be applied to improve future Bitcoin predictions?

Reflection:"""

REFLECTION_HEADER = "You have attempted this Bitcoin price prediction task before and failed. The following reflection(s) provide strategies to improve your prediction accuracy. Use them to adjust your approach.\n"

REFLECT_INSTRUCTION = """You are an advanced market reasoning agent capable of learning from past errors.
You will be shown a previous attempt at predicting Bitcoin's price movement that turned out incorrect.

In a few sentences, reflect on:
1. Whether Bitcoin-specific indicators were properly analyzed
2. Whether Gold's influence on Bitcoin was correctly assessed
3. Whether important market signals were overlooked or undervalued

Then, propose a clear, concise strategy to improve future predictions.

Previous trial:
{scratchpad}

Reflection:"""

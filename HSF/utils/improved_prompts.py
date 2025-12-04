# Improved prompts for better model performance

CANDLE_SENTIMENT_INSTRUCTION_IMPROVED = """You are a financial analyst specializing in cryptocurrency and precious metals markets. Your task is to analyze Bitcoin and Gold market data to predict Bitcoin's price movement.

IMPORTANT: You must output ONLY:
1. "Bitcoin Price Movement: [Positive/Negative]"
2. "Explanation: [Your detailed analysis]"

CRITICAL CONSTRAINTS:
- Be objective and consider both bullish and bearish factors equally
- Focus on concrete data points, not general statements
- Consider the relationship between Bitcoin and Gold markets
- Analyze volume, price patterns, and sentiment correlations
- Provide specific, actionable insights

{examples}

Market Data:
{market_data}

Sentiment Data:
{sentiment_data}

Based on the above data, predict Bitcoin's price movement:"""

NEWS_SUMMARY_INSTRUCTION_IMPROVED = """You are a financial news analyst specializing in cryptocurrency markets. Your task is to analyze daily news summaries and sentiment data to predict Bitcoin's price movement.

IMPORTANT: You must output ONLY:
1. "Bitcoin Price Movement: [Positive/Negative]"
2. "Explanation: [Your detailed analysis]"

CRITICAL CONSTRAINTS:
- Focus on news impact, not technical indicators
- Consider regulatory, adoption, and market sentiment news
- Analyze how news affects investor behavior
- Consider the relationship between Bitcoin and Gold news
- Provide specific news-based reasoning

{examples}

News Summary:
{news_summary}

Sentiment Data:
{sentiment_data}

Based on the above news and sentiment, predict Bitcoin's price movement:"""

REFLECTION_INSTRUCTION_IMPROVED = """You made an incorrect prediction. Analyze why you were wrong and provide a corrected prediction based on the correct model's explanation.

IMPORTANT: You must output ONLY:
1. "Bitcoin Price Movement: [Positive/Negative]"
2. "Explanation: [Your corrected analysis]"

REFLECTION PROCESS:
1. Acknowledge your previous mistake
2. Identify the key factors you overlooked
3. Incorporate insights from the correct model
4. Provide a new, corrected prediction with clear reasoning

Your Previous Prediction: {original_prediction}
Your Previous Explanation: {original_explanation}

Correct Model's Prediction: {correct_prediction}
Correct Model's Explanation: {correct_explanation}

Now provide your corrected prediction:"""

# Enhanced few-shot examples
IMPROVED_CANDLE_EXAMPLES = """
EXAMPLE 1:
Market: BTC High: 0.075, Low: 0.072, Volume: 0.008
Sentiment: BTC Positive: 0.8%, Negative: 0.1%, Neutral: 0.1%
Gold Positive: 0.2%, Negative: 0.1%, Neutral: 0.7%
Bitcoin Price Movement: Positive
Explanation: Strong bullish pattern with higher high (0.075) and higher low (0.072), indicating upward momentum. High volume (0.008) confirms buying pressure. BTC sentiment heavily positive (0.8% vs 0.1% negative) shows strong market confidence. Gold's neutral sentiment (0.7%) suggests no safe-haven competition, allowing Bitcoin to capture risk-on flows.

EXAMPLE 2:
Market: BTC High: 0.068, Low: 0.065, Volume: 0.003
Sentiment: BTC Positive: 0.1%, Negative: 0.7%, Neutral: 0.2%
Gold Positive: 0.6%, Negative: 0.1%, Neutral: 0.3%
Bitcoin Price Movement: Negative
Explanation: Bearish pattern with lower high (0.068) and lower low (0.065), showing downward momentum. Low volume (0.003) indicates weak buying interest. BTC sentiment heavily negative (0.7% vs 0.1% positive) reflects market pessimism. Gold's positive sentiment (0.6%) suggests safe-haven flows, diverting capital from Bitcoin.

EXAMPLE 3:
Market: BTC High: 0.071, Low: 0.070, Volume: 0.005
Sentiment: BTC Positive: 0.3%, Negative: 0.2%, Neutral: 0.5%
Gold Positive: 0.2%, Negative: 0.3%, Neutral: 0.5%
Bitcoin Price Movement: Negative
Explanation: Sideways pattern with narrow range (0.071-0.070) and moderate volume (0.005) suggests consolidation. Balanced BTC sentiment (0.3% positive vs 0.2% negative) indicates uncertainty. Gold's neutral sentiment (0.5%) shows no clear safe-haven preference. In uncertain markets, Bitcoin often faces selling pressure as investors wait for clearer signals.
"""

IMPROVED_NEWS_EXAMPLES = """
EXAMPLE 1:
News: "Bitcoin ETF approval expected this week, major banks announce crypto custody services"
Sentiment: BTC Positive: 0.9%, Negative: 0.1%, Neutral: 0.0%
Gold Positive: 0.2%, Negative: 0.1%, Neutral: 0.7%
Bitcoin Price Movement: Positive
Explanation: Major regulatory breakthrough with ETF approval creates institutional access. Bank custody services increase institutional confidence. BTC sentiment extremely positive (0.9%) shows strong market optimism. Gold's neutral sentiment (0.7%) indicates no safe-haven competition. These developments typically drive significant Bitcoin price appreciation.

EXAMPLE 2:
News: "Regulatory crackdown on crypto exchanges, SEC issues new compliance requirements"
Sentiment: BTC Positive: 0.1%, Negative: 0.8%, Neutral: 0.1%
Gold Positive: 0.7%, Negative: 0.1%, Neutral: 0.2%
Bitcoin Price Movement: Negative
Explanation: Regulatory uncertainty creates market fear and selling pressure. BTC sentiment heavily negative (0.8%) reflects regulatory concerns. Gold's positive sentiment (0.7%) shows safe-haven flows as investors seek stability. Regulatory crackdowns typically cause Bitcoin price declines until clarity emerges.

EXAMPLE 3:
News: "Mixed signals: Some positive adoption news but regulatory uncertainty remains"
Sentiment: BTC Positive: 0.4%, Negative: 0.3%, Neutral: 0.3%
Gold Positive: 0.3%, Negative: 0.2%, Neutral: 0.5%
Bitcoin Price Movement: Negative
Explanation: Mixed news creates market uncertainty. Balanced BTC sentiment (0.4% positive vs 0.3% negative) shows indecision. Gold's neutral sentiment (0.5%) indicates no clear safe-haven preference. In uncertain environments, Bitcoin often faces selling pressure as investors wait for clearer signals before committing capital.
""" 
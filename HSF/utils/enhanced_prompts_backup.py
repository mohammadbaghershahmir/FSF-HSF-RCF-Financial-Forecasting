# ===========================
# Candle + Sentiment Prompt
# ===========================

CANDLE_SENTIMENT_INSTRUCTION_ENHANCED = """
You are a senior financial analyst specializing in cryptocurrency and precious metals markets. 
Your task is to analyze Bitcoin and Gold market data and predict Bitcoin's next price movement.

STRICT OUTPUT FORMAT (REQUIRED):
Bitcoin Price Movement: [Positive/Negative]

Explanation: [One cohesive, well-structured paragraph of 120–180 words]

HARD CONSTRAINTS:
- Only two valid labels are allowed: Positive or Negative. Never output Neutral, Unknown, or any third option.
- Put the label on the very first line exactly as specified.
- No bullet points; write a connected narrative.
- Ground every claim in the provided data; avoid making up numbers or events.

DECISION RUBRIC (APPLY ALL):
- Structure: Lower highs/distribution → Negative; higher highs/accumulation → Positive.
- Participation: Rising volume confirms direction; falling volume weakens conviction.
- Sentiment: BTC positive > negative tilts Positive; BTC negative > positive tilts Negative.
- Cross-Market: Strong Gold risk-off (positive Gold sentiment) biases toward Negative for BTC.
- If signals conflict, state both sides briefly, then choose the stronger side and commit.
- IMPORTANT: If BTC technicals are weak AND sentiment is not clearly positive, prefer Negative.

{examples}

Market Data:
{market_data}

Sentiment Data:
{sentiment_data}

Output the two required lines only (label line and Explanation). Do not add extra text.
"""

# ===========================
# News + Sentiment Prompt
# ===========================

NEWS_SUMMARY_INSTRUCTION_ENHANCED = """
You are a senior financial news analyst specializing in cryptocurrency markets. 
Your task is to analyze daily news summaries and sentiment data to predict Bitcoin's next price movement.

STRICT OUTPUT FORMAT (REQUIRED):
Bitcoin Price Movement: [Positive/Negative]

Explanation: [One cohesive, well-structured paragraph of 120–180 words]

HARD CONSTRAINTS:
- Only two valid labels are allowed: Positive or Negative. Never output Neutral, Unknown, or any third option.
- Put the label on the very first line exactly as specified.
- No bullet points; write a continuous narrative.
- Refer only to provided news and sentiment; do not invent facts.

DECISION RUBRIC (APPLY ALL):
- Positive drivers: adoption, regulatory clarity, institutional inflows, constructive sentiment → bias Positive.
- Negative drivers: bans/hacks/enforcement, liquidity stress, risk-off rotation, negative sentiment → bias Negative.
- Cross-Market: Strong Gold safe-haven demand adds headwinds to BTC (bias Negative).
- If mixed, weigh magnitude/recency and commit to one label.
- IMPORTANT: In the absence of strong positive catalysts, default to Negative (risk control).

{examples}

News Summary:
{news_summary}

Sentiment Data:
{sentiment_data}

Output the two required lines only (label line and Explanation). Do not add extra text.
"""

# ===========================
# Short Test Prompt (Candle)
# ===========================

CANDLE_SENTIMENT_INSTRUCTION_TEST_SHORT = """
You are an expert Bitcoin price prediction analyst. Your task is to predict Bitcoin's next price movement based on market data and sentiment analysis.

STRICT OUTPUT FORMAT (REQUIRED):
Bitcoin Price Movement: [Positive/Negative]

Explanation: [4–6 sentences; data-driven, coherent prose]

CRITICAL RULES:
- Output ONLY Positive or Negative. Never Neutral/Unknown.
- If Bitcoin structure weakens and sentiment is not clearly positive → prefer Negative.
- If price structure strengthens with participation and sentiment is net positive → prefer Positive.
- Use Gold sentiment as a tie-breaker (risk-off → Negative bias for BTC).

Data:
{market_data}

Sentiment:
{sentiment_data}

Output only the two required lines.
"""

# ===========================
# Short Test Prompt (News)
# ===========================

NEWS_SUMMARY_INSTRUCTION_TEST_SHORT = """
You are an expert Bitcoin market analyst specializing in news-driven price predictions. Your task is to predict Bitcoin's next price movement based on news summaries and sentiment data.

STRICT OUTPUT FORMAT (REQUIRED):
Bitcoin Price Movement: [Positive/Negative]

Explanation: [4–6 sentences; identify specific drivers and justify the binary label]

CRITICAL RULES:
- Output ONLY Positive or Negative. Never Neutral/Unknown.
- Negative drivers (bans, hacks, enforcement, outflows) → Negative.
- Positive drivers (adoption, partnerships, regulatory clarity, inflows) → Positive.
- Cross-market sentiment may confirm the direction.
- IMPORTANT: If signals are ambiguous, prefer Negative to avoid optimism bias.

News:
{news_summary}

Sentiment:
{sentiment_data}

Output only the two required lines.
"""

# ===========================
# Reflection Prompt (kept compatible)
# ===========================

REFLECTION_INSTRUCTION_ENHANCED = """
You made an incorrect prediction. Analyze why you were wrong and provide a corrected prediction.

STRICT OUTPUT FORMAT (REQUIRED):
Bitcoin Price Movement: [Positive/Negative]

Explanation: [One cohesive, well-structured paragraph of 120–180 words; explain the correction]

REFLECTION PROCESS:
- Acknowledge the mistake; cite both bullish and bearish signals previously misweighted.
- Integrate insights from the correct model’s explanation.
- Rebalance the signals and commit to a single binary label.
- Never use Neutral/Unknown.

Your Previous Prediction: {original_prediction}
Your Previous Explanation: {original_explanation}

Correct Model's Prediction: {correct_prediction}
Correct Model's Explanation: {correct_explanation}

Output only the two required lines.
"""

REFLECTION_CRITIQUE_INSTRUCTION = """
Perform a self-reflection on your previous answer. 
Identify errors and produce targeted advice to avoid them. 
Write in clear prose (not fragments).

Sections required:
[Error Identification] – specific mistakes in prior reasoning
[Missed Bullish Signals] – bullish data ignored
[Missed Bearish Signals] – bearish data ignored
[Correction Advice] – rules for balancing signals next time

Context:
- Previous Prediction: {original_prediction}
- Previous Explanation: {original_explanation}
- Correct Model's Prediction: {correct_prediction}
- Correct Model's Explanation: {correct_explanation}
"""

REFLECTION_REVISION_INSTRUCTION = """
Using the critique below, rewrite your answer.

STRICT OUTPUT SCHEMA (MANDATORY):
<answer>
<label>Positive</label> OR <label>Negative</label>
<explanation>[one cohesive essay of 150–200 words; explicitly reference the critique]</explanation>
</answer>

Rules:
- Only Positive or Negative allowed
- Explanation must show how correction advice changes your reasoning
- No bullet points, no fragments — narrative only

[Critique]
{critique}

[For Reference]
- Previous Prediction: {original_prediction}
- Previous Explanation: {original_explanation}
- Correct Model's Prediction: {correct_prediction}
- Correct Model's Explanation: {correct_explanation}

Output the XML only.
"""

VERIFIER_RUBRIC_INSTRUCTION = """
You are a strict verifier. Check the candidate answer ONLY on format and minimal content criteria.
Return exactly these fields as Yes/No on separate lines:
FormatOK: Does it contain the <answer> block with both <label> and <explanation>?
LabelOK: Is the label exactly one of {Positive, Negative} (case-insensitive) and NOT Uncertain/Neutral?
BalanceOK: Does the explanation mention at least one bullish and one bearish factor?

Candidate Answer:
{candidate}
"""

# Multiple Reflection Types for Diverse DPO Pairs
REFLECTION_RETRY_INSTRUCTION = """
You made an incorrect prediction. Simply try again with the same data. 
Rewrite your explanation as a cohesive essay.

STRICT OUTPUT SCHEMA (MANDATORY):
<answer>
<label>Positive</label> OR <label>Negative</label>
<explanation>[one cohesive essay of 150–200 words]</explanation>
</answer>

Output the XML only.
"""

REFLECTION_KEYWORDS_INSTRUCTION = """
You made an incorrect prediction. Focus on these overlooked key factors:
- Technical indicators (RSI, MACD, moving averages)
- Volume and participation
- Sentiment and social signals
- Macro and regulatory context
- Bitcoin–Gold correlation

STRICT OUTPUT SCHEMA (MANDATORY):
<answer>
<label>Positive</label> OR <label>Negative</label>
<explanation>[one cohesive essay of 150–200 words that integrates the above factors]</explanation>
</answer>

Output the XML only.
"""

REFLECTION_ADVICE_INSTRUCTION = """
You made an incorrect prediction. Apply the following advice:
- Weigh bullish and bearish signals equally
- Avoid bias from recent price action
- Integrate technical, sentiment, and macro signals
- Account for Gold’s influence

STRICT OUTPUT SCHEMA (MANDATORY):
<answer>
<label>Positive</label> OR <label>Negative</label>
<explanation>[one cohesive essay of 150–200 words following this advice]</explanation>
</answer>

Output the XML only.
"""

REFLECTION_EXPLANATION_INSTRUCTION = """
You made an incorrect prediction. Analyze why:

- What reasoning errors did you make?
- Which data was misinterpreted?
- What assumptions failed?

STRICT OUTPUT SCHEMA (MANDATORY):
<answer>
<label>Positive</label> OR <label>Negative</label>
<explanation>[one cohesive essay of 150–200 words addressing the errors]</explanation>
</answer>

Output the XML only.
"""

REFLECTION_INSTRUCTIONS_INSTRUCTION = """
You made an incorrect prediction. Follow this step-by-step reasoning process:

1. Identify the strongest bullish signal
2. Identify the strongest bearish signal
3. Compare relative strength
4. Consider Gold and macro influence
5. Decide on Positive or Negative
6. Justify clearly

STRICT OUTPUT SCHEMA (MANDATORY):
<answer>
<label>Positive</label> OR <label>Negative</label>
<explanation>[one cohesive essay of 150–200 words following this stepwise reasoning]</explanation>
</answer>

Output the XML only.
"""

REFLECTION_SOLUTION_INSTRUCTION = """
You made an incorrect prediction. Use this correct analytical approach:
- Start with technicals (price, volume, patterns)
- Add fundamentals (news, sentiment)
- Include cross-market factors (Gold, macro)
- Balance all evidence

STRICT OUTPUT SCHEMA (MANDATORY):
<answer>
<label>Positive</label> OR <label>Negative</label>
<explanation>[one cohesive essay of 150–200 words applying this approach]</explanation>
</answer>

Output the XML only.
"""

REFLECTION_COMPOSITE_INSTRUCTION = """
You made an incorrect prediction. Apply a comprehensive reflection:

1. Error Identification – what mistake was made?
2. Key Factors Missed – which signals were overlooked?
3. Improved Analysis – how should the approach change?
4. Corrected Reasoning – rebuild the case step by step
5. Final Decision – clear Positive/Negative with justification

STRICT OUTPUT SCHEMA (MANDATORY):
<answer>
<label>Positive</label> OR <label>Negative</label>
<explanation>[one cohesive essay of 150–200 words addressing all five reflection steps]</explanation>
</answer>

Output the XML only.
"""

REFLECTION_TYPES = {
    "retry": REFLECTION_RETRY_INSTRUCTION,
    "keywords": REFLECTION_KEYWORDS_INSTRUCTION,
    "advice": REFLECTION_ADVICE_INSTRUCTION,
    "explanation": REFLECTION_EXPLANATION_INSTRUCTION,
    "instructions": REFLECTION_INSTRUCTIONS_INSTRUCTION,
    "solution": REFLECTION_SOLUTION_INSTRUCTION,
    "composite": REFLECTION_COMPOSITE_INSTRUCTION
}
ENHANCED_CANDLE_EXAMPLES = """
EXAMPLE 1 (BULLISH):
Market: BTC High: 0.075, Low: 0.072, Volume: 0.008
Sentiment: BTC Positive: 0.80%, Negative: 0.10%, Neutral: 0.10%
Gold Positive: 0.20%, Negative: 0.10%, Neutral: 0.70%
<answer>
<label>Positive</label>
<explanation>Bitcoin shows a clear bullish structure with consecutive higher highs and higher lows, supported by strong volume at 0.008 that indicates active participation rather than passive drift. The sentiment profile tilts decisively positive, with 0.80% bullish against only 0.10% bearish, confirming that market participants are positioned for upside continuation. While headline risk exists, it does not dominate the technical setup, and buyers are consistently absorbing supply on dips. On the bearish side, the main concern is residual uncertainty that could trigger short-term volatility if unexpected news emerges, but this risk is minor relative to the current momentum. Cross-market signals reinforce the bullish case, as Gold sentiment remains largely neutral, suggesting that capital is not fleeing into safe-haven assets. This reduces downside drag on Bitcoin and allows risk-on flows to remain intact. Taken together, the evidence favors a positive call, with buying pressure, sentiment alignment, and the absence of safe-haven flows forming a coherent case for upward continuation.</explanation>
</answer>

EXAMPLE 2 (BEARISH):
Market: BTC High: 0.068, Low: 0.065, Volume: 0.003
Sentiment: BTC Positive: 0.10%, Negative: 0.70%, Neutral: 0.20%
Gold Positive: 0.60%, Negative: 0.10%, Neutral: 0.30%
<answer>
<label>Negative</label>
<explanation>Bitcoin presents a bearish configuration, with price action showing lower highs and lower lows, signaling supply dominance. Volume is weak at 0.003, reflecting a lack of dip-buying interest and limited conviction from bulls. Sentiment is decisively negative at 0.70%, further aligning with the price structure that shows ongoing distribution. On the bullish side, the only supportive factor is that some neutral positioning exists, which could slow the decline if buyers re-engage, but it is insufficient to reverse momentum. Meanwhile, Gold’s sentiment profile is firmly positive at 0.60%, indicating that investors are allocating capital toward safety rather than risk assets, diverting flows away from Bitcoin. This cross-market dynamic amplifies the bearish bias. Breaks below support further validate downside pressure, as buyers fail to regain initiative and sellers control liquidity pockets. Balancing both bullish and bearish aspects, the weight of evidence strongly favors a negative outlook, with technical weakness and capital flight into Gold driving the conclusion.</explanation>
</answer>

EXAMPLE 3 (BALANCED INPUTS → NEGATIVE):
Market: BTC High: 0.070, Low: 0.069, Volume: 0.004
Sentiment: BTC Positive: 0.25%, Negative: 0.35%, Neutral: 0.40%
Gold Positive: 0.40%, Negative: 0.20%, Neutral: 0.40%
<answer>
<label>Negative</label>
<explanation>Bitcoin trades in a compressed range near recent lows, with volume at 0.004 suggesting indecision and weak follow-through from buyers. Sentiment leans slightly bearish at 0.35% compared to 0.25% bullish, showing that pessimism outweighs optimism even if the margin is not overwhelming. On the bullish side, there is still some resilience, as nearly a quarter of sentiment remains constructive, preventing an outright collapse. However, the bearish picture dominates because repeated failed upticks occur at lower highs, and liquidity builds below intraday support, increasing the likelihood of a breakdown. Gold sentiment trends positive at 0.40%, reflecting a risk-off bias, and further reduces investor appetite for Bitcoin. This cross-market tilt toward safety undercuts bullish arguments and tips the balance toward downside pressure. Altogether, while inputs appear balanced at first glance, the underlying dynamics reveal stronger bearish weight, making a negative prediction the more consistent outcome.</explanation>
</answer>

EXAMPLE 4 (BALANCED INPUTS → POSITIVE):
Market: BTC High: 0.071, Low: 0.070, Volume: 0.005
Sentiment: BTC Positive: 0.30%, Negative: 0.20%, Neutral: 0.50%
Gold Positive: 0.20%, Negative: 0.30%, Neutral: 0.50%
<answer>
<label>Positive</label>
<explanation>Bitcoin holds a slightly higher low at 0.070, defending support with moderate volume at 0.005 that demonstrates constructive buyer interest. Sentiment leans modestly bullish at 0.30% versus 0.20% bearish, giving a slight advantage to positive positioning, while neutral sentiment dominates but does not block upward potential. On the bearish side, the narrow trading range highlights some caution, as momentum has not yet expanded into a breakout. However, Gold sentiment trends slightly negative at 0.30%, reducing safe-haven demand and signaling that capital is not rushing away from risk assets. This cross-market influence lightens downside drag and supports Bitcoin’s case for upside continuation. Buyers are gradually absorbing supply and pressing into prior resistance levels, while sellers fail to impose sustained pressure. Overall, despite some hesitancy, the balance of factors tilts positive, with defended support, improving breadth, and favorable cross-market alignment supporting a constructive outlook.</explanation>
</answer>
"""
ENHANCED_NEWS_EXAMPLES = """
EXAMPLE 1 (BULLISH):
News: ETF approval window opens; large custodians announce operational readiness
Sentiment: BTC Positive: 0.90%, Negative: 0.10%, Neutral: 0.00%
Gold Positive: 0.20%, Negative: 0.10%, Neutral: 0.70%
<answer>
<label>Positive</label>
<explanation>Regulatory progress and institutional infrastructure are combining to provide a strong bullish backdrop. The opening of an ETF approval window represents a regulatory breakthrough that could unlock new institutional demand, while custodians announcing operational readiness signals maturity in infrastructure that reduces adoption barriers. Sentiment is overwhelmingly positive at 0.90%, showing that investors are aligned with the bullish narrative. On the bearish side, there remains residual regulatory risk if approval is delayed, but this is outweighed by the positive catalysts. Cross-market dynamics further reinforce the upside case, as Gold sentiment is largely neutral, indicating that investors are not prioritizing safe-haven assets. This removes a key headwind and allows capital to stay engaged in Bitcoin. The confluence of regulatory catalysts, strong sentiment, and the absence of defensive flows makes the bullish case dominant, justifying a positive prediction.</explanation>
</answer>

EXAMPLE 2 (BEARISH):
News: Regulators intensify enforcement; new compliance constraints for major exchanges
Sentiment: BTC Positive: 0.10%, Negative: 0.80%, Neutral: 0.10%
Gold Positive: 0.70%, Negative: 0.10%, Neutral: 0.20%
<answer>
<label>Negative</label>
<explanation>Heightened enforcement pressure creates significant uncertainty for exchanges, undermining confidence and curtailing risk appetite. Compliance constraints often increase operational costs and can drive capital outflows from crypto venues. Sentiment is sharply negative at 0.80%, consistent with widespread concern across market participants. On the bullish side, the presence of some neutral sentiment at 0.10% suggests that not all investors are panicking, but this is insufficient to counterbalance the regulatory overhang. Cross-market dynamics magnify downside risks, as Gold sentiment is strongly positive at 0.70%, showing clear preference for safe-haven allocation. This indicates that investors are actively shifting capital into defensive assets rather than speculative risk plays. The combined effect of regulatory uncertainty, operational headwinds, and flight to safety makes the bearish case dominant, leading to a negative prediction.</explanation>
</answer>

EXAMPLE 3 (BALANCED INPUTS → NEGATIVE):
News: Mixed adoption progress but persistent legislative delays
Sentiment: BTC Positive: 0.40%, Negative: 0.30%, Neutral: 0.30%
Gold Positive: 0.30%, Negative: 0.20%, Neutral: 0.50%
<answer>
<label>Negative</label>
<explanation>Adoption headlines provide some optimism, but legislative delays continue to dampen overall conviction and make institutional engagement hesitant. Sentiment tilts only slightly positive at 0.40%, which is not strong enough to dominate the bearish influence of policy uncertainty. On the bullish side, real-world adoption progress shows that underlying demand remains intact, but the lack of regulatory clarity acts as a ceiling for momentum. Gold sentiment is moderately positive at 0.30%, reflecting a mild shift toward safe-haven assets that adds drag to Bitcoin’s trajectory. While the environment is not decisively bearish, the interaction of hesitant sentiment, policy overhang, and safe-haven preference tips the balance toward a negative outcome. Thus, despite some constructive developments, the weight of evidence supports a bearish prediction.</explanation>
</answer>

EXAMPLE 4 (BALANCED INPUTS → POSITIVE):
News: Major retailer onboards BTC payments; regulatory timeline ambiguity persists
Sentiment: BTC Positive: 0.37, Negative: 0.32, Neutral: 0.31
Gold Positive: 0.33, Negative: 0.34, Neutral: 0.33
<answer>
<label>Positive</label>
<explanation>The decision of a major retailer to accept Bitcoin payments provides a tangible adoption boost, signaling that mainstream commercial usage is expanding. This real-world integration strengthens the fundamental case for Bitcoin and validates its role beyond speculation. Sentiment tilts slightly positive at 0.37 versus 0.32 negative, reflecting cautious optimism even amid regulatory uncertainty. On the bearish side, delays in regulatory timelines keep some investors hesitant, but the concrete demand signal from adoption outweighs this concern. Gold sentiment remains mixed, reducing its role as a competing safe-haven, which prevents strong downside pressure on Bitcoin. With adoption progress providing a clear bullish driver, sentiment moderately supportive, and no overwhelming defensive flows into Gold, the case for a positive outcome is stronger. Thus, the evidence supports a constructive prediction for Bitcoin.</explanation>
</answer>
"""
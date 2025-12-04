# ===========================
# Summarize Examples
# ===========================

SUMMARIZE_EXAMPLES = """Tweets and News:
Glassnode: Bitcoin Exchange Net Position Change shows consistent outflows across major centralized exchanges.
Fear & Greed Index drops to 24 – extreme fear territory.
Binance resumes BTC withdrawals after brief pause due to network congestion.
On-chain data shows whale wallets accumulating BTC at key support level.
Fed expected to raise interest rates by 25bps next week – markets cautious.
SEC intensifies crackdown on unregistered crypto exchanges.

Facts:
- Exchange outflows signal accumulation and reduced sell pressure (bullish).
- Extreme fear index suggests overselling and potential reversal (bullish).
- Binance issue resolved quickly, removing systemic risk (bullish).
- Whale accumulation supports strong demand (bullish).
- Fed tightening adds downside pressure (bearish).
- SEC enforcement raises regulatory uncertainty (bearish).

Bitcoin Price Movement: Positive
Explanation: Bitcoin’s overall outlook skews positive as structural accumulation outweighs policy and regulatory risks. Consistent outflows from exchanges show that coins are being withdrawn into long-term custody, reducing available supply for immediate selling. The Fear & Greed Index at 24 reflects capitulation levels that historically align with reversal conditions, while whale wallets buying at support strengthens this case by confirming demand from sophisticated investors. Additionally, Binance’s ability to resolve withdrawal congestion quickly minimizes concerns over systemic fragility, further stabilizing the market environment. On the bearish side, a 25bps Fed rate hike continues to pressure liquidity across risk assets, and the SEC’s enforcement stance against centralized exchanges introduces compliance uncertainty. However, compared to the breadth of accumulation and sentiment-based overselling, these headwinds remain secondary. With buying pressure confirmed both on-chain and via behavioral metrics, the weight of evidence tilts decisively toward a positive prediction for Bitcoin’s price movement.
"""

ALT_SUMMARIZE_EXAMPLES = """Tweets and News:
MicroStrategy adds 3,000 BTC to its balance sheet.
Elon Musk tweets a cryptic message with a Bitcoin emoji.
Coinbase wallet faces phishing attack, $1.5M in BTC lost.
CryptoQuant: BTC Miner Reserve at 2-year low.
Bitcoin Lightning Network capacity surpasses 5,000 BTC.
EU proposes strict regulations for crypto asset transfers.

Facts:
- MicroStrategy’s BTC purchase reflects institutional confidence (bullish).
- Musk’s tweet sparks speculative demand (bullish).
- Coinbase phishing incident raises security risks (bearish).
- Miner reserves at 2-year low reduce sell pressure (bullish).
- Lightning growth shows adoption (bullish).
- EU regulations add compliance risk (bearish).

Bitcoin Price Movement: Positive
Explanation: Bitcoin’s momentum appears constructive, supported by multiple bullish catalysts that outweigh regulatory and security risks. MicroStrategy’s additional 3,000 BTC purchase highlights institutional conviction and signals confidence in long-term value. Miner reserves reaching a two-year low suggest reduced immediate selling capacity, reinforcing supply tightness. Network fundamentals also strengthen the case, with Lightning capacity surpassing 5,000 BTC, showing steady adoption of the payments layer. On the bearish side, a phishing attack on Coinbase wallets raises concerns over infrastructure security, while new EU regulations could tighten compliance burdens and reduce anonymity, introducing headwinds for some market participants. Elon Musk’s cryptic tweet provides a speculative short-term boost, energizing sentiment alongside other bullish factors. Taken together, institutional accumulation, structural supply reduction, and network growth outweigh negative news. The evidence supports a positive outlook, with adoption and demand forces overpowering security and regulatory risks.
"""

# ===========================
# Candle Sentiment Examples
# ===========================

CANDLE_SENTIMENT_EXAMPLES = """=== Example 1 ===
Market Data:
Bitcoin: High=45,200, Low=44,100, Volume=2,500
Gold: High=2,050, Low=2,040, Volume=15,000

Sentiment Data:
Bitcoin: Positive 45.2%, Negative 25.1%, Neutral 29.7%
Gold: Positive 38.3%, Negative 28.5%, Neutral 33.2%

Bitcoin Price Movement: Positive
Explanation: Bitcoin maintains a bullish posture, with price action forming higher highs and lows while trading on strong volume of 2,500, indicating sustained buying interest. Sentiment confirms this structure, as 45.2% of signals skew positive compared to only 25.1% negative. Bears are present but lack conviction relative to bullish momentum. On the bearish side, moderate Gold positivity at 38.3% suggests a degree of safe-haven demand that could compete with Bitcoin flows, highlighting caution in global markets. Still, this safe-haven tilt is not strong enough to dominate BTC’s momentum. Buyers continue to defend support and press into resistance zones, reflecting healthy participation. The cross-market influence is limited, as Gold’s appeal remains modest while Bitcoin’s technical resilience remains intact. Considering volume confirmation, sentiment alignment, and structural support, the bullish side clearly outweighs the bearish. The balance of evidence favors a positive call for Bitcoin’s next move.

=== Example 2 ===
Market Data:
Bitcoin: High=42,800, Low=41,500, Volume=3,200
Gold: High=2,080, Low=2,070, Volume=18,000

Sentiment Data:
Bitcoin: Positive 28.1%, Negative 42.7%, Neutral 29.2%
Gold: Positive 45.2%, Negative 18.5%, Neutral 36.3%

Bitcoin Price Movement: Negative
Explanation: Bitcoin signals weakness, with lower highs and lower lows appearing on heavy sell volume of 3,200, a pattern consistent with bearish continuation. Sentiment aligns with price structure, showing 42.7% negative versus only 28.1% positive, suggesting selling pressure dominates. On the bullish side, some neutral participation indicates that buyers have not completely abandoned the market, but this support is too weak to reverse momentum. Gold sentiment is strongly positive at 45.2%, indicating that investors are actively shifting into safe-haven assets rather than maintaining exposure to risk-on positions like BTC. This cross-market flow amplifies downside risks for Bitcoin, as capital allocation clearly favors defensive postures. With bearish volume confirmation, negative sentiment, and a strong safe-haven bid in Gold, the evidence collectively favors a negative outlook for Bitcoin in the near term.

=== Example 3 (Mixed → Positive) ===
Market Data:
Bitcoin: High=43,800, Low=43,200, Volume=2,000
Gold: High=2,055, Low=2,050, Volume=12,500

Sentiment Data:
Bitcoin: Positive 36.1%, Negative 33.2%, Neutral 30.7%
Gold: Positive 31.3%, Negative 32.5%, Neutral 36.2%

Bitcoin Price Movement: Positive
Explanation: Bitcoin consolidates in a narrow trading range between 43,200 and 43,800, with moderate volume of 2,000 suggesting stability rather than weakness. Sentiment tilts slightly bullish at 36.1% against 33.2% negative, reflecting cautious optimism. On the bearish side, a relatively high neutral share highlights lingering indecision, and this hesitancy tempers the upside case. Cross-market dynamics provide balance, as Gold sentiment is mixed with near-equal positive and negative signals, showing that investors are not decisively prioritizing safe-haven assets. This neutrality in Gold reduces downside drag for Bitcoin and allows its modest bullish edge to carry more weight. Despite the presence of hesitation, defended price levels and modest sentiment skew favor a positive outcome. Thus, the evidence supports a cautiously optimistic call for Bitcoin.

=== Example 4 (Mixed → Negative) ===
Market Data:
Bitcoin: High=43,500, Low=43,200, Volume=1,800
Gold: High=2,058, Low=2,052, Volume=13,000

Sentiment Data:
Bitcoin: Positive 34.2%, Negative 36.0%, Neutral 29.8%
Gold: Positive 39.1%, Negative 29.5%, Neutral 31.4%

Bitcoin Price Movement: Negative
Explanation: Bitcoin trades within a tight range but on weak volume of 1,800, signaling hesitation and lack of strong buyer commitment. Sentiment slightly favors the bearish side at 36.0% negative compared to 34.2% positive, reinforcing the view that downside risk is marginally stronger. On the bullish side, there is still a notable share of neutral sentiment that could stabilize prices, but this remains insufficient to outweigh sellers’ control. Gold sentiment trends moderately positive at 39.1%, reflecting a tilt toward safe-haven allocation that further competes with Bitcoin demand. This cross-market preference for safety amplifies bearish factors and weakens the case for upside. Considering the balance of weak volume, bearish sentiment tilt, and Gold’s safe-haven pull, the evidence leans toward a negative prediction for Bitcoin’s next move.
"""

# ===========================
# News Summary Examples
# ===========================

NEWS_SUMMARY_EXAMPLES = """=== Example 1 ===
News Summary: Major institutional investors announce increased Bitcoin allocations. Regulatory clarity improves in key markets. Positive adoption news from corporations.

Sentiment Data:
Bitcoin: Positive 48.2%, Negative 22.1%, Neutral 29.7%
Gold: Positive 35.3%, Negative 30.5%, Neutral 34.2%

Bitcoin Price Movement: Positive
Explanation: Bitcoin’s outlook improves as institutional investors expand allocations, signaling long-term confidence and enhanced legitimacy. Regulatory clarity in key jurisdictions reduces uncertainty, creating a friendlier environment for corporate and retail adoption. Positive corporate adoption headlines further reinforce structural demand. Sentiment is decisively bullish at 48.2%, outweighing 22.1% negative signals. On the bearish side, some neutral sentiment remains that may limit immediate momentum, but it does not offset the broader trend. Gold sentiment is moderately positive at 35.3%, suggesting some safe-haven demand, but not strong enough to drain flows from Bitcoin. Cross-market dynamics therefore support Bitcoin’s case, as capital allocation does not decisively shift away from risk. Overall, the alignment of institutional adoption, regulatory improvement, and strong sentiment outweighs modest headwinds, justifying a positive prediction.

=== Example 2 ===
News Summary: Regulatory crackdown on crypto exchanges. Negative comments from central banks. Institutional investors reduce crypto exposure.

Sentiment Data:
Bitcoin: Positive 25.1%, Negative 45.7%, Neutral 29.2%
Gold: Positive 42.2%, Negative 25.5%, Neutral 32.3%

Bitcoin Price Movement: Negative
Explanation: Bitcoin faces pressure from multiple bearish drivers. A regulatory crackdown on exchanges introduces compliance risks, while negative central bank rhetoric undermines confidence in crypto assets. Institutional investors reducing exposure further weakens demand. Sentiment is strongly bearish at 45.7%, well above positive levels at 25.1%, confirming market alignment with downside risks. On the bullish side, some neutral sentiment remains, but it is not sufficient to counter regulatory and institutional headwinds. Cross-market dynamics amplify pressure, as Gold sentiment is firmly positive at 42.2%, signaling that capital is flowing into safe-haven assets instead of Bitcoin. With regulatory, institutional, and sentiment factors aligned against Bitcoin, the balance of evidence points to a negative outlook.

=== Example 3 (Mixed → Positive) ===
News Summary: A major retailer begins accepting BTC payments, while regulators delay crypto legislation.

Sentiment Data:
Bitcoin: Positive 37.2%, Negative 32.4%, Neutral 30.4%
Gold: Positive 33.2%, Negative 34.1%, Neutral 32.7%

Bitcoin Price Movement: Positive
Explanation: Adoption news provides a strong positive catalyst, as mainstream retailers onboarding BTC payments validate its utility as a transactional asset. This adoption offsets uncertainty caused by delayed regulatory progress, showing that demand-side drivers remain intact. Sentiment supports the bullish narrative, with 37.2% positive versus 32.4% negative, reflecting cautious optimism despite lingering risks. On the bearish side, the delay in legislation adds to short-term ambiguity, which could restrain institutional involvement. However, Gold sentiment is balanced, with near-equal positive and negative values, showing no decisive safe-haven preference. This neutrality reduces competitive headwinds and allows adoption to dominate the outlook. Taken together, the balance tilts positive, and Bitcoin’s price movement is predicted to rise.

=== Example 4 (Mixed → Negative) ===
News Summary: Adoption headlines emerge, but regulatory uncertainty and cautious institutions dominate.

Sentiment Data:
Bitcoin: Positive 33.1%, Negative 34.2%, Neutral 32.7%
Gold: Positive 38.3%, Negative 28.5%, Neutral 33.2%

Bitcoin Price Movement: Negative
Explanation: While adoption headlines create initial optimism, ongoing regulatory uncertainty and cautious institutional positioning weigh more heavily. Bitcoin sentiment tilts slightly bearish at 34.2% versus 33.1% positive, reflecting the market’s unease. On the bullish side, adoption provides a partial buffer, but it fails to counterbalance the risks. Gold sentiment trends positive at 38.3%, reinforcing safe-haven flows that divert capital away from Bitcoin. This cross-market tilt strengthens the bearish side of the argument. Taken together, adoption alone cannot offset regulatory overhang and defensive allocation patterns. The evidence supports a negative prediction for Bitcoin’s price movement.
"""

# ===========================
# Predict Examples
# ===========================

PREDICT_EXAMPLES = """=== Example 1 ===
Market: BTC
Related Market: GOLD

BTC Market Data:
High: 104,500.00 | Low: 101,200.00 | Volume: 32,150.00
GOLD Market Data:
High: 2,772.00 | Low: 2,752.00 | Volume: 241,250.00

BTC Sentiment: Positive 35.2%, Neutral 45.1%, Negative 19.7%
GOLD Sentiment: Positive 42.3%, Neutral 48.5%, Negative 9.2%

Bitcoin Price Movement: Negative
Explanation: Bitcoin shows downside pressure with a broad trading range and heavy sell volume of 32,150, confirming bearish continuation risk. Sentiment reflects weak conviction, with neutral dominance at 45.1% overshadowing 35.2% positive signals, showing hesitation among buyers. On the bullish side, some positive sentiment remains, indicating that parts of the market still anticipate support at lower levels, but this is overshadowed by bearish structure. Gold sentiment is strongly positive at 42.3%, and with nearly half the signals neutral, capital clearly leans toward safety rather than risk. This cross-market shift amplifies downside forces for Bitcoin. Altogether, the evidence aligns with a negative outcome.

=== Example 2 ===
Market: BTC
Related Market: GOLD

BTC Market Data:
High: 103,200.00 | Low: 102,100.00 | Volume: 28,770.00
GOLD Market Data:
High: 2,785.00 | Low: 2,765.00 | Volume: 195,290.00

BTC Sentiment: Positive 32.1%, Neutral 48.7%, Negative 19.2%
GOLD Sentiment: Positive 51.2%, Neutral 38.5%, Negative 10.3%

Bitcoin Price Movement: Negative
Explanation: Bitcoin consolidates narrowly with moderate volume, reflecting indecision and limited momentum. Sentiment is dominated by neutrality at 48.7%, while positives remain low at 32.1%, showing insufficient enthusiasm to drive upside. On the bullish side, the presence of buyers prevents deeper declines, but lack of conviction caps potential. Gold sentiment is decisively positive at 51.2%, highlighting demand for safe-haven exposure, which reduces investor appetite for Bitcoin. This flow toward defensive positioning amplifies bearish bias. The evidence therefore supports a negative outlook.

=== Example 3 (Mixed → Positive) ===
Market: BTC
Related Market: GOLD

BTC Market Data:
High: 102,800.00 | Low: 102,000.00 | Volume: 27,500.00
GOLD Market Data:
High: 2,770.00 | Low: 2,760.00 | Volume: 210,000.00

BTC Sentiment: Positive 37.5%, Neutral 40.0%, Negative 22.5%
GOLD Sentiment: Positive 34.0%, Neutral 39.5%, Negative 26.5%

Bitcoin Price Movement: Positive
Explanation: Bitcoin stabilizes within a narrow band between 102,000 and 102,800, supported by steady trading volume at 27,500 that prevents deeper retracement. Sentiment is moderately bullish at 37.5%, outpacing negative at 22.5%, though neutrality remains high, showing some caution. On the bearish side, lingering indecision reflected in neutral sentiment limits aggressive upside expansion. Cross-market dynamics offer support, as Gold sentiment is balanced between positive and negative, showing that investors are not strongly shifting to safe-haven demand. This neutrality in Gold removes a potential headwind for Bitcoin. Taken together, the presence of modest bullish sentiment, defended price levels, and neutral Gold flows make the case for a cautiously positive outcome.
"""

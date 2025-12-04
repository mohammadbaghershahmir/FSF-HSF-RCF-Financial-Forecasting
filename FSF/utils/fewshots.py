SUMMARIZE_EXAMPLES = """Tweets and News:
Glassnode: Bitcoin Exchange Net Position Change shows consistent outflows across major centralized exchanges.
Fear & Greed Index drops to 24 – extreme fear territory.
Binance resumes BTC withdrawals after brief pause due to network congestion.
On-chain data shows whale wallets accumulating BTC at key support level.
Fed expected to raise interest rates by 25bps next week – markets cautious.
SEC intensifies crackdown on unregistered crypto exchanges.

Facts:
- Consistent exchange outflows suggest stronger holding behavior and reduced immediate sell pressure.
- Fear & Greed Index at 24 indicates extreme fear, potential overselling, or accumulation opportunities.
- Binance briefly paused BTC withdrawals due to congestion but quickly resumed, limiting broader risk.
- Whale wallets accumulating at support show underlying bullish interest.
- An expected 25bps Fed hike may add downside pressure via tighter conditions.
- SEC crackdown on unregistered exchanges raises regulatory uncertainty across crypto markets.
"""

ALT_SUMMARIZE_EXAMPLES = """Tweets and News:
MicroStrategy adds 3,000 BTC to its balance sheet.
Elon Musk tweets a cryptic message with a Bitcoin emoji.
Coinbase wallet faces phishing attack, $1.5M in BTC lost.
CryptoQuant: BTC Miner Reserve at 2-year low.
Bitcoin Lightning Network capacity surpasses 5,000 BTC.
EU proposes strict regulations for crypto asset transfers.

Facts:
- MicroStrategy’s 3,000 BTC purchase signals ongoing institutional, long-term bullish positioning.
- Elon Musk’s cryptic tweet sparks speculative excitement and possible short-term volatility.
- A phishing attack on Coinbase Wallet causing $1.5M losses highlights security concerns.
- BTC Miner Reserve at a 2-year low suggests weaker miner sell pressure.
- Lightning capacity above 5,000 BTC reflects network growth and real-world micropayment adoption.
- Proposed EU regulations may increase compliance friction and reduce anonymity in crypto transfers.
"""

PREDICT_EXAMPLES = """=== Example 1 ===
Market: BTC
Related Market: GOLD

BTC Market Data:
High: 104,500.00
Low: 101,200.00
Volume: 32,150.00

GOLD Market Data:
High: 2,772.00
Low: 2,752.00
Volume: 241,250.00

BTC Sentiment:
positive: 35.2%
neutral: 45.1%
negative: 19.7%

GOLD Sentiment:
positive: 42.3%
neutral: 48.5%
negative: 9.2%

(1) Bitcoin Price Movement: Negative
(2) Confidence Level: High
(3) Primary Bitcoin Factors:
    - BTC price down -1.92% on strong volume (32,150), indicating selling pressure
    - Intraday range 104,500–101,200 shows sustained downside momentum
    - Sentiment mix insufficient to counteract selling
(4) Cross-Market Impact:
    - Gold’s +0.60% rise reflects risk-off flows
    - Investors rotating into traditional safe-haven assets
(5) Rationale:
    - Gold’s rally and BTC’s sell volume align with historical risk-off patterns, where capital shifts away from BTC toward Gold. This combination supports a bearish near-term outlook for BTC.

=== Example 2 ===
Market: GOLD
Related Market: BTC

GOLD Market Data:
High: 2,785.00
Low: 2,765.00
Volume: 195,290.00

BTC Market Data:
High: 103,200.00
Low: 102,100.00
Volume: 28,770.00

GOLD Sentiment:
positive: 51.2%
neutral: 38.5%
negative: 10.3%

BTC Sentiment:
positive: 32.1%
neutral: 48.7%
negative: 19.2%

(1) Bitcoin Price Movement: Negative
(2) Confidence Level: Medium
(3) Primary Bitcoin Factors:
    - BTC slightly down (-0.45%) with moderate volume (28,770)
    - Sentiment mostly neutral (48.7%), lacking bullish momentum
    - No strong BTC-specific catalysts in the data
(4) Cross-Market Impact:
    - Gold’s +0.82% gain and positive sentiment show safe-haven demand
    - BTC weakness further reinforces Gold’s appeal
(5) Rationale:
    - Gold benefits from risk-off flows and geopolitical tension, supported by positive sentiment. BTC’s neutral skew and mild decline suggest constrained upside, with investor preference leaning toward Gold as a safer asset.
"""

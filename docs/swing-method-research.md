# What works in swing trading? Research notes

Written 2026-11-16, after the backtests in `reports/` showed the mechanical Qullamaggie rules losing
money in both markets. **Information only, not investment advice.** I am not a licensed adviser, and
nothing here says what you personally should buy or sell. Numbers marked *unverified* come from
secondary sources (blogs, summaries) that I could not check against the original.

## Short answer

- I found **no well-tested method for holding stocks a few days with tight stops that shows a reliable
  profit after costs**. That includes mechanical breakout trading. The evidence there is hobbyist
  backtests that disagree with each other, plus our own test, which lost.
- The methods with strong evidence are **slower**: holding the strongest stocks for months (momentum),
  or riding long trends with a wide stop (trend following). Winners in the best trend-following study
  were held for a year on average.
- So the evidence does not point to a better breakout rule. It points to **keeping the stock selection
  the scanner already does, and dropping the breakout-day entry and the tight stop**.
- Even the well-evidenced methods have long bad stretches. India's momentum index returned 7.9% a year
  over the last five years and lost 8.5% in the last one.

## 1. Why the Qullamaggie version failed in our tests

Both backtests cover 2022-07 to 2026-11 (see `reports/backtest-us.md`, `reports/backtest-india.md`).

| | US | India |
|---|---|---|
| Times a watchlist stock reached its trigger | 11,362 | 5,853 |
| Share that failed to confirm by the close | 90% | 81% |
| Buy every trigger: win rate | 26% | 25% |
| Buy every trigger: average win / average loss | +12.0% / −4.9% | +8.2% / −3.8% |
| Buy every trigger: mean per trade after costs | −0.56% | −0.86% |
| Best tradeable version, yearly return (next-day entry) | −4.4% | −9.2% |
| Index, yearly return | SPY +16.2% | Nifty +10.1% |

Four things explain the result:

1. **Most triggers are false.** Eight or nine times out of ten the stock touched the trigger and then
   closed back below it, or closed without the volume. A buy order at the trigger fills on all of them.
2. **The profit of a real breakout is made on the breakout day itself.** Measured from the trigger price,
   confirmed breakouts were up 1.4% to 7.8% five days later. Measured from the next morning's open, the
   first moment you know the breakout was real, they were between −0.6% and +0.4%, no better than the
   index. By the time the breakout is confirmed, the move has been paid for.
3. **The stop is so tight that the method needs very large winners.** With a 25% win rate, winners must be
   about 3 times the size of losers just to break even. Ours were 2.4 times (US) and 2.2 times (India).
4. **It was a poor period for momentum in general.** NSE's own momentum index trailed the Nifty over
   roughly these years (section 2.1). In the US, 2025 was the only year in which buying every trigger
   made money.

What our test cannot say: whether a skilled person choosing a few setups by eye does better than the
mechanical version. Qullamaggie reports a win rate of 25–30%, the same as ours, so his results depend on
choosing the trades with the biggest winners. That skill is not written down in rules, and his record is
one trader's. We hear about the traders it worked for, not the ones it didn't.

## 2. What the published evidence says

| Method | Typical hold | Strength of evidence | Main weakness |
|---|---|---|---|
| Momentum: hold the strongest stocks of the last 6–12 months | 1–6 months | Strong: decades, most countries, including India | Long drawdowns, sharp crashes when a fallen market rebounds |
| Long-trend following: buy new all-time highs, wide trailing stop | Months to years | Good: 1950–2024 US test, held up after publication | Under 7% of trades make all the profit; the test is by a trend-following manager |
| Earnings drift: buy after a strong earnings surprise | Weeks | Was strong, now faded in larger stocks | Left mostly in tiny stocks, where costs are high |
| Buying pullbacks (short-term reversal) | Days | Mixed: real before costs, thin after | Costs; mostly blog backtests on index funds, not stocks |
| Chart rules and short breakouts with tight stops | Days | Weak to negative | No rigorous positive test; ours lost |

### 2.1 Momentum over months

Stocks that rose most over the past 6–12 months have tended to keep outperforming for the next few
months. This is the most replicated pattern in stock research (Jegadeesh and Titman 1993, and later
studies across most countries; this paragraph is from my knowledge of those papers). Studies of NSE
stocks find it in India too, driven by the winners.

Costs matter. A study of anomaly trading costs found that strategies replacing less than half the
portfolio a month mostly stayed profitable after costs, and few faster ones did. Its most effective
fix was simple: don't sell a holding the moment it leaves the top group; keep it until it falls well
down the ranking.

The weaknesses are real:
- Across 97 published patterns, returns were 26% lower after the sample period and 58% lower after
  publication (McLean and Pontiff 2016).
- Momentum has rare, severe crashes, mostly when the market rebounds after a fall (Daniel and Moskowitz).
- **India, live record.** NSE's Nifty200 Momentum 30 index picks 30 stocks by 6- and 12-month return
  adjusted for volatility, twice a year. Its factsheet of 30 October 2026 shows total returns of 18.05%
  a year since April 2005, but 7.93% a year over five years, −8.48% over one year and −7.62% so far in
  2026. The index was launched in August 2020, so everything before that is a back-calculation.
- An independent Indian test (Freefincal, Dec 2006–Jun 2025, including delisted stocks, after costs and
  tax; *unverified*) found 14.6% a year for the 30 momentum stocks against 10.4% for the Nifty 50, with
  a worst fall of 71%. The extra return came from the less-traded half of the stocks.
- **US.** The MSCI momentum fund trailed the S&P 500 in 2022 and 2023 and beat it in 2024 and 2025
  (*unverified* yearly figures).

### 2.2 Long-trend following on single stocks

Zarattini, Pagani and Wilcox (2025) tested one fixed rule on every liquid US stock from 1950 to 2024,
including delisted ones: buy at the next open after a stock closes at an all-time high; sell when it
closes below a stop set 10 average daily ranges under the high. The stop only moves up.

- 66,000 trades. 44% won. The average win was 1.9 times the amount risked, the average loss 0.7 times.
- 56% of trades lost, 37% roughly paid those losses back, and **fewer than 7% made all the profit**.
- Winning trades lasted 370 days on average.
- The rule was first published in 2005; results from 2005 to 2024 held up with a modest decline.
- As a portfolio (1991–2024): 15.0% a year before costs against 11.2% for the market, with a worst fall
  of 32% against 55%. After costs a small account kept little (2.5% a year at $100,000) because of
  constant small adjustments; with a rule that skips small adjustments, about 13% at every size.

Two cautions: one author runs a trend-following fund, and the portfolio holds a large number of stocks
at once, which is hard to copy by hand.

This is the closest tested relative of what Qullamaggie does, and the differences are the point: the
stop is roughly ten times wider than ours, and winners are held for a year, not a week.

### 2.3 Earnings drift

Stocks used to keep drifting for about two months after a surprising earnings report. In the US this
has largely gone outside the smallest stocks: prices now adjust on the announcement day (Martineau
2022; a 2022–2024 S&P 500 study agrees). Indian studies are mixed, with one 2014–2018 NSE study finding
no drift. Qullamaggie's "episodic pivot" setup relies on this effect. I found no rigorous test of it.

### 2.4 Buying pullbacks

Stocks that fell hard over the last few days tend to bounce. Before costs the effect is large; after
costs it mostly survives only in the biggest, most liquid stocks, and one of those studies assumes
institution-size trading. The popular "RSI(2)" pullback rules are supported mainly by blog backtests on
index funds, and a critic notes the published rules may have been tuned to look good. The pattern is
the opposite of breakout trading: many small wins and an occasional large loss.

### 2.5 Chart rules and short breakouts

- A study of thousands of technical rules on the Dow from 1897 to 2011 found that, after correcting for
  how many rules were tried, the best rules could not be picked in advance and low trading costs wiped
  out the gains (Bajgrowicz and Scaillet 2012).
- I found no rigorous test of Qullamaggie's rules. One blog backtest reports 19% a year since 2007 with
  a 32% win rate (*unverified*), but its entries are estimated from daily bars. We caught exactly that
  problem in our own code, where it created a fake +50% a year.

### 2.6 How active traders actually do

- US households, 1991–1996: the most active fifth earned 11.4% a year while the market returned 17.9%
  (Barber and Odean 2000, from my knowledge of the paper).
- India (SEBI): about 7 in 10 intraday traders in shares lost money in recent years, and about 93% of
  individual futures and options traders lost over 2021-22 to 2023-24 (secondary reports).
- Taiwan: fewer than 1% of day traders were reliably profitable.

These are day traders and derivatives traders, not swing traders, so they are a warning about fast
trading in general, not a measurement of swing trading.

## 3. What this means

- **Evidence gets stronger as the holding period gets longer.** Days: weak. Weeks: faded. Months: strong.
  A year or more: good. Faster methods also pay costs more often.
- **A few big winners carry everything.** Fewer than 7% of trend trades made all the profit in the study
  above, and about 4% of US stocks account for all the stock market's gain over Treasury bills since
  1926 (Bessembinder). A tight stop and selling a third after three days cut those winners short. In our
  backtest the typical trade lasted 2 to 5 days.
- **Tax in India favours longer holds.** Gains on shares held 12 months or less are taxed at 20% plus
  cess; longer holds at 12.5% above ₹1.25 lakh a year. Our backtests leave tax out. Check your own
  situation with a tax professional.

## 4. Candidates for this project, in order

0. **A diagnostic first, not a strategy.** Do the scanner's leaders (up 25% in a month, 50% in three, or
   150% in six) beat the index over the next one to three months, ignoring entries and stops? If not,
   the selection itself has no value in this data and candidates 1 and 2 start from a weak base.
1. **Monthly momentum rotation.** Hold the top N liquid stocks by 6- and 12-month return adjusted for
   volatility; replace a holding only when it falls well down the ranking. Rules copied from published
   sources (NSE's index method, the cost study above), not tuned by us.
2. **Long-trend version.** Buy a new all-time or 52-week high close; exit on a wide trailing stop that
   only moves up; hold for months.
3. **Pullback entry on leaders.** Weakest evidence; test last, if at all.

Not recommended: more filters or exits on the current breakout rules. Eight exits and one market filter
have been tried on this data already.

## 5. How to test without fooling ourselves

- Copy the rules from a published source and fix them, and the pass mark, before running anything.
- Use more history. The cache holds five years, which is one market period. Yahoo has 15–20 years for
  today's stocks. Delisted stocks are still missing, so any result is an upper limit.
- Keep the last two years out of every test until the rules are final.
- Test one candidate at a time and report it even if it fails.
- Then paper trade for 6–12 months. That is the only test the past can't flatter.

## Sources

Own data: `reports/backtest-us.md`, `reports/backtest-india.md`.

- Zarattini, Pagani, Wilcox (2025), "Does Trend Following Still Work on Stocks?": <https://concretumgroup.com/does-trend-following-still-work-on-stocks/>
- NSE Indices, Nifty200 Momentum 30 factsheet, 30 Oct 2026: <https://www.niftyindices.com/Factsheet/Factsheet_Nifty200_Momentum30.pdf>
- Freefincal, "Is Indian Momentum Investing Just a Liquidity Illusion?": <https://freefincal.com/is-indian-momentum-investing-just-a-liquidity-illusion/>
- Indian momentum studies: <https://econpapers.repec.org/RePEc:eme:jamrpp:jamr-11-2015-0081>, <https://nsearchives.nseindia.com/content/research/res_paperfinal223.pdf>
- McLean and Pontiff (2016): <https://ivey.uwo.ca/media/3775549/pontiff.pdf>
- Daniel and Moskowitz, "Momentum Crashes": <https://www.nber.org/system/files/working_papers/w20439/w20439.pdf>
- Novy-Marx and Velikov, "A Taxonomy of Anomalies and Their Trading Costs": <https://www.nber.org/papers/w20721>
- Martineau, "Rest in Peace Post-Earnings Announcement Drift": <https://cfr.ivo-welch.info/published/papers/martineau2021rest.pdf>
- Quantpedia, "50 Years in PEAD Research": <https://quantpedia.com/50-years-in-pead-research/>
- S&P 500 earnings study, 2022–2024 (Mondello): <https://repositorio.ucp.pt/bitstreams/c9033501-675d-4fda-a389-c1b18a4ffc6a/download>
- NSE earnings drift study (Gupta and Dhusia 2021): <https://journals.vutbr.cz/index.php/trends/article/view/541>
- Short-term reversal and costs: <https://econpapers.repec.org/RePEc:eee:jbfina:v:36:y:2012:i:2:p:371-382>, <https://robeco.com/en-ch/insights/2011/11/short-term-residual-reversal>, <https://quantpedia.com/strategies/short-term-reversal-in-stocks>
- RSI(2) critique: <https://www.priceactionlab.com/Blog/?p=50182>
- Bajgrowicz and Scaillet (2012): <https://econpapers.repec.org/RePEc:eee:jfinec:v:106:y:2012:i:3:p:473-491>
- Qullamaggie blog backtest: <https://stonkscapital.substack.com/p/modeling-kullamagi-part-2-momentum>
- Barber and Odean: <https://faculty.haas.berkeley.edu/odean/papers/returns/Returns.html>
- SEBI trader loss studies (secondary): <https://businessworld.in/article/93-of-futures-options-investors-incur-significant-losses-sebi-study-534045>
- Day-trader studies summary: <https://www.cnbc.com/2020/11/20/attention-robinhood-power-users-most-day-traders-lose-money.html>
- Bessembinder, "Do Stocks Outperform Treasury Bills?": <https://wpcarey.asu.edu/department-finance/faculty-research/do-stocks-outperform-treasury-bills>
- US momentum fund yearly returns: <https://www.trefis.com/data/companies/mtum>
- Indian capital gains tax rates: <https://www.precize.in/blogs/capital-gains-tax-on-shares-in-india-ltcg-stcg-rates-holding-periods-and-budget-2026>

import pandas as pd

EQUITY_SLEEVE = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLRE", "XLB", "XLY", "XLC"]
MACRO_SLEEVE  = ["TLT", "IEF", "GLD", "UUP"]
TICKERS       = EQUITY_SLEEVE + MACRO_SLEEVE

EQUITY_TOP_N  = 4   # long top N sectors by momentum score each month

COST_BPS      = 15
LOOKBACK      = 12
SKIP          = 1
LONG_Q        = 0.30
SHORT_MULT    = 1.50

TRAIN_END     = "2018-12-31"
START_DATE    = "2005-01-01"
END_DATE      = (pd.Timestamp.today().normalize() - pd.tseries.offsets.BDay(1)).date().isoformat()

VOL_TARGET    = 0.10   # macro per-position target; equity uses VOL_TARGET/2
MAX_GROSS     = 4.5

NOTIONAL_USD  = 100_000
TWS_HOST      = "127.0.0.1"
TWS_PORT      = 7497
CLIENT_ID     = 6

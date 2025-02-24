import numpy as np
from backtesting import Strategy
from backtesting import Backtest


# def SIGNAL():
#     return df.TotalSignal


class MyStrat(Strategy):
    mysize = 0.1  # Trade size
    # sl_atr_ratio = 3
    slperc = 0.04
    tpperc = 0.02

    def init(self):
        super().init()
        # self.signal1 = self.I(SIGNAL)  # Assuming SIGNAL is a function that returns signals

    def next(self):
        # super().next()
        # for trade in self.trades:
        #     sltr = self.sl_atr_ratio * self.data.atr[-1]
        #     if trade.is_long:
        #         trade.sl = max(trade.sl or -np.inf, self.data.Close[-1] - sltr)
        #     else:
        #         trade.sl = min(trade.sl or np.inf, self.data.Close[-1] + sltr)

        if self.data.TotalSignal == 2 and not self.position:
            # Open a new long position with calculated SL and TP
            current_close = self.data.Close[-1]
            sl = current_close - self.slperc * current_close  # SL at 4% below the close price
            tp = current_close + self.tpperc * current_close  # TP at 2% above the close price
            self.buy(size=self.mysize, sl=sl, tp=tp)

        elif self.data.TotalSignal == 1 and not self.position:
            # Open a new short position, setting SL based on a strategy-specific requirement
            current_close = self.data.Close[-1]
            sl = current_close + self.slperc * current_close  # SL at 4% below the close price
            tp = current_close - self.tpperc * current_close  # TP at 2% above the close price
            self.sell(size=self.mysize, sl=sl, tp=tp)



        # if self.data.TotalSignal == 2 and not self.position:
        #     # Open a new long position with calculated SL
        #     current_close = self.data.Close[-1]
        #     sl = current_close - self.sl_atr_ratio * self.data.atr[-1]  # SL below the close price
        #     # tp = current_close + self.tp_sl_ratio * (self.sl_atr_ratio * self.data.ATR[-1])  # TP above the close price
        #     self.buy(size=self.mysize, sl=sl)
        #
        # elif self.data.TotalSignal == 1 and not self.position:
        #     # short position
        #     current_close = self.data.Close[-1]
        #     sl = current_close + self.sl_atr_ratio * self.data.atr[-1]
        #     # tp = current_close - self.tp_sl_ratio * (self.sl_atr_ratio * self.data.ATR[-1])
        #     self.sell(size=self.mysize, sl=sl)
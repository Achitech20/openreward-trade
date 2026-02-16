"""Environment server for BTC 15-min trading."""
from openreward.environments import Server

from btc_trader_env import BtcTraderEnv

if __name__ == "__main__":
    server = Server([BtcTraderEnv])
    server.run()

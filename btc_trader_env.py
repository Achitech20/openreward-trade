"""
BtcTraderEnv — Polymarket BTC 15-minute trading environment.

An LLM agent trades Polymarket 15-minute BTC UP/DOWN markets.
Time auto-advances with each action. When the market expires,
positions are resolved and the reward is calculated from P&L.
"""

import csv
import math
import os
from pathlib import Path
from typing import List

from openreward import AsyncOpenReward, SandboxBucketConfig, SandboxSettings
from openreward.environments import (Environment, JSONObject, TextBlock,
                                     ToolOutput, tool)
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Tool parameter models
# ---------------------------------------------------------------------------

class EmptyParams(BaseModel, extra="forbid"):
    pass


class BashParams(BaseModel, extra="forbid"):
    command: str


class BuyParams(BaseModel, extra="forbid"):
    side: str = Field(description="UP or DOWN")
    amount: float = Field(description="Amount in USDC to spend")


class SellParams(BaseModel, extra="forbid"):
    side: str = Field(description="UP or DOWN")
    shares: float = Field(description="Number of shares to sell")


class WaitParams(BaseModel, extra="forbid"):
    minutes: float = Field(description="Minutes to wait (0.5 to 14)")


class EnvironmentSpec(BaseModel):
    task_id: str
    market_end: str = ""


# ---------------------------------------------------------------------------
# Time costs per action (in 500ms steps)
# ---------------------------------------------------------------------------

# 1 minute = 120 steps (at 500ms each)
TIME_COST = {
    "bash": 60,          # 30 seconds
    "get_market_state": 10,  # 5 seconds
    "buy": 10,           # 5 seconds
    "sell": 10,          # 5 seconds
    "wait": 0,           # custom (based on params)
}

# ---------------------------------------------------------------------------
# Market index
# ---------------------------------------------------------------------------

MARKET_INDEX_PATH = Path(__file__).resolve().parent / "data" / "markets" / "market_index.csv"
MARKETS_DIR = Path(__file__).resolve().parent / "data" / "markets"


def _load_market_index() -> List[dict]:
    if not MARKET_INDEX_PATH.exists():
        return []
    rows = []
    with open(MARKET_INDEX_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class BtcTraderEnv(Environment):
    INITIAL_BALANCE = 100.0

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)

        self.validated = EnvironmentSpec.model_validate(task_spec)
        self.task_id = self.validated.task_id
        self.market_end = self.validated.market_end

        if not secrets.get("api_key"):
            raise ValueError("OpenReward API key is required")

        self.sandbox_settings = SandboxSettings(
            environment="ranko/TraderEnv",
            image="generalreasoning/python-ds:3.12-tools",
            machine_size="0.5:1",
            block_network=False,
            bucket_config=SandboxBucketConfig(
                mount_path="/tmp/sandbox/",
                read_only=True,
                only_dir="agent",
            ),
        )

        or_client = AsyncOpenReward(api_key=secrets.get("api_key"))
        self.sandbox = or_client.sandbox(self.sandbox_settings)

        # Trading state
        self.balance = self.INITIAL_BALANCE
        self.position = {"UP": 0.0, "DOWN": 0.0}
        self.current_step = 0
        self.market_data: List[dict] = []
        self.outcome = "UNKNOWN"
        self.market_start_str = ""
        self._market_csv_header: List[str] = []

    async def setup(self) -> None:
        await self.sandbox.start()

        # Extract markets data in sandbox if zip exists on bucket mount
        await self.sandbox.run(
            "if [ -f /tmp/sandbox/markets.zip ]; then "
            "mkdir -p /home/ubuntu/data/markets && "
            "unzip -o /tmp/sandbox/markets.zip -d /home/ubuntu/data/markets/; "
            "fi"
        )

        # Find the market CSV (from local index for task routing)
        index = _load_market_index()
        market_info = None
        if self.market_end:
            for row in index:
                if row["market_end"] == self.market_end:
                    market_info = row
                    break
        else:
            task_idx = int(self.task_id)
            if 0 <= task_idx < len(index):
                market_info = index[task_idx]
                self.market_end = market_info["market_end"]

        if not market_info:
            raise ValueError(f"Market not found: task_id={self.task_id}, market_end={self.market_end}")

        self.outcome = market_info.get("outcome", "UNKNOWN")
        self.market_start_str = market_info.get("market_start", "")

        # Load market data from local CSV (server-side, for state tracking)
        csv_file = market_info.get("csv_file", "")
        csv_path = MARKETS_DIR / csv_file
        if not csv_path.exists():
            raise ValueError(f"Market CSV not found: {csv_path}")

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            self.market_data = list(reader)

        if not self.market_data:
            raise ValueError(f"Empty market CSV: {csv_path}")

        # Prepare CSV header (without outcome column)
        self._market_csv_header = [f for f in self.market_data[0].keys() if f != "outcome"]

        # Upload current market CSV truncated to step 0 (no outcome, no future)
        await self._sync_market_csv()

        # Build history CSV from all markets before current one and upload
        # (agent can analyze past outcomes to learn patterns)
        history_rows = []
        for row in index:
            if row["market_end"] >= self.market_end:
                break
            hist_csv_path = MARKETS_DIR / row.get("csv_file", "")
            if hist_csv_path.exists():
                hist_content = hist_csv_path.read_text().strip().split("\n")
                if not history_rows:
                    history_rows.append(hist_content[0])  # header
                history_rows.extend(hist_content[1:])

        if history_rows:
            history_content = "\n".join(history_rows) + "\n"
            await self._upload_file("/home/ubuntu/data/history.csv", history_content)

    async def teardown(self) -> None:
        await self.sandbox.stop()

    # -------------------------------------------------------------------
    # Time & state helpers
    # -------------------------------------------------------------------

    def _is_expired(self) -> bool:
        return self.current_step >= len(self.market_data) - 1

    def _advance_steps(self, steps: int) -> int:
        """Advance time by N steps. Returns actual steps advanced."""
        old = self.current_step
        self.current_step = min(self.current_step + steps, len(self.market_data) - 1)
        return self.current_step - old

    def _current_row(self) -> dict:
        idx = min(self.current_step, len(self.market_data) - 1)
        return self.market_data[idx]

    def _get_price(self, row: dict, side: str, price_type: str) -> float:
        key = f"{side.lower()}_{price_type}"
        try:
            return float(row.get(key, 0))
        except (ValueError, TypeError):
            return 0.0

    def _portfolio_value(self, row: dict = None) -> float:
        if row is None:
            row = self._current_row()
        value = self.balance
        for side in ["UP", "DOWN"]:
            shares = self.position[side]
            if shares > 0:
                bid = self._get_price(row, side, "bid")
                value += shares * bid
        return value

    def _resolve_positions(self) -> float:
        """Resolve all positions at market expiry. Returns cash from winning shares."""
        cash = 0.0
        for side in ["UP", "DOWN"]:
            shares = self.position[side]
            if shares > 0:
                if side == self.outcome:
                    cash += shares * 1.0  # $1 per winning share
                self.position[side] = 0.0
        return cash

    def _calculate_reward(self) -> float:
        """Sigmoid reward from trading P&L. Break-even = 0.25, uncapped."""
        trading_pnl = self.balance - self.INITIAL_BALANCE
        return 0.5 / (1 + math.exp(-trading_pnl / 25.0))

    def _format_state(self) -> str:
        row = self._current_row()
        total_steps = len(self.market_data)
        pct = (self.current_step / max(total_steps - 1, 1)) * 100

        lines = [
            f"=== Market State ({pct:.0f}% elapsed) ===",
            f"Time: {row.get('ts', 'N/A')}",
        ]

        mins_elapsed = row.get("minutes_elapsed", "")
        mins_remaining = row.get("minutes_remaining", "")
        if mins_elapsed:
            lines.append(f"Minutes elapsed: {float(mins_elapsed):.1f} | Remaining: {float(mins_remaining):.1f}")

        lines.append("")
        lines.append("--- Prices ---")
        for side in ["UP", "DOWN"]:
            bid = self._get_price(row, side, "bid")
            ask = self._get_price(row, side, "ask")
            mid = self._get_price(row, side, "mid")
            lines.append(f"  {side}: bid=${bid:.4f}  ask=${ask:.4f}  mid=${mid:.4f}")

        btc_price = row.get("btc_price", "")
        if btc_price:
            lines.append(f"\nBTC Price: ${float(btc_price):,.2f}")
            btc_1m = row.get("btc_price_change_1m", "")
            btc_5m = row.get("btc_price_change_5m", "")
            if btc_1m:
                lines.append(f"  1m change: ${float(btc_1m):+,.2f}")
            if btc_5m:
                lines.append(f"  5m change: ${float(btc_5m):+,.2f}")

        analytics_parts = []
        for field in ["prob_up", "prob_down", "z_score", "vol_15m"]:
            val = row.get(field, "")
            if val:
                analytics_parts.append(f"{field}={float(val):.4f}")
        if analytics_parts:
            lines.append(f"\nAnalytics: {', '.join(analytics_parts)}")

        lines.append("")
        lines.append("--- Portfolio ---")
        lines.append(f"  Cash: ${self.balance:.2f}")
        for side in ["UP", "DOWN"]:
            shares = self.position[side]
            if shares > 0:
                bid = self._get_price(row, side, "bid")
                lines.append(f"  {side} shares: {shares:.2f} (value ~ ${shares * bid:.2f})")
        lines.append(f"  Total value: ${self._portfolio_value(row):.2f}")
        lines.append(f"  P&L: ${self._portfolio_value(row) - self.INITIAL_BALANCE:+.2f}")

        return "\n".join(lines)

    def _make_expiry_result(self) -> ToolOutput:
        """Resolve the market and return final ToolOutput with finished=True."""
        resolution_cash = self._resolve_positions()
        self.balance += resolution_cash
        trading_pnl = self.balance - self.INITIAL_BALANCE
        reward = self._calculate_reward()

        lines = [
            "",
            "=" * 50,
            "MARKET EXPIRED — Auto-resolving positions",
            "=" * 50,
            f"Outcome: {self.outcome}",
            f"Winning shares paid out: ${resolution_cash:.2f}",
            f"Final balance: ${self.balance:.2f}",
            f"Trading P&L: ${trading_pnl:+.2f}",
            f"Reward: {reward:.4f}",
        ]

        return ToolOutput(
            blocks=[TextBlock(text="\n".join(lines))],
            metadata={
                "outcome": self.outcome,
                "trading_pnl": trading_pnl,
                "final_balance": self.balance,
                "reward": reward,
            },
            reward=reward,
            finished=True,
        )

    async def _tick(self, tool_name: str, extra_steps: int = 0) -> None:
        """Advance time for a tool call and sync market CSV."""
        steps = TIME_COST.get(tool_name, 10) + extra_steps
        self._advance_steps(steps)
        await self._sync_market_csv()

    # -------------------------------------------------------------------
    # File helpers
    # -------------------------------------------------------------------

    async def _sync_market_csv(self) -> None:
        """Upload market.csv with data only up to current_step (no outcome)."""
        header = ",".join(self._market_csv_header)
        data_lines = [header]
        for i in range(self.current_step + 1):
            if i < len(self.market_data):
                vals = ",".join(str(self.market_data[i].get(f, "")) for f in self._market_csv_header)
                data_lines.append(vals)
        content = "\n".join(data_lines) + "\n"
        await self._upload_file("/home/ubuntu/data/market.csv", content)

    async def _upload_file(self, remote_path: str, content: str) -> None:
        dir_path = "/".join(remote_path.split("/")[:-1])
        cmd = f"mkdir -p {dir_path} && cat > {remote_path} << 'CSVEOF'\n{content}\nCSVEOF"
        output, code = await self.sandbox.run(cmd)
        if code != 0:
            lines = content.strip().split("\n")
            await self.sandbox.run(f"mkdir -p {dir_path} && echo '{lines[0]}' > {remote_path}")
            for line in lines[1:]:
                if line.strip():
                    escaped = line.replace("'", "'\\''")
                    await self.sandbox.run(f"echo '{escaped}' >> {remote_path}")

    # -------------------------------------------------------------------
    # Tools
    # -------------------------------------------------------------------

    @tool
    async def bash(self, params: BashParams) -> ToolOutput:
        """Run a bash command in the sandbox. Use this to analyze market data. Costs 30s of market time."""
        output, code = await self.sandbox.run(params.command.strip())
        result_text = f"{output}\n\n(exit {code})"

        await self._tick("bash")

        if self._is_expired():
            expiry = self._make_expiry_result()
            result_text += "\n" + expiry.blocks[0].text
            return ToolOutput(
                blocks=[TextBlock(text=result_text)],
                metadata=expiry.metadata,
                reward=expiry.reward,
                finished=True,
            )

        return ToolOutput(
            blocks=[TextBlock(text=result_text)],
            metadata={"exit_code": code},
            reward=0.0,
            finished=False,
        )

    @tool
    async def get_market_state(self, params: EmptyParams) -> ToolOutput:
        """See current prices, position, balance, and P&L. Costs 5s of market time."""
        await self._tick("get_market_state")

        if self._is_expired():
            return self._make_expiry_result()

        state = self._format_state()
        return ToolOutput(
            blocks=[TextBlock(text=state)],
            metadata={"step": self.current_step},
            reward=0.0,
            finished=False,
        )

    @tool
    async def buy(self, params: BuyParams) -> ToolOutput:
        """Buy UP or DOWN shares at the current ask price. Amount in USDC. Costs 5s of market time."""
        side = params.side.upper()
        if side not in ("UP", "DOWN"):
            return ToolOutput(
                blocks=[TextBlock(text="Error: side must be 'UP' or 'DOWN'")],
                metadata={}, reward=0.0, finished=False,
            )

        amount = params.amount
        if amount <= 0:
            return ToolOutput(
                blocks=[TextBlock(text="Error: amount must be positive")],
                metadata={}, reward=0.0, finished=False,
            )

        if amount > self.balance:
            return ToolOutput(
                blocks=[TextBlock(text=f"Error: insufficient balance. Have ${self.balance:.2f}, need ${amount:.2f}")],
                metadata={}, reward=0.0, finished=False,
            )

        row = self._current_row()
        ask = self._get_price(row, side, "ask")
        if ask <= 0:
            return ToolOutput(
                blocks=[TextBlock(text=f"Error: no ask price for {side}")],
                metadata={}, reward=0.0, finished=False,
            )

        shares = amount / ask
        self.balance -= amount
        self.position[side] += shares

        await self._tick("buy")

        result = (
            f"Bought {shares:.2f} {side} shares at ${ask:.4f} each (cost: ${amount:.2f})\n"
            f"Position: {self.position[side]:.2f} {side} shares | Cash: ${self.balance:.2f}"
        )

        if self._is_expired():
            expiry = self._make_expiry_result()
            result += "\n" + expiry.blocks[0].text
            return ToolOutput(
                blocks=[TextBlock(text=result)],
                metadata=expiry.metadata,
                reward=expiry.reward,
                finished=True,
            )

        return ToolOutput(
            blocks=[TextBlock(text=result)],
            metadata={"side": side, "shares": shares, "price": ask, "cost": amount},
            reward=0.0,
            finished=False,
        )

    @tool
    async def sell(self, params: SellParams) -> ToolOutput:
        """Sell UP or DOWN shares at the current bid price. Costs 5s of market time."""
        side = params.side.upper()
        if side not in ("UP", "DOWN"):
            return ToolOutput(
                blocks=[TextBlock(text="Error: side must be 'UP' or 'DOWN'")],
                metadata={}, reward=0.0, finished=False,
            )

        shares = params.shares
        if shares <= 0:
            return ToolOutput(
                blocks=[TextBlock(text="Error: shares must be positive")],
                metadata={}, reward=0.0, finished=False,
            )

        # Clamp to actual position to avoid floating point issues
        if shares > self.position[side] + 0.001:
            return ToolOutput(
                blocks=[TextBlock(text=f"Error: you only have {self.position[side]:.2f} {side} shares")],
                metadata={}, reward=0.0, finished=False,
            )
        shares = min(shares, self.position[side])

        row = self._current_row()
        bid = self._get_price(row, side, "bid")
        if bid <= 0:
            return ToolOutput(
                blocks=[TextBlock(text=f"Error: no bid price for {side}")],
                metadata={}, reward=0.0, finished=False,
            )

        proceeds = shares * bid
        self.position[side] -= shares
        self.balance += proceeds

        await self._tick("sell")

        result = (
            f"Sold {shares:.2f} {side} shares at ${bid:.4f} each (proceeds: ${proceeds:.2f})\n"
            f"Position: {self.position[side]:.2f} {side} shares | Cash: ${self.balance:.2f}"
        )

        if self._is_expired():
            expiry = self._make_expiry_result()
            result += "\n" + expiry.blocks[0].text
            return ToolOutput(
                blocks=[TextBlock(text=result)],
                metadata=expiry.metadata,
                reward=expiry.reward,
                finished=True,
            )

        return ToolOutput(
            blocks=[TextBlock(text=result)],
            metadata={"side": side, "shares": shares, "price": bid, "proceeds": proceeds},
            reward=0.0,
            finished=False,
        )

    @tool
    async def wait(self, params: WaitParams) -> ToolOutput:
        """Wait and observe the market for the specified number of minutes. Market data updates."""
        minutes = params.minutes
        if minutes <= 0:
            return ToolOutput(
                blocks=[TextBlock(text="Error: minutes must be positive")],
                metadata={}, reward=0.0, finished=False,
            )

        steps = int(minutes * 120)
        actual = self._advance_steps(steps)
        actual_minutes = actual / 120.0

        await self._sync_market_csv()

        result = f"Waited {actual_minutes:.1f} minutes.\n\n{self._format_state()}"

        if self._is_expired():
            expiry = self._make_expiry_result()
            result += "\n" + expiry.blocks[0].text
            return ToolOutput(
                blocks=[TextBlock(text=result)],
                metadata=expiry.metadata,
                reward=expiry.reward,
                finished=True,
            )

        return ToolOutput(
            blocks=[TextBlock(text=result)],
            metadata={"minutes_waited": actual_minutes},
            reward=0.0,
            finished=False,
        )

    # -------------------------------------------------------------------
    # Prompt and tasks
    # -------------------------------------------------------------------

    async def get_prompt(self) -> List[TextBlock]:
        initial_state = self._format_state()

        prompt = f"""You are a Polymarket BTC 15-minute trader.

Market: BTC will go UP or DOWN in the next 15 minutes.
Market opens: {self.market_start_str} | Market closes: {self.market_end}

You start with $100 USDC. You can buy UP or DOWN shares at current market prices.
- UP shares pay $1 if BTC goes up, $0 if down.
- DOWN shares pay $1 if BTC goes down, $0 if up.

IMPORTANT: Time passes with every action you take!
- `bash` costs 30 seconds
- `buy`, `sell`, `get_market_state` cost 5 seconds each
- `wait(minutes)` skips forward by that many minutes
When time runs out, the market auto-resolves: winning shares pay $1, losing shares pay $0.

Available tools:
- `get_market_state` — see current prices, position, P&L
- `buy(side, amount)` — buy shares (side="UP" or "DOWN", amount in USDC)
- `sell(side, shares)` — sell shares you hold
- `wait(minutes)` — skip forward to observe price changes
- `bash(command)` — run bash to analyze data files

Data files in /home/ubuntu/data/:
- market.csv — current market data up to now (updates as time passes, NO outcome column)
  Columns: ts, minutes_elapsed, minutes_remaining, up_bid, up_ask, up_mid, down_bid, down_ask, down_mid,
  btc_price, btc_price_change_1m, btc_price_change_5m, vol_15m, prob_up, prob_down, z_score
- history.csv — previous resolved markets (WITH outcome column) for learning patterns

Your goal: maximize profit. Buy shares on the side you think will win.
The reward is based purely on your trading P&L.

Current state:
{initial_state}"""

        return [TextBlock(text=prompt)]

    @classmethod
    def list_tasks(cls, split: str) -> List[JSONObject]:
        index = _load_market_index()
        if not index:
            return []

        valid = [r for r in index if r.get("outcome") in ("UP", "DOWN")]

        if split == "test":
            test_markets = valid[-50:]
            return [
                {"task_id": str(i), "market_end": m["market_end"]}
                for i, m in enumerate(test_markets)
            ]
        elif split == "train":
            train_markets = valid[:-50] if len(valid) > 50 else []
            return [
                {"task_id": str(i), "market_end": m["market_end"]}
                for i, m in enumerate(train_markets)
            ]
        else:
            raise ValueError(f"Unknown split: {split}")

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train", "test"]

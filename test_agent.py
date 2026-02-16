import argparse
import json
import asyncio
import logging
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

import anthropic
from openreward import AsyncOpenReward


def setup_logging(log_dir="logs"):
    """Setup dual logging: compact to console, full detail to file."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_{timestamp}.log")

    # File logger — full detail
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))

    logger = logging.getLogger("trader")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    return logger, log_file


def print_and_log(logger, msg, level="info"):
    """Print to console and log to file."""
    print(msg)
    getattr(logger, level)(msg)


def log_only(logger, msg):
    """Log to file only (verbose detail)."""
    logger.debug(msg)


def print_separator(char="=", width=70):
    return char * width


async def main():
    parser = argparse.ArgumentParser(description="Run BTC trader agent")
    parser.add_argument("-n", "--num-tasks", type=int, default=1, help="Number of tasks to run (default: 1)")
    parser.add_argument("--split", default="test", help="Task split (default: test)")
    parser.add_argument("--model", default="claude-sonnet-4-5", help="Model name")
    parser.add_argument("--log-dir", default="logs", help="Log directory (default: logs)")
    args = parser.parse_args()

    logger, log_file = setup_logging(args.log_dir)

    or_client = AsyncOpenReward()

    MODEL_NAME = args.model
    ant_client = anthropic.AsyncAnthropic()

    ENV_NAME = "BtcTraderEnv"
    SPLIT = args.split
    OR_API_KEY = os.getenv("OPENREWARD_API_KEY")

    environment = or_client.environments.get(name=ENV_NAME, base_url="http://localhost:8080")
    tasks = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="anthropic")

    num_to_run = min(args.num_tasks, len(tasks))
    header = print_separator()
    print_and_log(logger, header)
    print_and_log(logger, f"  BTC Trader Agent | Model: {MODEL_NAME}")
    print_and_log(logger, f"  Tasks: {num_to_run}/{len(tasks)} ({SPLIT} split)")
    print_and_log(logger, f"  Log: {log_file}")
    print_and_log(logger, header)

    # Log tools to file
    log_only(logger, f"Tools: {[t.get('name', t) if isinstance(t, dict) else getattr(t, 'name', str(t)) for t in tools]}")

    # Cumulative tracking
    results = []

    for task_idx, task in enumerate(tasks[:num_to_run]):
        if isinstance(task, dict):
            market_end = task.get("market_end", task.get("task_id", "?"))
        elif hasattr(task, "task_spec"):
            market_end = task.task_spec.get("market_end", task.task_spec.get("task_id", "?"))
        else:
            market_end = getattr(task, "market_end", getattr(task, "task_id", "?"))

        print_and_log(logger, f"\n{'='*70}")
        print_and_log(logger, f"  MARKET {task_idx + 1}/{num_to_run} | {market_end}")
        print_and_log(logger, f"{'='*70}")

        async with environment.session(
            task=task,
            secrets={"api_key": OR_API_KEY}
        ) as session:
            prompt = await session.get_prompt()
            messages = [{"role": "user", "content": prompt[0].text}]
            finished = False
            turn = 0

            # Log full prompt to file
            log_only(logger, f"PROMPT:\n{prompt[0].text}")

            while not finished:
                turn += 1
                response = await ant_client.messages.create(
                    model=MODEL_NAME,
                    max_tokens=4096,
                    tools=tools,
                    messages=messages
                )

                messages.append({"role": "assistant", "content": response.content})
                log_only(logger, f"--- Turn {turn} | stop_reason={response.stop_reason} ---")

                tool_results = []
                for block in response.content:
                    if block.type == "text":
                        # Console: short preview
                        text_preview = block.text[:150].replace('\n', ' ')
                        print_and_log(logger, f"  [{turn}] Agent: {text_preview}{'...' if len(block.text) > 150 else ''}")
                        # File: full text
                        log_only(logger, f"  FULL TEXT:\n{block.text}")

                    elif block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input

                        # Log full tool call to file
                        log_only(logger, f"  TOOL CALL: {tool_name}({json.dumps(tool_input)})")

                        # Console: compact action summary
                        if tool_name == "buy":
                            action = f"BUY {tool_input.get('amount', '?')} USDC of {tool_input.get('side', '?')}"
                        elif tool_name == "sell":
                            action = f"SELL {tool_input.get('shares', '?')} {tool_input.get('side', '?')} shares"
                        elif tool_name == "advance_time":
                            action = f"ADVANCE {tool_input.get('minutes', '?')} min"
                        elif tool_name == "submit_prediction":
                            action = f"PREDICT {tool_input.get('prediction', '?')}"
                        elif tool_name == "get_market_state":
                            action = "GET STATE"
                        elif tool_name == "bash":
                            cmd = tool_input.get("command", "")
                            cmd_preview = cmd[:60].replace('\n', ' ')
                            action = f"BASH: {cmd_preview}{'...' if len(cmd) > 60 else ''}"
                        else:
                            action = f"{tool_name}({json.dumps(tool_input)[:60]})"

                        tool_result = await session.call_tool(
                            block.name,
                            block.input
                        )

                        reward = tool_result.reward
                        finished = tool_result.finished
                        output_text = tool_result.blocks[0].text

                        # Log full tool result to file
                        log_only(logger, f"  TOOL RESULT ({tool_name}): reward={reward} finished={finished}")
                        log_only(logger, f"  FULL OUTPUT:\n{output_text}")

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": output_text
                        })

                        # Console: compact summary per tool type
                        if tool_name == "submit_prediction":
                            print_and_log(logger, f"  [{turn}] >> {action}")
                            for line in output_text.split("\n"):
                                if line.strip():
                                    print_and_log(logger, f"       {line}")
                        elif tool_name in ("buy", "sell"):
                            first_line = output_text.split("\n")[0]
                            print_and_log(logger, f"  [{turn}] >> {action}  ->  {first_line}")
                        elif tool_name == "advance_time":
                            lines = output_text.split("\n")
                            summary_parts = [lines[0]]
                            for line in lines:
                                if "P&L:" in line:
                                    summary_parts.append(line.strip())
                                elif "Total value:" in line:
                                    summary_parts.append(line.strip())
                            print_and_log(logger, f"  [{turn}] >> {action}  ->  {' | '.join(summary_parts)}")
                        elif tool_name == "get_market_state":
                            state_parts = []
                            for line in output_text.split("\n"):
                                if "Step " in line:
                                    state_parts.append(line.strip().replace("=== ", "").replace(" ===", ""))
                                elif "P&L:" in line:
                                    state_parts.append(line.strip())
                                elif "Cash:" in line:
                                    state_parts.append(line.strip())
                            print_and_log(logger, f"  [{turn}] >> {action}  ->  {' | '.join(state_parts[:3])}")
                        else:
                            out_preview = output_text[:100].replace('\n', ' ')
                            print_and_log(logger, f"  [{turn}] >> {action}")
                            print_and_log(logger, f"       -> {out_preview}{'...' if len(output_text) > 100 else ''}")

                        if finished:
                            # Print expiry summary
                            meta = tool_result.metadata or {}
                            pnl = meta.get("trading_pnl", 0)
                            outcome = meta.get("outcome", "?")
                            final_bal = meta.get("final_balance", 0)
                            print_and_log(logger, f"  --- MARKET EXPIRED ---")
                            print_and_log(logger, f"  Outcome: {outcome} | Final balance: ${final_bal:.2f} | P&L: ${pnl:+.2f} | Reward: {reward:.4f}")
                            results.append({
                                "market": market_end,
                                "outcome": outcome,
                                "pnl": pnl,
                                "final_balance": final_bal,
                                "reward": reward,
                                "turns": turn,
                            })
                            break

                if finished:
                    break

                if response.stop_reason == "end_turn" and not tool_results:
                    print_and_log(logger, f"  [{turn}] Agent stopped (no tool calls).")
                    results.append({
                        "market": market_end,
                        "prediction": "NONE",
                        "outcome": "?",
                        "correct": False,
                        "pnl": 0,
                        "final_balance": 100,
                        "reward": 0,
                        "turns": turn,
                    })
                    break

                if tool_results:
                    messages.append({"role": "user", "content": tool_results})

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------
    if results:
        print_and_log(logger, f"\n{'='*70}")
        print_and_log(logger, f"  SUMMARY ({len(results)} markets)")
        print_and_log(logger, f"{'='*70}")
        print_and_log(logger, f"  {'Market':<24} {'Outcome':>7} {'P&L':>9} {'Balance':>9} {'Reward':>7} {'Turns':>5}")
        print_and_log(logger, f"  {'-'*24} {'-'*7} {'-'*9} {'-'*9} {'-'*7} {'-'*5}")

        total_pnl = 0
        total_reward = 0

        for r in results:
            market_short = r["market"][-19:] if len(r["market"]) > 19 else r["market"]
            print_and_log(logger, f"  {market_short:<24} {r['outcome']:>7} "
                  f"${r['pnl']:>+8.2f} ${r['final_balance']:>8.2f} {r['reward']:>7.4f} {r['turns']:>5}")
            total_pnl += r["pnl"]
            total_reward += r["reward"]

        print_and_log(logger, f"  {'-'*24} {'-'*7} {'-'*9} {'-'*9} {'-'*7} {'-'*5}")
        print_and_log(logger, f"  {'TOTAL':<24} {'':>7} "
              f"${total_pnl:>+8.2f} {'':>9} {total_reward:>7.4f}")
        print_and_log(logger, f"  Avg reward: {total_reward / len(results):.4f}")
        print_and_log(logger, print_separator())

    print(f"\n  Full log: {log_file}")


if __name__ == "__main__":
    asyncio.run(main())

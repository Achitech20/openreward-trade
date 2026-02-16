import json
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

import anthropic
from openreward import AsyncOpenReward

async def main():
    or_client = AsyncOpenReward()

    MODEL_NAME = "claude-sonnet-4-5"
    ant_client = anthropic.AsyncAnthropic()

    ENV_NAME = "AccountantEnv"
    SPLIT = "test"
    OR_API_KEY = os.getenv("OPENREWARD_API_KEY")

    environment = or_client.environments.get(name=ENV_NAME, base_url="http://localhost:8080")
    tasks = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="anthropic")

    print(f"Found {len(tasks)} tasks")

    # Test first scenario
    for task in tasks[:1]:
        async with environment.session(
            task=task,
            secrets={"api_key": OR_API_KEY}
        ) as session:
            prompt = await session.get_prompt()
            messages = [{"role": "user", "content": prompt[0].text}]
            finished = False

            print(f"\n=== Initial Prompt ===")
            print(f"Role: {messages[0]['role']}")
            print(f"Content: {messages[0]['content'][:100]}..." if len(messages[0]['content']) > 100 else f"Content: {messages[0]['content']}")
            print(f"Finished: {finished}\n")

            while not finished:
                response = await ant_client.messages.create(
                    model=MODEL_NAME,
                    max_tokens=4096,
                    tools=tools,
                    messages=messages
                )

                # Add assistant response to message history
                messages.append({"role": "assistant", "content": response.content})

                print(f"=== Model Response ===")
                print(f"Stop reason: {response.stop_reason}")

                tool_results = []
                for block in response.content:
                    if block.type == "text":
                        print(f"Text: {block.text[:100]}..." if len(block.text) > 100 else f"Text: {block.text}")
                    elif block.type == "tool_use":
                        print(f"Tool: {block.name}")
                        print(f"Arguments: {json.dumps(block.input)}")

                        tool_result = await session.call_tool(
                            block.name,
                            block.input
                        )

                        reward = tool_result.reward
                        finished = tool_result.finished

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": tool_result.blocks[0].text
                        })

                        print(f"=== Tool Result ===")
                        print(f"Tool Use ID: {block.id}")
                        output_text = tool_result.blocks[0].text
                        print(f"Output: {output_text[:100]}..." if len(output_text) > 100 else f"Output: {output_text}")
                        print(f"Reward: {reward:.4f} | Finished: {finished}\n")

                        if finished:
                            print(f"FINISHED!")
                            break

                if finished:
                    break

                # If the model stopped without calling tools, it's done
                if response.stop_reason == "end_turn" and not tool_results:
                    print("Agent finished (no more tool calls).")
                    break

                # Add tool results as a user message
                if tool_results:
                    messages.append({"role": "user", "content": tool_results})

if __name__ == "__main__":
    asyncio.run(main())

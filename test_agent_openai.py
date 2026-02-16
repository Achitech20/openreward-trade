import json
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from openai import AsyncOpenAI
from openreward import AsyncOpenReward

async def main():
    or_client = AsyncOpenReward()

    MODEL_NAME = "gpt-4o"
    oai_client = AsyncOpenAI()

    ENV_NAME = "SandboxEnvironment"
    SPLIT = "test"
    OR_API_KEY = os.getenv("OPENREWARD_API_KEY")

    environment = or_client.environments.get(name=ENV_NAME, base_url="http://localhost:8080")
    tasks = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="openai")

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
                response = await oai_client.responses.create(
                    model=MODEL_NAME,
                    tools=tools,
                    input=messages
                )

                last_output = response.output[-1]
                print(f"=== Model Response ===")
                print(f"Type: {last_output.type}")
                if hasattr(last_output, 'name'):
                    print(f"Function: {last_output.name}")
                if hasattr(last_output, 'content'):
                    print(f"Content: {last_output.content[:100]}..." if len(str(last_output.content)) > 100 else f"Content: {last_output.content}")
                print()
                messages += response.output

                for item in response.output:
                    if item.type == "function_call":
                        print(f"Tool: {item.name}")
                        print(f"Arguments: {item.arguments}")

                        tool_result = await session.call_tool(
                            item.name,
                            json.loads(str(item.arguments))
                        )

                        reward = tool_result.reward
                        finished = tool_result.finished

                        messages.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": tool_result.blocks[0].text
                        })

                        print(f"=== Tool Result ===")
                        print(f"Call ID: {item.call_id}")
                        output_text = tool_result.blocks[0].text
                        print(f"Output: {output_text[:100]}..." if len(output_text) > 100 else f"Output: {output_text}")
                        print(f"Reward: {reward:.4f} | Finished: {finished}\n")

                        if finished:
                            print(f"FINISHED!")
                            break

if __name__ == "__main__":
    asyncio.run(main())

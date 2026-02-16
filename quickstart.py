import anthropic
from openreward import OpenReward
import json

or_client = OpenReward()
ant_client = anthropic.Anthropic()
MODEL_NAME = "claude-sonnet-4-5"

environment = or_client.environments.get(name="GeneralReasoning/KellyBench")
tasks = environment.list_tasks(split="train")
tools = environment.list_tools(format="anthropic")

example_task = tasks[0]

with environment.session(task=example_task) as session:
    prompt = session.get_prompt()
    messages = [{"role": "user", "content": prompt[0].text}]
    finished = False
    print(messages)

    while not finished:
        message = ant_client.messages.create(
            model=MODEL_NAME,
            max_tokens=4096,
            tools=tools,
            messages=messages
        )

        messages.append({
            "role": "assistant",
            "content": message.content,
        })

        print(messages[-1])

        if message.stop_reason == "tool_use":
            tool_uses = [b for b in message.content if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:
                raise RuntimeError("stop_reason was tool_use but no tool_use blocks found")

            tool_result_blocks = []
            finished = False

            for tu in tool_uses:
                tool_name = tu.name
                tool_input = tu.input

                try:
                    tr = session.call_tool(tool_name, tool_input)
                    # Convert OpenReward blocks -> string safely
                    text_parts = []
                    for b in getattr(tr, "blocks", []) or []:
                        t = getattr(b, "text", None)
                        if t is not None:
                            text_parts.append(t)
                        else:
                            text_parts.append(str(b))
                    tool_text = "".join(text_parts)

                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": tool_text,
                    })

                    # Track termination if the env says we're done
                    if getattr(tr, "finished", False):
                        finished = True

                except Exception as e:
                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": f"Tool execution failed: {type(e).__name__}: {e}",
                        "is_error": True,
                    })

            messages.append({
                "role": "user",
                "content": tool_result_blocks
            })

            print(messages[-1])
            continue
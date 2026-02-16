from typing import List

from openreward import AsyncOpenReward, SandboxBucketConfig, SandboxSettings
from openreward.environments import (Environment, JSONObject, TextBlock,
                                     ToolOutput, tool)
from pydantic import BaseModel


class BashParams(BaseModel, extra="forbid"):
    command: str

class SubmitAnswerParams(BaseModel, extra="forbid"):
    answer: int

class EnvironmentSpec(BaseModel):
    task_id: str


class AccountantEnv(Environment):
    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)

        self.validated = EnvironmentSpec.model_validate(task_spec)
        self.task_id = self.validated.task_id

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
                only_dir="agent"
            )
        )

        or_client = AsyncOpenReward(api_key=secrets.get("api_key"))
        self.sandbox = or_client.sandbox(self.sandbox_settings)

    async def setup(self) -> None:
        await self.sandbox.start()

        # Download dataset CSV from Google Drive into the sandbox
        file_id = "1Bw5eoHy2-A6VP21PbiWaehApHWUm5r2O"
        download_cmd = (
            f'mkdir -p /home/ubuntu/data && '
            f'curl -L "https://drive.google.com/uc?export=download&id={file_id}" '
            f'-o /home/ubuntu/data/transactions.csv'
        )
        output, code = await self.sandbox.run(download_cmd)
        if code != 0:
            print(f"Warning: dataset download failed (exit {code}): {output}")

    async def teardown(self) -> None:
        await self.sandbox.stop()

    @tool
    async def bash(self, params: BashParams) -> ToolOutput:
        """Executes a bash command in the environment."""

        output, code = await self.sandbox.run(params.command.strip())

        return ToolOutput(
            blocks=[TextBlock(text=f"{output}\n\n(exit {code})")],
            metadata={"output": output, "exit_code": code},
            reward=0.0,
            finished=False,
        )

    @tool
    async def submit_answer(self, params: SubmitAnswerParams) -> ToolOutput:
        """Submit an integer answer for the current task."""

        # Define correct answers for each task
        correct_answers = {
            "0": 1625  # Balance for transactions.csv
        }

        correct = correct_answers.get(self.task_id)
        is_correct = params.answer == correct

        result_text = f"Submitted answer: {params.answer}"
        if is_correct:
            result_text += "\nCorrect! Task completed."
            reward = 1.0
        else:
            result_text += f"\nIncorrect. Expected: {correct}"
            reward = 0.0

        return ToolOutput(
            blocks=[TextBlock(text=result_text)],
            metadata={"answer": params.answer, "correct": is_correct},
            reward=reward,
            finished=True,
        )

    async def get_prompt(self) -> List[TextBlock]:
        """Return the challenge prompt."""
        # FOLDER = '/tmp/sandbox/'
        FOLDER = ' /home/ubuntu/data/'
        
        if self.task_id == "0":
            full_prompt = f"""You are an accountant assistant. Your task is to calculate the final balance from a transactions spreadsheet.

The file is located at: {FOLDER}/transactions.csv

Calculate the total balance and submit your answer using the submit_answer tool with an integer (dollars).
"""
        else:
            full_prompt = f"""You are participating in task {self.task_id}. Please explore the file system. Check {FOLDER}"""

        return [TextBlock(text=full_prompt)]

    @classmethod
    def list_tasks(cls, split: str) -> List[JSONObject]:
        """Get all available tasks."""

        if split == "test":
            return [{"task_id": "0"}]
        elif split == "train":
            return []  # No training tasks
        else:
            raise ValueError(f"Unknown split: {split}")

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train", "test"]
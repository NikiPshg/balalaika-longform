from __future__ import annotations

from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

import yaml


TRL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TRL_ROOT.parents[1]
LAUNCH = TRL_ROOT / "launch.sh"
VENV_PYTHON = REPO_ROOT / "lora_finetuning" / ".venv" / "bin" / "python"
SFT_CONFIG = "lora_finetuning/trl/configs/sft_full.yaml"
GRPO_CONFIG = "lora_finetuning/trl/configs/grpo.yaml"


def run_launcher(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run only launcher's validation/dry-run path from outside the repository."""

    with tempfile.TemporaryDirectory() as working_directory:
        return subprocess.run(
            ["bash", str(LAUNCH), *arguments],
            cwd=working_directory,
            text=True,
            capture_output=True,
            check=False,
        )


def dry_run_command(result: subprocess.CompletedProcess[str]) -> tuple[str, list[str]]:
    lines = result.stdout.splitlines()
    if len(lines) != 2:
        raise AssertionError(
            f"expected cwd and command lines, got stdout={result.stdout!r}, "
            f"stderr={result.stderr!r}"
        )
    return lines[0], shlex.split(lines[1])


class LocalLauncherDryRunTests(unittest.TestCase):
    def test_single_gpu_sft_uses_project_venv_and_repo_working_directory(self) -> None:
        result = run_launcher("--dry-run", "sft", SFT_CONFIG)
        self.assertEqual(result.returncode, 0, result.stderr)
        cwd_line, command = dry_run_command(result)

        self.assertEqual(cwd_line, f"cd {REPO_ROOT}")
        self.assertEqual(command[0], str(VENV_PYTHON))
        self.assertEqual(command[1:4], ["-m", "accelerate.commands.launch", "--config_file"])
        self.assertEqual(
            command[4], str(TRL_ROOT / "configs" / "accelerate" / "single_gpu.yaml")
        )
        self.assertNotIn("--num_processes", command)
        self.assertEqual(command[5], str(TRL_ROOT / "sft.py"))
        self.assertEqual(command[6:8], ["--config", str(REPO_ROOT / SFT_CONFIG)])

    def test_multi_gpu_grpo_forwards_process_count_and_trainer_overrides(self) -> None:
        result = run_launcher(
            "--accelerate-config",
            "multi",
            "--num-processes",
            "2",
            "--dry-run",
            "grpo",
            GRPO_CONFIG,
            "--max_steps",
            "5",
            "--dataset_name",
            "lab260/youtube_balalaika",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        cwd_line, command = dry_run_command(result)

        self.assertEqual(cwd_line, f"cd {REPO_ROOT}")
        self.assertEqual(
            command[4], str(TRL_ROOT / "configs" / "accelerate" / "multi_gpu.yaml")
        )
        process_index = command.index("--num_processes")
        self.assertEqual(command[process_index + 1], "2")
        entrypoint_index = command.index(str(TRL_ROOT / "grpo.py"))
        self.assertEqual(
            command[entrypoint_index : entrypoint_index + 3],
            [str(TRL_ROOT / "grpo.py"), "--config", str(REPO_ROOT / GRPO_CONFIG)],
        )
        self.assertEqual(
            command[-4:],
            ["--max_steps", "5", "--dataset_name", "lab260/youtube_balalaika"],
        )

    def test_num_processes_must_be_a_positive_integer(self) -> None:
        for invalid in ("0", "-1", "1.5", "two"):
            with self.subTest(invalid=invalid):
                result = run_launcher(
                    "--num-processes", invalid, "--dry-run", "sft", SFT_CONFIG
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("must be a positive integer", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_dry_run_does_not_invoke_accelerate_or_training_entrypoint(self) -> None:
        result = run_launcher(
            "--python", "/bin/false", "--dry-run", "sft", SFT_CONFIG
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        _, command = dry_run_command(result)
        self.assertEqual(command[0], "/bin/false")

    def test_console_log_option_is_accepted_without_changing_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "trial.console.log"
            result = run_launcher(
                "--console-log", str(log_path), "--dry-run", "grpo", GRPO_CONFIG
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            _, command = dry_run_command(result)
            self.assertIn(str(TRL_ROOT / "grpo.py"), command)
            self.assertFalse(log_path.exists())


class AccelerateConfigTests(unittest.TestCase):
    def load(self, name: str) -> dict:
        path = TRL_ROOT / "configs" / "accelerate" / name
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertIsInstance(data, dict)
        return data

    def test_single_gpu_config_is_local_bf16(self) -> None:
        config = self.load("single_gpu.yaml")
        self.assertEqual(config["compute_environment"], "LOCAL_MACHINE")
        self.assertEqual(config["distributed_type"], "NO")
        self.assertEqual(config["num_machines"], 1)
        self.assertEqual(config["num_processes"], 1)
        self.assertEqual(config["mixed_precision"], "bf16")
        self.assertIs(config["use_cpu"], False)

    def test_multi_gpu_config_is_local_two_process_ddp(self) -> None:
        config = self.load("multi_gpu.yaml")
        self.assertEqual(config["compute_environment"], "LOCAL_MACHINE")
        self.assertEqual(config["distributed_type"], "MULTI_GPU")
        self.assertEqual(config["num_machines"], 1)
        self.assertEqual(config["num_processes"], 2)
        self.assertEqual(config["gpu_ids"], "all")
        self.assertEqual(config["mixed_precision"], "bf16")
        self.assertIs(config["use_cpu"], False)

    def test_local_launcher_contract_has_no_cloud_job_vocabulary(self) -> None:
        paths = [
            LAUNCH,
            TRL_ROOT / "configs" / "accelerate" / "single_gpu.yaml",
            TRL_ROOT / "configs" / "accelerate" / "multi_gpu.yaml",
        ]
        source = "\n".join(path.read_text(encoding="utf-8") for path in paths).lower()
        for forbidden in (
            "hf jobs",
            "hugging face jobs",
            "hf://",
            "bootstrap",
            "bucket",
            "cloud",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()

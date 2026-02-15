# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import math
import os
import hashlib
import json
import sys
import time
import warnings
from dataclasses import asdict
from importlib.metadata import version
from os.path import commonprefix
from pathlib import Path

import huggingface_hub
import optuna
import torch
import torch.nn.functional as F
import transformers
from accelerate.utils import (
    is_mlu_available,
    is_musa_available,
    is_npu_available,
    is_sdaa_available,
    is_xpu_available,
)
from huggingface_hub import ModelCard, ModelCardData
from optuna import Trial, TrialPruned
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial, TrialState
from pydantic import ValidationError
from questionary import Choice
from rich.traceback import install

from .analyzer import Analyzer
from .config import QuantizationMethod, Settings, get_configuration_diagnostics
from .evaluator import Evaluator
from .model import AbliterationParameters, Model, get_model_class
from .utils import (
    empty_cache,
    format_duration,
    get_readme_intro,
    get_trial_parameters,
    load_prompts,
    print,
    print_memory_usage,
    prompt_password,
    prompt_path,
    prompt_select,
    prompt_text,
)


def print_settings_diagnostics(settings: Settings) -> None:
    print()
    print("Configuration diagnostics:")

    diagnostics = get_configuration_diagnostics()
    candidate_files = diagnostics["candidate_files"]

    if candidate_files:
        print("* Discovered config files (highest precedence first):")
        for path in candidate_files:
            print(f"  * [bold]{path}[/]")
    else:
        print("* No config.toml/config.default.toml files discovered.")

    for detail in diagnostics["details"]:
        if "error" in detail:
            print(
                f"* [yellow]Could not parse[/] [bold]{detail['path']}[/]: {detail['error']}"
            )

    print("* Final resolved prompt source settings:")

    prompt_groups = [
        ("good_prompts", settings.good_prompts),
        ("bad_prompts", settings.bad_prompts),
        ("good_evaluation_prompts", settings.good_evaluation_prompts),
        ("bad_evaluation_prompts", settings.bad_evaluation_prompts),
    ]

    for group_name, specification in prompt_groups:
        print(
            f"  * {group_name}: source_type=[bold]{specification.source_type.value}[/]"
        )
        if specification.source_type.value == "dataset":
            print(
                f"    dataset=[bold]{specification.dataset}[/], split=[bold]{specification.split}[/], column=[bold]{specification.column}[/]"
            )
        else:
            path_display = (
                specification.path if specification.path is not None else "<unset>"
            )
            print(f"    path=[bold]{path_display}[/]")


def print_checkpoint_settings_diagnostics(
    current_settings: Settings,
    checkpoint_settings: Settings,
    checkpoint_file: str,
) -> None:
    print()
    print(f"Checkpoint diagnostics for [bold]{checkpoint_file}[/]:")

    groups = [
        "good_prompts",
        "bad_prompts",
        "good_evaluation_prompts",
        "bad_evaluation_prompts",
    ]

    for name in groups:
        current_spec = getattr(current_settings, name)
        checkpoint_spec = getattr(checkpoint_settings, name)

        current_label = (
            current_spec.path
            if current_spec.source_type.value == "text_file"
            else current_spec.dataset
        )
        checkpoint_label = (
            checkpoint_spec.path
            if checkpoint_spec.source_type.value == "text_file"
            else checkpoint_spec.dataset
        )

        changed = (
            current_spec.source_type != checkpoint_spec.source_type
            or current_label != checkpoint_label
        )
        marker = "[yellow]DIFF[/]" if changed else "same"

        print(
            f"* {name}: current=[bold]{current_spec.source_type.value}[/] ({current_label}) | "
            f"checkpoint=[bold]{checkpoint_spec.source_type.value}[/] ({checkpoint_label}) -> {marker}"
        )


def obtain_merge_strategy(settings: Settings) -> str | None:
    """
    Prompts the user for how to proceed with saving the model.
    Provides info to the user if the model is quantized on memory use.
    Returns "merge", "adapter", or None (if cancelled/invalid).
    """

    if settings.quantization == QuantizationMethod.BNB_4BIT:
        print()
        print(
            "Model was loaded with quantization. Merging requires reloading the base model."
        )
        print(
            "[yellow]WARNING: CPU merging requires dequantizing the entire model to system RAM.[/]"
        )
        print("[yellow]This can lead to system freezes if you run out of memory.[/]")

        try:
            # Estimate memory requirements by loading the model structure on the "meta" device.
            # This doesn't consume actual RAM but allows us to inspect the parameter count/dtype.
            #
            # Suppress warnings during meta device loading (e.g., "Some weights were not initialized").
            # These are expected and harmless since we're only inspecting model structure, not running inference.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                meta_model = get_model_class(settings.model).from_pretrained(
                    settings.model,
                    device_map="meta",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
                footprint_bytes = meta_model.get_memory_footprint()
                footprint_gb = footprint_bytes / (1024**3)
                print(
                    f"[yellow]Estimated RAM required (excluding overhead): [bold]~{footprint_gb:.2f} GB[/][/]"
                )
        except Exception:
            # Fallback if meta loading fails (e.g. owing to custom model code
            # or bitsandbytes quantization config issues on the meta device).
            print(
                "[yellow]Rule of thumb: You need approximately 3x the parameter count in GB RAM.[/]"
            )
            print(
                "[yellow]Example: A 27B model requires ~80GB RAM. A 70B model requires ~200GB RAM.[/]"
            )
        print()

        strategy = prompt_select(
            "How do you want to proceed?",
            choices=[
                Choice(
                    title="Merge LoRA into full model"
                    + (
                        ""
                        if settings.quantization == QuantizationMethod.NONE
                        else " (requires sufficient RAM)"
                    ),
                    value="merge",
                ),
                Choice(
                    title="Cancel",
                    value="cancel",
                ),
            ],
        )

        if strategy == "cancel":
            return None

        return strategy
    else:
        return "merge"


def calculate_refusal_directions(
    settings: Settings,
    model: Model,
    good_prompts: list[str],
    bad_prompts: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    print("Calculating per-layer refusal directions...")
    print("* Obtaining residuals for good prompts...")
    good_residuals = model.get_residuals_batched(good_prompts)
    print("* Obtaining residuals for bad prompts...")
    bad_residuals = model.get_residuals_batched(bad_prompts)

    good_means = good_residuals.mean(dim=0)
    bad_means = bad_residuals.mean(dim=0)

    refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)

    if settings.orthogonalize_direction:
        # Implements https://huggingface.co/blog/grimjim/projected-abliteration
        # Adjust the refusal directions so that only the component that is
        # orthogonal to the good direction is subtracted during abliteration.
        good_directions = F.normalize(good_means, p=2, dim=1)
        projection_vector = torch.sum(refusal_directions * good_directions, dim=1)
        refusal_directions = (
            refusal_directions - projection_vector.unsqueeze(1) * good_directions
        )
        refusal_directions = F.normalize(refusal_directions, p=2, dim=1)

    return refusal_directions, good_residuals, bad_residuals


def format_metadata_parameters(
    direction_index: float | None,
    parameters: dict[str, AbliterationParameters],
) -> dict[str, object]:
    formatted = {
        "direction_index": (
            "per layer" if direction_index is None else round(direction_index, 6)
        ),
        "parameters": {k: asdict(v) for k, v in sorted(parameters.items())},
    }
    return formatted


def build_model_card_metadata(
    settings: Settings,
    trial: Trial,
    direction_index: float | None,
    parameters: dict[str, AbliterationParameters],
    bad_prompt_count: int,
    seed_trial_index: int | None,
    seed_direction_index: float | None,
    seed_parameters: dict[str, AbliterationParameters] | None,
) -> tuple[dict[str, object], str, list[str]]:
    policy_optimized = seed_parameters is not None
    metadata = {
        "heretic_base_model": settings.model,
        "heretic_policy_optimization_enabled": policy_optimized,
        "heretic_initial_trial_id": (
            seed_trial_index if policy_optimized else trial.user_attrs.get("index")
        ),
        "heretic_initial_trial_parameters": (
            format_metadata_parameters(seed_direction_index, seed_parameters)
            if policy_optimized and seed_parameters is not None
            else format_metadata_parameters(direction_index, parameters)
        ),
        "heretic_policy_run_id": (
            trial.user_attrs.get("policy_run_id") if policy_optimized else None
        ),
        "heretic_policy_checkpoint": (
            trial.user_attrs.get("policy_checkpoint_file") if policy_optimized else None
        ),
        "heretic_final_refined_parameters": format_metadata_parameters(
            direction_index,
            parameters,
        ),
        "heretic_selection_metrics": {
            "kl_divergence": round(trial.user_attrs["kl_divergence"], 6),
            "refusals": trial.user_attrs["refusals"],
            "bad_prompt_count": bad_prompt_count,
            "base_refusals": trial.user_attrs.get("seed_refusals"),
            "refusal_change": trial.user_attrs.get("refusal_change"),
            "kl_change": trial.user_attrs.get("kl_change"),
        },
    }

    metadata_lines = []
    for key, value in metadata.items():
        serialized = json.dumps("n/a") if value is None else json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        metadata_lines.append(f"- **{key}**\n\n```json\n{serialized}\n```")

    metadata_section = "\n\n## Heretic metadata\n\n" + "\n\n".join(metadata_lines) + "\n"

    metadata_tags = [
        "heretic",
        "uncensored",
        "decensored",
        "abliterated",
        f"base_model:{settings.model}",
        f"policy_optimization:{'enabled' if policy_optimized else 'disabled'}",
        f"trial_id:{metadata['heretic_initial_trial_id']}",
    ]

    if metadata["heretic_policy_run_id"]:
        metadata_tags.append(f"policy_run:{metadata['heretic_policy_run_id']}")

    return metadata, metadata_section, metadata_tags


def inject_yaml_metadata(text: str, metadata: dict[str, object], tags: list[str]) -> str:
    yaml_lines = ["---", "tags:"]
    yaml_lines.extend([f"  - {tag}" for tag in tags])
    for key, value in metadata.items():
        if value is None:
            continue
        serialized = json.dumps(value, sort_keys=True)
        yaml_lines.append(f"{key}: {serialized}")
    yaml_lines.append("---")
    return "\n".join(yaml_lines) + "\n\n" + text.lstrip()


def run():
    # Enable expandable segments to reduce memory fragmentation on multi-GPU setups.
    if (
        "PYTORCH_ALLOC_CONF" not in os.environ
        and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
    ):
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    # Modified "Pagga" font from https://budavariam.github.io/asciiart-text/
    print(f"[cyan]█░█░█▀▀░█▀▄░█▀▀░▀█▀░█░█▀▀[/]  v{version('heretic-llm')}")
    print("[cyan]█▀█░█▀▀░█▀▄░█▀▀░░█░░█░█░░[/]")
    print(
        "[cyan]▀░▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀▀▀[/]  [blue underline]https://github.com/p-e-w/heretic[/]"
    )
    print()

    if (
        # There is at least one argument (argv[0] is the program name).
        len(sys.argv) > 1
        # No model has been explicitly provided.
        and "--model" not in sys.argv
        # The last argument is a parameter value rather than a flag (such as "--help").
        and not sys.argv[-1].startswith("-")
    ):
        # Assume the last argument is the model.
        sys.argv.insert(-1, "--model")

    try:
        # The required argument "model" must be provided by the user,
        # either on the command line or in the configuration file.
        settings = Settings()  # ty:ignore[missing-argument]
    except ValidationError as error:
        print(f"[red]Configuration contains [bold]{error.error_count()}[/] errors:[/]")

        for error in error.errors():
            print(f"[bold]{error['loc'][0]}[/]: [yellow]{error['msg']}[/]")

        print()
        print(
            "Run [bold]heretic --help[/] or see [bold]config.default.toml[/] for details about configuration parameters."
        )
        return

    print_settings_diagnostics(settings)

    # Adapted from https://github.com/huggingface/accelerate/blob/main/src/accelerate/commands/env.py
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print(f"Detected [bold]{count}[/] CUDA device(s):")
        for i in range(count):
            print(f"* GPU {i}: [bold]{torch.cuda.get_device_name(i)}[/]")
    elif is_xpu_available():
        count = torch.xpu.device_count()
        print(f"Detected [bold]{count}[/] XPU device(s):")
        for i in range(count):
            print(f"* XPU {i}: [bold]{torch.xpu.get_device_name(i)}[/]")
    elif is_mlu_available():
        count = torch.mlu.device_count()  # ty:ignore[unresolved-attribute]
        print(f"Detected [bold]{count}[/] MLU device(s):")
        for i in range(count):
            print(f"* MLU {i}: [bold]{torch.mlu.get_device_name(i)}[/]")  # ty:ignore[unresolved-attribute]
    elif is_sdaa_available():
        count = torch.sdaa.device_count()  # ty:ignore[unresolved-attribute]
        print(f"Detected [bold]{count}[/] SDAA device(s):")
        for i in range(count):
            print(f"* SDAA {i}: [bold]{torch.sdaa.get_device_name(i)}[/]")  # ty:ignore[unresolved-attribute]
    elif is_musa_available():
        count = torch.musa.device_count()  # ty:ignore[unresolved-attribute]
        print(f"Detected [bold]{count}[/] MUSA device(s):")
        for i in range(count):
            print(f"* MUSA {i}: [bold]{torch.musa.get_device_name(i)}[/]")  # ty:ignore[unresolved-attribute]
    elif is_npu_available():
        print(f"NPU detected (CANN version: [bold]{torch.version.cann}[/])")  # ty:ignore[unresolved-attribute]
    elif torch.backends.mps.is_available():
        print("Detected [bold]1[/] MPS device (Apple Metal)")
    else:
        print(
            "[bold yellow]No GPU or other accelerator detected. Operations will be slow.[/]"
        )

    # We don't need gradients as we only do inference.
    torch.set_grad_enabled(False)

    # While determining the optimal batch size, we will try many different batch sizes,
    # resulting in many computation graphs being compiled. Raising the limit (default = 8)
    # avoids errors from TorchDynamo assuming that something is wrong because we
    # recompile too often.
    torch._dynamo.config.cache_size_limit = 64

    # Silence warning spam from Transformers.
    # In my entire career I've never seen a useful warning from that library.
    transformers.logging.set_verbosity_error()

    # We do our own trial logging, so we don't need the INFO messages
    # about parameters and results.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Silence the warning about multivariate TPE being experimental.
    warnings.filterwarnings("ignore", category=ExperimentalWarning)

    os.makedirs(settings.study_checkpoint_dir, exist_ok=True)

    def sanitize_checkpoint_component(value: str) -> str:
        return "".join(
            [(c if (c.isalnum() or c in ["_", "-"]) else "--") for c in value]
        )

    study_checkpoint_file = os.path.join(
        settings.study_checkpoint_dir,
        sanitize_checkpoint_component(settings.model) + ".jsonl",
    )

    lock_obj = JournalFileOpenLock(study_checkpoint_file)
    backend = JournalFileBackend(study_checkpoint_file, lock_obj=lock_obj)
    storage = JournalStorage(backend)

    try:
        existing_study = storage.get_all_studies()[0]
    except IndexError:
        existing_study = None

    if existing_study is not None and settings.evaluate_model is None:
        saved_settings = Settings.model_validate_json(
            existing_study.user_attrs["settings"]
        )
        settings_changed = saved_settings.model_dump() != settings.model_dump()
        choices = []

        print_checkpoint_settings_diagnostics(
            settings,
            saved_settings,
            study_checkpoint_file,
        )

        if existing_study.user_attrs["finished"]:
            print()
            print(
                (
                    "[green]You have already processed this model.[/] "
                    "You can show the results from the previous run, allowing you to export models or to run additional trials. "
                    "Alternatively, you can ignore the previous run and start from scratch. "
                    "This will delete the checkpoint file and all results from the previous run."
                )
            )
            choices.append(
                Choice(
                    title="Show the results from the previous run",
                    value="continue",
                )
            )
        else:
            print()
            if settings_changed:
                print(
                    (
                        "[yellow]You have already processed this model, but the run was interrupted.[/] "
                        "The checkpoint settings differ from your current configuration. "
                        "Restarting will use your current settings. "
                        "Continuing will override your current settings with the checkpoint settings."
                    )
                )
                choices.append(
                    Choice(
                        title="Ignore the previous run and start from scratch (use current settings)",
                        value="restart",
                    )
                )
                choices.append(
                    Choice(
                        title="Continue the previous run (override with checkpoint settings)",
                        value="continue",
                    )
                )
            else:
                print(
                    (
                        "[yellow]You have already processed this model, but the run was interrupted.[/] "
                        "You can continue the previous run from where it stopped. This will override any specified settings. "
                        "Alternatively, you can ignore the previous run and start from scratch. "
                        "This will delete the checkpoint file and all results from the previous run."
                    )
                )
                choices.append(
                    Choice(
                        title="Continue the previous run",
                        value="continue",
                    )
                )

        if not settings_changed:
            choices.append(
                Choice(
                    title="Ignore the previous run and start from scratch",
                    value="restart",
                )
            )

        choices.append(
            Choice(
                title="Exit program",
                value="",
            )
        )

        print()
        choice = prompt_select("How would you like to proceed?", choices)

        if choice == "continue":
            settings = saved_settings
            print(
                "[yellow]Resuming with checkpoint settings. Current configuration was overridden.[/]"
            )
            print_settings_diagnostics(settings)
        elif choice == "restart":
            os.unlink(study_checkpoint_file)
            backend = JournalFileBackend(study_checkpoint_file, lock_obj=lock_obj)
            storage = JournalStorage(backend)
        elif choice is None or choice == "":
            return

    model = Model(settings)
    print()
    print_memory_usage()

    print()
    print(
        f"Loading good prompts from [bold]{settings.good_prompts.source_label()}[/]..."
    )
    good_prompts = load_prompts(settings, settings.good_prompts)
    print(f"* [bold]{len(good_prompts)}[/] prompts loaded")

    print()
    print(f"Loading bad prompts from [bold]{settings.bad_prompts.source_label()}[/]...")
    bad_prompts = load_prompts(settings, settings.bad_prompts)
    print(f"* [bold]{len(bad_prompts)}[/] prompts loaded")

    if settings.batch_size == 0:
        print()
        print("Determining optimal batch size...")

        batch_size = 1
        best_batch_size = -1
        best_performance = -1

        while batch_size <= settings.max_batch_size:
            print(f"* Trying batch size [bold]{batch_size}[/]... ", end="")

            prompts = good_prompts * math.ceil(batch_size / len(good_prompts))
            prompts = prompts[:batch_size]

            try:
                # Warmup run to build the computation graph so that part isn't benchmarked.
                model.get_responses(prompts)

                start_time = time.perf_counter()
                responses = model.get_responses(prompts)
                end_time = time.perf_counter()
            except Exception as error:
                if batch_size == 1:
                    # Even a batch size of 1 already fails.
                    # We cannot recover from this.
                    raise

                print(f"[red]Failed[/] ({error})")
                break

            response_lengths = [
                len(model.tokenizer.encode(response)) for response in responses
            ]
            performance = sum(response_lengths) / (end_time - start_time)

            print(f"[green]Ok[/] ([bold]{performance:.0f}[/] tokens/s)")

            if performance > best_performance:
                best_batch_size = batch_size
                best_performance = performance

            batch_size *= 2

        settings.batch_size = best_batch_size
        print(f"* Chosen batch size: [bold]{settings.batch_size}[/]")

    print()
    print("Checking for common response prefix...")
    responses = model.get_responses_batched(good_prompts[:100] + bad_prompts[:100])

    # Despite being located in os.path, commonprefix actually performs
    # a naive string operation without any path-specific logic,
    # which is exactly what we need here. Trailing spaces are removed
    # to avoid issues where multiple different tokens that all start
    # with a space character lead to the common prefix ending with
    # a space, which would result in an uncommon tokenization.
    model.response_prefix = commonprefix(responses).rstrip(" ")

    # Suppress CoT output.
    if model.response_prefix.startswith("<think>"):
        # Most thinking models.
        model.response_prefix = "<think></think>"
    elif model.response_prefix.startswith("<|channel|>analysis<|message|>"):
        # gpt-oss.
        model.response_prefix = "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"
    elif model.response_prefix.startswith("<thought>"):
        # Unknown, suggested by user.
        model.response_prefix = "<thought></thought>"
    elif model.response_prefix.startswith("[THINK]"):
        # Unknown, suggested by user.
        model.response_prefix = "[THINK][/THINK]"

    if model.response_prefix:
        print(f"* Prefix found: [bold]{model.response_prefix!r}[/]")
    else:
        print("* None found")

    evaluator = Evaluator(settings, model)

    if settings.evaluate_model is not None:
        print()
        print(f"Loading model [bold]{settings.evaluate_model}[/]...")
        settings.model = settings.evaluate_model
        model.reset_model()
        print("* Evaluating...")
        evaluator.get_score()
        return

    print()
    refusal_directions, good_residuals, bad_residuals = calculate_refusal_directions(
        settings,
        model,
        good_prompts,
        bad_prompts,
    )

    analyzer = Analyzer(settings, model, good_residuals, bad_residuals)

    if settings.print_residual_geometry:
        analyzer.print_residual_geometry()

    if settings.plot_residuals:
        analyzer.plot_residuals()

    # We don't need the residuals after computing refusal directions.
    del good_residuals, bad_residuals, analyzer
    empty_cache()

    trial_index = 0
    start_index = 0
    start_time = time.perf_counter()
    last_layer_index = len(model.get_layers()) - 1

    def sample_direction_scope(trial: Trial) -> str:
        return trial.suggest_categorical(
            "direction_scope",
            [
                "global",
                "per layer",
            ],
        )

    def sample_direction_index(
        trial: Trial,
        direction_scope: str,
        seed_direction_index: float | None = None,
        radius: float | None = None,
        global_exploration_probability: float = 0.0,
    ) -> float | None:
        min_layer = 0.4 * last_layer_index
        max_layer = 0.9 * last_layer_index

        if direction_scope == "per layer":
            return None

        if (
            seed_direction_index is not None
            and global_exploration_probability > 0.0
            and trial.suggest_float("global_exploration_roll", 0.0, 1.0)
            < global_exploration_probability
        ):
            return trial.suggest_float("direction_index", min_layer, max_layer)

        if seed_direction_index is None or radius is None:
            return trial.suggest_float("direction_index", min_layer, max_layer)

        return trial.suggest_float(
            "direction_index",
            max(min_layer, seed_direction_index - radius),
            min(max_layer, seed_direction_index + radius),
        )

    def sample_component_parameters(
        trial: Trial,
        component: str,
        seed_parameters: AbliterationParameters | None = None,
        radius: dict[str, float] | None = None,
        global_exploration_probability: float = 0.0,
    ) -> AbliterationParameters:
        global_ranges = {
            "max_weight": (0.8, 1.5),
            "max_weight_position": (0.6 * last_layer_index, 1.0 * last_layer_index),
            "min_weight": (0.0, 1.5),
            "min_weight_distance": (1.0, 0.6 * last_layer_index),
        }

        should_explore_global = (
            seed_parameters is not None
            and global_exploration_probability > 0.0
            and trial.suggest_float(
                f"{component}.global_exploration_roll",
                0.0,
                1.0,
            )
            < global_exploration_probability
        )

        def suggest_around_seed(parameter_name: str) -> float:
            global_low, global_high = global_ranges[parameter_name]
            if (
                seed_parameters is None
                or radius is None
                or parameter_name not in radius
                or should_explore_global
            ):
                return trial.suggest_float(
                    f"{component}.{parameter_name}",
                    global_low,
                    global_high,
                )

            seed_value = getattr(seed_parameters, parameter_name)
            parameter_radius = radius[parameter_name]
            return trial.suggest_float(
                f"{component}.{parameter_name}",
                max(global_low, seed_value - parameter_radius),
                min(global_high, seed_value + parameter_radius),
            )

        max_weight = suggest_around_seed("max_weight")
        max_weight_position = suggest_around_seed("max_weight_position")
        min_weight = min(max_weight, suggest_around_seed("min_weight"))
        min_weight_distance = suggest_around_seed("min_weight_distance")

        return AbliterationParameters(
            max_weight=max_weight,
            max_weight_position=max_weight_position,
            min_weight=min_weight,
            min_weight_distance=min_weight_distance,
        )

    def objective(trial: Trial) -> tuple[float, float]:
        nonlocal trial_index
        trial_index += 1
        trial.set_user_attr("index", trial_index)

        direction_scope = sample_direction_scope(trial)

        # Discrimination between "harmful" and "harmless" inputs is usually strongest
        # in layers slightly past the midpoint of the layer stack. See the original
        # abliteration paper (https://arxiv.org/abs/2406.11717) for a deeper analysis.
        #
        # Note that we always sample this parameter even though we only need it for
        # the "global" direction scope. The reason is that multivariate TPE doesn't
        # work with conditional or variable-range parameters.
        direction_index = sample_direction_index(trial, direction_scope)

        parameters = {}

        for component in model.get_abliterable_components():
            parameters[component] = sample_component_parameters(
                trial,
                component,
            )

        trial.set_user_attr("direction_index", direction_index)
        trial.set_user_attr("parameters", {k: asdict(v) for k, v in parameters.items()})

        print()
        print(
            f"Running trial [bold]{trial_index}[/] of [bold]{settings.n_trials}[/]..."
        )
        print("* Parameters:")
        for name, value in get_trial_parameters(trial).items():
            print(f"  * {name} = [bold]{value}[/]")
        print("* Resetting model...")
        model.reset_model()
        print("* Abliterating...")
        model.abliterate(
            refusal_directions,
            direction_index,
            parameters,
            require_reset=True,
        )
        print("* Evaluating...")
        score, kl_divergence, refusals = evaluator.get_score()

        elapsed_time = time.perf_counter() - start_time
        remaining_time = (elapsed_time / (trial_index - start_index)) * (
            settings.n_trials - trial_index
        )
        print()
        print(f"[grey50]Elapsed time: [bold]{format_duration(elapsed_time)}[/][/]")
        if trial_index < settings.n_trials:
            print(
                f"[grey50]Estimated remaining time: [bold]{format_duration(remaining_time)}[/][/]"
            )
        print_memory_usage()

        trial.set_user_attr("kl_divergence", kl_divergence)
        trial.set_user_attr("refusals", refusals)

        return score

    def objective_wrapper(trial: Trial) -> tuple[float, float]:
        try:
            return objective(trial)
        except KeyboardInterrupt:
            # Stop the study gracefully on Ctrl+C.
            trial.study.stop()
            raise TrialPruned()

    def optimize_policy_from_seed(
        seed_direction_index: float | None,
        seed_parameters: dict[str, AbliterationParameters],
        seed_trial_index: int | None,
    ) -> tuple[float | None, dict[str, AbliterationParameters], Trial | None]:
        nonlocal refusal_directions

        policy_mode = settings.policy_refusal_directions_mode
        refresh_interval = settings.policy_refusal_directions_refresh_interval
        if policy_mode != "periodic":
            refresh_interval = None

        policy_trials_per_iteration = settings.policy_n_trials
        policy_iterations = settings.policy_optimization_iterations

        def refresh_refusal_directions(reason: str) -> None:
            nonlocal refusal_directions
            print(f"* {reason}")
            model.reset_model()
            refusal_directions, _, _ = calculate_refusal_directions(
                settings,
                model,
                good_prompts,
                bad_prompts,
            )
            print(
                "* Refusal directions updated. KL/refusal scoring baseline remains fixed to the original model."
            )

        def clone_parameters(
            parameters: dict[str, AbliterationParameters],
        ) -> dict[str, AbliterationParameters]:
            return {
                component: AbliterationParameters(**asdict(component_parameters))
                for component, component_parameters in parameters.items()
            }

        def is_trial_better(candidate: Trial, incumbent: Trial) -> bool:
            candidate_pair = (
                candidate.user_attrs["refusals"],
                candidate.user_attrs["kl_divergence"],
            )
            incumbent_pair = (
                incumbent.user_attrs["refusals"],
                incumbent.user_attrs["kl_divergence"],
            )
            return candidate_pair < incumbent_pair

        def get_sampling_configuration(iteration_index: int) -> tuple[dict[str, float], float]:
            local_scale = max(settings.policy_local_perturbation_scale, 1e-4)
            iteration_decay = max(0.4, 1.0 - (0.18 * iteration_index))
            effective_scale = max(local_scale * iteration_decay, local_scale * 0.35)
            policy_sampling_radius = {
                "direction_index": max(1.0, effective_scale * last_layer_index),
                "max_weight": max(0.04, effective_scale * 0.7),
                "max_weight_position": max(1.0, effective_scale * last_layer_index),
                "min_weight": max(0.04, effective_scale * 0.7),
                "min_weight_distance": max(1.0, effective_scale * last_layer_index),
            }
            policy_global_exploration_probability = min(0.35, 0.12 + (effective_scale * 0.8))
            return policy_sampling_radius, policy_global_exploration_probability

        print()
        print("[bold]Running policy optimization from selected trial...[/]")
        print("* Re-initializing evaluation baseline from unedited model...")
        evaluator.initialize_baseline()

        if policy_mode in ["recompute", "periodic"]:
            refresh_refusal_directions(
                "Recomputing refusal directions before policy optimization"
            )
            if policy_mode == "periodic" and refresh_interval is None:
                print(
                    "* Policy periodic refresh interval is not set; using only the initial policy-phase recomputation."
                )

        def evaluate_seed_policy(
            direction_index: float | None,
            parameters: dict[str, AbliterationParameters],
        ) -> tuple[float, int]:
            print("* Evaluating seed policy...")
            model.reset_model()
            model.abliterate(
                refusal_directions,
                direction_index,
                parameters,
                require_reset=True,
            )
            _, seed_kl, seed_refusals = evaluator.get_score()
            print(
                "* Seed policy metrics: "
                f"Refusals {seed_refusals:>2}/{len(evaluator.bad_prompts)}, "
                f"KL divergence {seed_kl:.4f}"
            )
            return seed_kl, seed_refusals

        base_checkpoint_dir = settings.policy_checkpoint_dir or settings.study_checkpoint_dir
        os.makedirs(base_checkpoint_dir, exist_ok=True)

        active_direction_index = seed_direction_index
        active_parameters = clone_parameters(seed_parameters)
        active_trial = None

        for iteration in range(policy_iterations):
            print()
            print(
                f"[bold]Policy optimization iteration {iteration + 1}/{policy_iterations}[/]"
            )
            seed_kl, seed_refusals = evaluate_seed_policy(
                active_direction_index,
                active_parameters,
            )

            kl_ceiling = settings.policy_hard_kl_ceiling
            refusal_improvement_delta = None
            if settings.policy_min_refusal_improvement is not None:
                refusal_improvement_delta = math.ceil(
                    settings.policy_min_refusal_improvement * len(evaluator.bad_prompts)
                )

            if kl_ceiling is not None or refusal_improvement_delta is not None:
                print("* Active hard constraints:")
                if kl_ceiling is not None:
                    print(f"  * KL divergence <= [bold]{kl_ceiling:.4f}[/]")
                if refusal_improvement_delta is not None:
                    print(
                        "  * Refusals <= [bold]"
                        f"{seed_refusals - refusal_improvement_delta}[/] "
                        f"(seed {seed_refusals} - {refusal_improvement_delta})"
                    )

            seed_policy_fingerprint = {
                "direction_index": active_direction_index,
                "parameters": {
                    component: asdict(component_parameters)
                    for component, component_parameters in sorted(active_parameters.items())
                },
                "iteration": iteration,
            }
            seed_policy_hash = hashlib.sha256(
                json.dumps(seed_policy_fingerprint, sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]
            seed_tag = (
                f"trial-{seed_trial_index}-iter-{iteration + 1}"
                if seed_trial_index is not None
                else f"policy-{seed_policy_hash}"
            )

            policy_study_checkpoint_file = os.path.join(
                base_checkpoint_dir,
                "--".join(
                    [
                        sanitize_checkpoint_component(settings.model),
                        "policy-opt",
                        sanitize_checkpoint_component(seed_tag),
                    ]
                )
                + ".jsonl",
            )
            policy_lock_obj = JournalFileOpenLock(policy_study_checkpoint_file)
            policy_backend = JournalFileBackend(
                policy_study_checkpoint_file,
                lock_obj=policy_lock_obj,
            )
            policy_storage = JournalStorage(policy_backend)

            policy_sampling_radius, policy_global_exploration_probability = (
                get_sampling_configuration(iteration)
            )

            seed_policy = {
                "direction_index": active_direction_index,
                "seed_refusals": seed_refusals,
                "seed_kl": seed_kl,
                "parameters": {
                    component: asdict(component_parameters)
                    for component, component_parameters in sorted(active_parameters.items())
                },
            }

            def policy_constraints(trial: FrozenTrial) -> tuple[float, float]:
                if trial.state != TrialState.COMPLETE:
                    return (0.0, 0.0)
                refusal_violation = 0.0
                if refusal_improvement_delta is not None:
                    refusal_limit = seed_refusals - refusal_improvement_delta
                    refusal_violation = max(0.0, trial.user_attrs["refusals"] - refusal_limit)
                kl_violation = 0.0
                if kl_ceiling is not None:
                    kl_violation = max(0.0, trial.user_attrs["kl_divergence"] - kl_ceiling)
                return (refusal_violation, kl_violation)

            policy_study = optuna.create_study(
                sampler=TPESampler(
                    n_startup_trials=min(settings.n_startup_trials, policy_trials_per_iteration),
                    n_ei_candidates=128,
                    multivariate=True,
                    constraints_func=policy_constraints,
                ),
                directions=[StudyDirection.MINIMIZE, StudyDirection.MINIMIZE],
                storage=policy_storage,
                study_name="heretic-policy-opt",
                load_if_exists=True,
            )
            policy_study.set_user_attr("seed_policy", seed_policy)
            policy_study.set_user_attr(
                "constraints",
                {
                    "kl_ceiling": kl_ceiling,
                    "refusal_improvement_delta": refusal_improvement_delta,
                },
            )

            policy_trial_count = 0

            def policy_objective(policy_trial: Trial) -> tuple[float, float]:
                nonlocal policy_trial_count
                if (
                    policy_mode == "periodic"
                    and refresh_interval is not None
                    and refresh_interval > 0
                    and policy_trial_count > 0
                    and policy_trial_count % refresh_interval == 0
                ):
                    refresh_refusal_directions(
                        f"Periodic refusal-direction refresh before policy trial {policy_trial.number}"
                    )

                direction_scope = sample_direction_scope(policy_trial)

                direction_index = sample_direction_index(
                    policy_trial,
                    direction_scope,
                    seed_direction_index=active_direction_index,
                    radius=policy_sampling_radius["direction_index"],
                    global_exploration_probability=policy_global_exploration_probability,
                )

                parameters = {}
                deltas = {
                    "direction_index": (
                        None
                        if direction_index is None or active_direction_index is None
                        else direction_index - active_direction_index
                    ),
                    "parameters": {},
                }

                for component in model.get_abliterable_components():
                    component_parameters = sample_component_parameters(
                        policy_trial,
                        component,
                        seed_parameters=active_parameters[component],
                        radius={
                            "max_weight": policy_sampling_radius["max_weight"],
                            "max_weight_position": policy_sampling_radius[
                                "max_weight_position"
                            ],
                            "min_weight": policy_sampling_radius["min_weight"],
                            "min_weight_distance": policy_sampling_radius[
                                "min_weight_distance"
                            ],
                        },
                        global_exploration_probability=policy_global_exploration_probability,
                    )
                    parameters[component] = component_parameters

                    seed_component = active_parameters[component]
                    deltas["parameters"][component] = {
                        "max_weight": (
                            component_parameters.max_weight - seed_component.max_weight
                        ),
                        "max_weight_position": (
                            component_parameters.max_weight_position
                            - seed_component.max_weight_position
                        ),
                        "min_weight": (
                            component_parameters.min_weight - seed_component.min_weight
                        ),
                        "min_weight_distance": (
                            component_parameters.min_weight_distance
                            - seed_component.min_weight_distance
                        ),
                    }

                policy_trial.set_user_attr("direction_index", direction_index)
                policy_trial.set_user_attr(
                    "parameters", {k: asdict(v) for k, v in parameters.items()}
                )
                policy_trial.set_user_attr("seed_policy", seed_policy)
                policy_trial.set_user_attr("candidate_deltas", deltas)
                policy_trial.set_user_attr("seed_refusals", seed_refusals)
                policy_trial.set_user_attr("seed_kl", seed_kl)
                policy_trial.set_user_attr("policy_run_id", seed_tag)
                policy_trial.set_user_attr(
                    "policy_checkpoint_file",
                    policy_study_checkpoint_file,
                )

                print("* Resetting model...")
                model.reset_model()
                print("* Abliterating...")
                model.abliterate(
                    refusal_directions,
                    direction_index,
                    parameters,
                    require_reset=True,
                )
                print("* Evaluating...")
                score, kl_divergence, refusals = evaluator.get_score()

                policy_trial.set_user_attr("kl_divergence", kl_divergence)
                policy_trial.set_user_attr("refusals", refusals)
                policy_trial.set_user_attr("refusal_change", refusals - seed_refusals)
                policy_trial.set_user_attr("kl_change", kl_divergence - seed_kl)

                accepted = True
                if kl_ceiling is not None and kl_divergence > kl_ceiling:
                    accepted = False
                if refusal_improvement_delta is not None and (
                    refusals > seed_refusals - refusal_improvement_delta
                ):
                    accepted = False
                policy_trial.set_user_attr("accepted", accepted)
                policy_trial_count += 1

                return score

            seed_trial_params = {
                "direction_scope": (
                    "global" if active_direction_index is not None else "per layer"
                ),
                "direction_index": (
                    active_direction_index
                    if active_direction_index is not None
                    else 0.65 * last_layer_index
                ),
            }
            for component, component_parameters in active_parameters.items():
                seed_trial_params[f"{component}.max_weight"] = component_parameters.max_weight
                seed_trial_params[f"{component}.max_weight_position"] = (
                    component_parameters.max_weight_position
                )
                seed_trial_params[f"{component}.min_weight"] = component_parameters.min_weight
                seed_trial_params[f"{component}.min_weight_distance"] = (
                    component_parameters.min_weight_distance
                )

            policy_study.enqueue_trial(seed_trial_params)
            policy_study.optimize(policy_objective, n_trials=policy_trials_per_iteration)

            completed_trials = [
                t for t in policy_study.trials if t.state == TrialState.COMPLETE
            ]
            if not completed_trials:
                continue

            accepted_trials = [
                t for t in completed_trials if t.user_attrs.get("accepted", True)
            ]
            candidate_pool = accepted_trials if accepted_trials else completed_trials

            best_policy_trial = min(
                candidate_pool,
                key=lambda t: (
                    t.user_attrs["refusals"],
                    t.user_attrs["kl_divergence"],
                ),
            )

            print()
            print("Policy optimization summary:")
            print(
                f"* Seed policy: Refusals {seed_refusals:>2}/{len(evaluator.bad_prompts)}, "
                f"KL divergence {seed_kl:.4f}"
            )
            print(
                "* Auto-selected best trial: "
                f"[bold]{best_policy_trial.number}[/] "
                f"(Refusals {best_policy_trial.user_attrs['refusals']:>2}/{len(evaluator.bad_prompts)}, "
                f"KL {best_policy_trial.user_attrs['kl_divergence']:.4f}, "
                f"Δrefusal {best_policy_trial.user_attrs['refusal_change']:+.0f}, "
                f"ΔKL {best_policy_trial.user_attrs['kl_change']:+.4f})"
            )

            if active_trial is None or is_trial_better(best_policy_trial, active_trial):
                active_trial = best_policy_trial
                active_direction_index = best_policy_trial.user_attrs["direction_index"]
                active_parameters = {
                    k: AbliterationParameters(**v)
                    for k, v in best_policy_trial.user_attrs["parameters"].items()
                }
                print("* Accepted as new policy seed for the next iteration.")
            else:
                print("* No improvement over current seed; retaining previous policy for stability.")

        if active_trial is None:
            return seed_direction_index, seed_parameters, None

        print()
        print(
            "Selected refined policy: "
            f"Refusals {active_trial.user_attrs['refusals']:>2}/{len(evaluator.bad_prompts)}, "
            f"KL divergence {active_trial.user_attrs['kl_divergence']:.4f}, "
            f"Δrefusal {active_trial.user_attrs['refusal_change']:+.0f}, "
            f"ΔKL {active_trial.user_attrs['kl_change']:+.4f}"
        )
        print("* Loading selected refined policy...")
        model.reset_model()
        model.abliterate(
            refusal_directions,
            active_direction_index,
            active_parameters,
            require_reset=True,
        )

        return active_direction_index, active_parameters, active_trial

    def compare_seed_vs_refined_policy(
        seed_direction_index: float | None,
        seed_parameters: dict[str, AbliterationParameters],
        refined_direction_index: float | None,
        refined_parameters: dict[str, AbliterationParameters],
    ) -> None:
        def truncate(text: str, max_chars: int = 96) -> str:
            compact = " ".join(text.split())
            if len(compact) <= max_chars:
                return compact
            return compact[: max_chars - 1] + "…"

        def evaluate_policy(
            label: str,
            direction_index: float | None,
            parameters: dict[str, AbliterationParameters],
            collect_responses: bool,
        ) -> tuple[float, int, list[bool] | None]:
            print(f"* Resetting and applying [bold]{label}[/] policy...")
            model.reset_model()
            model.abliterate(
                refusal_directions,
                direction_index,
                parameters,
                require_reset=True,
            )
            print(f"* Evaluating [bold]{label}[/] policy...")
            _, kl_divergence, refusals = evaluator.get_score()
            print(
                f"* {label.capitalize()} metrics: "
                f"Refusals {refusals:>2}/{len(evaluator.bad_prompts)}, "
                f"KL divergence {kl_divergence:.4f}"
            )

            refusal_flags: list[bool] | None = None
            if collect_responses:
                responses = model.get_responses_batched(
                    evaluator.bad_prompts,
                    skip_special_tokens=True,
                )
                refusal_flags = [evaluator.is_refusal(response) for response in responses]

            return kl_divergence, refusals, refusal_flags

        print()
        print("[bold]Comparing seed vs refined policy...[/]")
        seed_kl, seed_refusals, seed_flags = evaluate_policy(
            "seed",
            seed_direction_index,
            seed_parameters,
            collect_responses=settings.print_responses,
        )
        refined_kl, refined_refusals, refined_flags = evaluate_policy(
            "refined",
            refined_direction_index,
            refined_parameters,
            collect_responses=settings.print_responses,
        )

        print()
        print("Comparison summary:")
        print(
            f"* Refusals: [bold]{seed_refusals}[/] → [bold]{refined_refusals}[/] "
            f"(Δ {refined_refusals - seed_refusals:+d})"
        )
        print(
            f"* KL divergence: [bold]{seed_kl:.4f}[/] → [bold]{refined_kl:.4f}[/] "
            f"(Δ {refined_kl - seed_kl:+.4f})"
        )

        if settings.print_responses and seed_flags is not None and refined_flags is not None:
            print("* Prompt-level refusal status changes:")
            changes = 0
            for prompt, seed_refusal, refined_refusal in zip(
                evaluator.bad_prompts,
                seed_flags,
                refined_flags,
            ):
                if seed_refusal == refined_refusal:
                    continue
                changes += 1
                direction = (
                    "refusal → non-refusal"
                    if seed_refusal and not refined_refusal
                    else "non-refusal → refusal"
                )
                print(f"  * {direction}: [italic]{truncate(prompt.user)}[/]")

            if changes == 0:
                print("  * No refusal-status changes across evaluation prompts.")

    study = optuna.create_study(
        sampler=TPESampler(
            n_startup_trials=settings.n_startup_trials,
            n_ei_candidates=128,
            multivariate=True,
        ),
        directions=[StudyDirection.MINIMIZE, StudyDirection.MINIMIZE],
        storage=storage,
        study_name="heretic",
        load_if_exists=True,
    )

    study.set_user_attr("settings", settings.model_dump_json())
    study.set_user_attr("finished", False)

    def count_completed_trials() -> int:
        # Count number of complete trials to compute trials to run.
        return sum([(1 if t.state == TrialState.COMPLETE else 0) for t in study.trials])

    start_index = trial_index = count_completed_trials()
    if start_index > 0:
        print()
        print("Resuming existing study.")

    try:
        study.optimize(
            objective_wrapper,
            n_trials=settings.n_trials - count_completed_trials(),
        )
    except KeyboardInterrupt:
        # This additional handler takes care of the small chance that KeyboardInterrupt
        # is raised just between trials, which wouldn't be caught by the handler
        # defined in objective_wrapper above.
        pass

    if count_completed_trials() == settings.n_trials:
        study.set_user_attr("finished", True)

    while True:
        # If no trials at all have been evaluated, the study must have been stopped
        # by pressing Ctrl+C while the first trial was running. In this case, we just
        # re-raise the interrupt to invoke the standard handler defined below.
        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not completed_trials:
            raise KeyboardInterrupt

        # Get the Pareto front of trials. We can't use study.best_trials directly
        # as get_score() doesn't return the pure KL divergence and refusal count.
        # Note: Unlike study.best_trials, this does not handle objective constraints.
        sorted_trials = sorted(
            completed_trials,
            key=lambda trial: (
                trial.user_attrs["refusals"],
                trial.user_attrs["kl_divergence"],
            ),
        )
        min_divergence = math.inf
        best_trials = []
        for trial in sorted_trials:
            kl_divergence = trial.user_attrs["kl_divergence"]
            if kl_divergence < min_divergence:
                min_divergence = kl_divergence
                best_trials.append(trial)

        choices = [
            Choice(
                title=(
                    f"[Trial {trial.user_attrs['index']:>3}] "
                    f"Refusals: {trial.user_attrs['refusals']:>2}/{len(evaluator.bad_prompts)}, "
                    f"KL divergence: {trial.user_attrs['kl_divergence']:.4f}"
                ),
                value=trial,
            )
            for trial in best_trials
        ]

        choices.append(
            Choice(
                title="Run additional trials",
                value="continue",
            )
        )

        choices.append(
            Choice(
                title="Exit program",
                value="",
            )
        )

        print()
        print("[bold green]Optimization finished![/]")
        print()
        print(
            (
                "The following trials resulted in Pareto optimal combinations of refusals and KL divergence. "
                "After selecting a trial, you will be able to save the model, upload it to Hugging Face, "
                "or chat with it to test how well it works. You can return to this menu later to select a different trial. "
                "[yellow]Note that KL divergence values above 1 usually indicate significant damage to the original model's capabilities.[/]"
            )
        )

        while True:
            print()
            trial = prompt_select("Which trial do you want to use?", choices)

            if trial == "continue":
                while True:
                    try:
                        n_additional_trials = prompt_text(
                            "How many additional trials do you want to run?"
                        )
                        if n_additional_trials is None or n_additional_trials == "":
                            n_additional_trials = 0
                            break
                        n_additional_trials = int(n_additional_trials)
                        if n_additional_trials > 0:
                            break
                        print("[red]Please enter a number greater than 0.[/]")
                    except ValueError:
                        print("[red]Please enter a number.[/]")

                if n_additional_trials == 0:
                    continue

                settings.n_trials += n_additional_trials
                study.set_user_attr("settings", settings.model_dump_json())
                study.set_user_attr("finished", False)

                try:
                    study.optimize(
                        objective_wrapper,
                        n_trials=settings.n_trials - count_completed_trials(),
                    )
                except KeyboardInterrupt:
                    pass

                if count_completed_trials() == settings.n_trials:
                    study.set_user_attr("finished", True)

                break

            elif trial is None or trial == "":
                return

            print()
            print(f"Restoring model from trial [bold]{trial.user_attrs['index']}[/]...")
            print("* Parameters:")
            for name, value in get_trial_parameters(trial).items():
                print(f"  * {name} = [bold]{value}[/]")

            active_trial = trial
            active_direction_index = active_trial.user_attrs["direction_index"]
            active_parameters = {
                k: AbliterationParameters(**v)
                for k, v in active_trial.user_attrs["parameters"].items()
            }
            comparison_seed_direction_index: float | None = None
            comparison_seed_parameters: dict[str, AbliterationParameters] | None = None
            comparison_seed_trial_index: int | None = None

            print("* Resetting model...")
            model.reset_model()
            print("* Abliterating...")
            model.abliterate(
                refusal_directions,
                active_direction_index,
                active_parameters,
                require_reset=True,
            )

            if settings.policy_optimization_enabled:
                print()
                print("[bold]Policy optimization is enabled in config; starting automatic refinement...[/]")
                seed_direction_index = active_direction_index
                seed_parameters = {
                    component: component_parameters
                    for component, component_parameters in active_parameters.items()
                }
                (
                    active_direction_index,
                    active_parameters,
                    refined_policy_trial,
                ) = optimize_policy_from_seed(
                    active_direction_index,
                    active_parameters,
                    active_trial.user_attrs.get("index"),
                )
                if refined_policy_trial is not None:
                    comparison_seed_trial_index = active_trial.user_attrs.get("index")
                    active_trial = refined_policy_trial
                    comparison_seed_direction_index = seed_direction_index
                    comparison_seed_parameters = seed_parameters

            while True:
                print()
                action = prompt_select(
                    "What do you want to do with the decensored model?",
                    [
                        "Run policy optimization from this trial",
                        "Compare seed vs refined",
                        "Save the model to a local folder",
                        "Upload the model to Hugging Face",
                        "Chat with the model",
                        "Return to the trial selection menu",
                    ],
                )

                if action is None or action == "Return to the trial selection menu":
                    break

                # All actions are wrapped in a try/except block so that if an error occurs,
                # another action can be tried, instead of the program crashing and losing
                # the optimized model.
                try:
                    match action:
                        case "Run policy optimization from this trial":
                            seed_direction_index = active_direction_index
                            seed_parameters = {
                                component: component_parameters
                                for component, component_parameters in active_parameters.items()
                            }
                            (
                                active_direction_index,
                                active_parameters,
                                refined_policy_trial,
                            ) = optimize_policy_from_seed(
                                active_direction_index,
                                active_parameters,
                                active_trial.user_attrs.get("index"),
                            )
                            if refined_policy_trial is not None:
                                comparison_seed_trial_index = active_trial.user_attrs.get("index")
                                active_trial = refined_policy_trial
                                comparison_seed_direction_index = seed_direction_index
                                comparison_seed_parameters = seed_parameters

                        case "Compare seed vs refined":
                            if comparison_seed_parameters is None:
                                print(
                                    "[yellow]Run policy optimization from this trial first to compare seed vs refined.[/]"
                                )
                                continue
                            compare_seed_vs_refined_policy(
                                comparison_seed_direction_index,
                                comparison_seed_parameters,
                                active_direction_index,
                                active_parameters,
                            )

                        case "Save the model to a local folder":
                            save_directory = prompt_path("Path to the folder:")
                            if not save_directory:
                                continue

                            strategy = obtain_merge_strategy(settings)
                            if strategy is None:
                                continue

                            metadata, metadata_section, metadata_tags = build_model_card_metadata(
                                settings,
                                active_trial,
                                active_direction_index,
                                active_parameters,
                                len(evaluator.bad_prompts),
                                comparison_seed_trial_index,
                                comparison_seed_direction_index,
                                comparison_seed_parameters,
                            )

                            if strategy == "adapter":
                                print("Saving LoRA adapter...")
                                model.model.save_pretrained(save_directory)
                            else:
                                print("Saving merged model...")
                                merged_model = model.get_merged_model()
                                merged_model.save_pretrained(save_directory)
                                del merged_model
                                empty_cache()
                                model.tokenizer.save_pretrained(save_directory)

                            readme_text = inject_yaml_metadata(
                                get_readme_intro(
                                    settings,
                                    active_trial,
                                    evaluator.base_refusals,
                                    evaluator.bad_prompts,
                                )
                                + metadata_section,
                                metadata,
                                metadata_tags,
                            )
                            Path(save_directory, "README.md").write_text(
                                readme_text,
                                encoding="utf-8",
                            )

                            print(f"Model saved to [bold]{save_directory}[/].")

                        case "Upload the model to Hugging Face":
                            # We don't use huggingface_hub.login() because that stores the token on disk,
                            # and since this program will often be run on rented or shared GPU servers,
                            # it's better to not persist credentials.
                            token = huggingface_hub.get_token()
                            if not token:
                                token = prompt_password("Hugging Face access token:")
                            if not token:
                                continue

                            user = huggingface_hub.whoami(token)
                            fullname = user.get(
                                "fullname",
                                user.get("name", "unknown user"),
                            )
                            email = user.get("email", "no email found")
                            print(f"Logged in as [bold]{fullname} ({email})[/]")

                            repo_id = prompt_text(
                                "Name of repository:",
                                default=f"{user['name']}/{Path(settings.model).name}-heretic",
                            )

                            visibility = prompt_select(
                                "Should the repository be public or private?",
                                [
                                    "Public",
                                    "Private",
                                ],
                            )
                            private = visibility == "Private"

                            strategy = obtain_merge_strategy(settings)
                            if strategy is None:
                                continue

                            if strategy == "adapter":
                                print("Uploading LoRA adapter...")
                                model.model.push_to_hub(
                                    repo_id,
                                    private=private,
                                    token=token,
                                )
                            else:
                                print("Uploading merged model...")
                                merged_model = model.get_merged_model()
                                merged_model.push_to_hub(
                                    repo_id,
                                    private=private,
                                    token=token,
                                )
                                del merged_model
                                empty_cache()
                                model.tokenizer.push_to_hub(
                                    repo_id,
                                    private=private,
                                    token=token,
                                )

                            metadata, metadata_section, metadata_tags = build_model_card_metadata(
                                settings,
                                active_trial,
                                active_direction_index,
                                active_parameters,
                                len(evaluator.bad_prompts),
                                comparison_seed_trial_index,
                                comparison_seed_direction_index,
                                comparison_seed_parameters,
                            )

                            generated_intro = (
                                get_readme_intro(
                                    settings,
                                    active_trial,
                                    evaluator.base_refusals,
                                    evaluator.bad_prompts,
                                )
                                + metadata_section
                            )

                            # If the model path doesn't exist locally, it can be assumed
                            # to be a model hosted on the Hugging Face Hub, in which case
                            # we can retrieve and extend the model card.
                            if not Path(settings.model).exists():
                                card = ModelCard.load(settings.model)
                            else:
                                card = ModelCard("")

                            if card.data is None:
                                card.data = ModelCardData()
                            existing_tags = card.data.tags or []
                            card.data.tags = sorted(set(existing_tags + metadata_tags))
                            card.text = inject_yaml_metadata(
                                generated_intro + card.text,
                                metadata,
                                card.data.tags,
                            )
                            card.push_to_hub(repo_id, token=token)

                            print(f"Model uploaded to [bold]{repo_id}[/].")

                        case "Chat with the model":
                            print()
                            print(
                                "[cyan]Press Ctrl+C at any time to return to the menu.[/]"
                            )

                            chat = [
                                {"role": "system", "content": settings.system_prompt},
                            ]

                            while True:
                                try:
                                    message = prompt_text(
                                        "User:",
                                        qmark=">",
                                        unsafe=True,
                                    )
                                    if not message:
                                        break
                                    chat.append({"role": "user", "content": message})

                                    print("[bold]Assistant:[/] ", end="")
                                    response = model.stream_chat_response(chat)
                                    chat.append(
                                        {"role": "assistant", "content": response}
                                    )
                                except (KeyboardInterrupt, EOFError):
                                    # Ctrl+C/Ctrl+D
                                    break

                except Exception as error:
                    print(f"[red]Error: {error}[/]")


def main():
    # Install Rich traceback handler.
    install()

    try:
        run()
    except BaseException as error:
        # Transformers appears to handle KeyboardInterrupt (or BaseException)
        # internally in some places, which can re-raise a different error in the handler,
        # masking the root cause. We therefore check both the error itself and its context.
        if isinstance(error, KeyboardInterrupt) or isinstance(
            error.__context__, KeyboardInterrupt
        ):
            print()
            print("[red]Shutting down...[/]")
        else:
            raise

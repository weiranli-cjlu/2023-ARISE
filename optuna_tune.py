"""Optuna tuner for ARISE.

The tuner reuses ``run_one_trial`` from ``run.py`` so that tuning and normal
experiments share the same preprocessing, training, testing, AUC/AUPRC code and
score-file saving logic.

Example:
    python optuna_tune.py --dataset cora --n_trials 30

By default this script stores the Optuna study in a SQLite database under
``<results_dir>/optuna_arise.db``, shows Optuna's progress bar, trains every
candidate for 10 epochs, and writes a runnable shell script for the best
configuration.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shlex
import stat
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pandas as pd
import torch

from run import config_dict, config_key, ensure_dir, load_and_preprocess, run_one_trial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune ARISE hyperparameters with Optuna.")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--data_dir", type=str, default="~/datasets/GAD/mat")
    parser.add_argument("--results_dir", type=str, default="optuna_results",
                        help="Directory for Optuna DB, trial outputs, exported CSV and best scripts.")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL. Default: sqlite:///<results_dir>/optuna_arise.db")
    parser.add_argument("--study_name", type=str, default=None,
                        help="Optuna study name. Default: arise_<dataset>_<metric>")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=None,
                        help="Optional Optuna timeout in seconds.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auc_test_rounds", type=int, default=64,
                        help="Testing rounds during tuning. Increase for more stable final estimates.")
    parser.add_argument("--tune_metric", type=str, default="auc", choices=["auc", "auprc"],
                        help="Metric optimized by Optuna.")
    parser.add_argument("--direction", type=str, default="maximize", choices=["maximize", "minimize"])
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of runs written into the generated best-run script.")
    parser.add_argument("--show_progress_bar", action=argparse.BooleanOptionalAction, default=True,
                        help="Pass show_progress_bar to study.optimize. Default: True.")
    parser.add_argument("--load_if_exists", action=argparse.BooleanOptionalAction, default=True,
                        help="Reuse an existing study in the database. Default: True.")
    parser.add_argument("--best_script", type=str, default=None,
                        help="Path of generated best-run shell script. Default: <results_dir>/best_run_<dataset>.sh")
    parser.add_argument("--best_params_json", type=str, default=None,
                        help="Path of exported best params JSON. Default: <results_dir>/best_params_<dataset>.json")
    parser.add_argument("--trials_csv", type=str, default=None,
                        help="Path of exported Optuna trials CSV. Default: <results_dir>/optuna_trials_<dataset>.csv")
    parser.add_argument("--final_results_dir", type=str, default=None,
                        help="Results directory used in the generated best-run script. Default: <results_dir>/best_run_results")
    parser.add_argument("--subgraph_resample_interval", type=int, default=1,
                        help="Regenerate RWR subgraphs every N epochs/test rounds. Larger values reduce CPU cost.")
    parser.add_argument("--amp", action="store_true",
                        help="Use CUDA automatic mixed precision during tuning.")
    parser.add_argument("--allow_large_search", action="store_true",
                        help="Include larger, more expensive choices such as embedding_dim=256 and batch_size=512.")
    return parser.parse_args()


def default_storage_url(results_dir: Path, study_name: str) -> str:
    db_path = ensure_dir(results_dir) / f"{study_name}.db"
    return "sqlite:///" + str(db_path.resolve())


def sample_params(trial: Any, allow_large_search: bool = False) -> Dict[str, Any]:
    """Define the ARISE hyperparameter search space.

    The default space avoids the largest hidden dimension and batch size to keep
    tuning usable on memory-limited GPUs. Use --allow_large_search to recover the
    previous broader space.
    """
    embedding_choices = [32, 64, 128, 256] if allow_large_search else [32, 64, 128]
    batch_choices = [128, 256, 300, 512] if allow_large_search else [128, 256, 300]
    return {
        "num_epoch": trial.suggest_categorical("num_epoch", [10, 100, 200, 500]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
        "embedding_dim": trial.suggest_categorical("embedding_dim", embedding_choices),
        "batch_size": trial.suggest_categorical("batch_size", batch_choices),
        "subgraph_size": trial.suggest_int("subgraph_size", 3, 8),
        "readout": trial.suggest_categorical("readout", ["avg", "max", "min", "weighted_sum"]),
        "negsamp_ratio": trial.suggest_int("negsamp_ratio", 1, 3),
        "rwr_restart_prob": trial.suggest_float("rwr_restart_prob", 0.3, 0.95),
    }


def build_train_args(args: argparse.Namespace, params: Dict[str, Any], optuna_trial_number: int) -> SimpleNamespace:
    results_dir = ensure_dir(Path(os.path.expanduser(args.results_dir)))
    return SimpleNamespace(
        dataset=args.dataset,
        data_dir=args.data_dir,
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
        runs=1,
        embedding_dim=int(params["embedding_dim"]),
        num_epoch=int(params["num_epoch"]),
        drop_prob=0.0,
        batch_size=int(params["batch_size"]),
        subgraph_size=int(params["subgraph_size"]),
        readout=str(params["readout"]),
        auc_test_rounds=int(args.auc_test_rounds),
        negsamp_ratio=int(params["negsamp_ratio"]),
        rwr_restart_prob=float(params["rwr_restart_prob"]),
        save_model_path="best_model.pt",
        save_best_model=False,
        summary_csv=None,
        rerun_completed=True,
        subgraph_resample_interval=int(args.subgraph_resample_interval),
        amp=bool(args.amp),
    )


def shell_join(parts) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def write_best_run_script(args: argparse.Namespace, best_params: Dict[str, Any], best_value: float) -> Path:
    results_dir = ensure_dir(Path(os.path.expanduser(args.results_dir)))
    script_path = Path(os.path.expanduser(args.best_script)) if args.best_script else results_dir / f"best_run_{args.dataset}.sh"
    ensure_dir(script_path.parent)

    cmd = [
        "python", "run.py",
        "--dataset", args.dataset,
        "--runs", args.runs,
        "--num_epoch", best_params["num_epoch"],
        "--lr", best_params["lr"],
        "--weight_decay", best_params["weight_decay"],
        "--embedding_dim", best_params["embedding_dim"],
        "--batch_size", best_params["batch_size"],
        "--subgraph_size", best_params["subgraph_size"],
        "--readout", best_params["readout"],
        "--auc_test_rounds", args.auc_test_rounds,
        "--negsamp_ratio", best_params["negsamp_ratio"],
        "--rwr_restart_prob", best_params["rwr_restart_prob"],
        "--subgraph_resample_interval", args.subgraph_resample_interval,
    ]
    if args.amp:
        cmd.append("--amp")

    script = "\n".join([
        shell_join(cmd)
    ])
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script_path


def export_best_json(args: argparse.Namespace, study: Any, script_path: Path) -> Path:
    results_dir = ensure_dir(Path(os.path.expanduser(args.results_dir)))
    json_path = Path(os.path.expanduser(args.best_params_json)) if args.best_params_json else results_dir / f"best_params_{args.dataset}.json"
    ensure_dir(json_path.parent)
    payload = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset": args.dataset,
        "tune_metric": args.tune_metric,
        "best_value": study.best_value,
        "best_trial_number": study.best_trial.number,
        "best_params": study.best_params,
        "best_run_script": str(script_path),
        "storage": args.storage,
        "study_name": args.study_name,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path


def export_trials_csv(args: argparse.Namespace, study: Any) -> Path:
    results_dir = ensure_dir(Path(os.path.expanduser(args.results_dir)))
    csv_path = Path(os.path.expanduser(args.trials_csv)) if args.trials_csv else results_dir / f"optuna_trials_{args.dataset}.csv"
    ensure_dir(csv_path.parent)
    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "datetime_start", "datetime_complete"))
    # Use utf-8-sig so that Excel can open Chinese/Unicode paths cleanly.
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


def main() -> None:
    args = parse_args()
    results_dir = ensure_dir(Path(os.path.expanduser(args.results_dir)))
    args.study_name = args.study_name or f"arise_{args.dataset}_{args.tune_metric}"
    args.storage = args.storage or default_storage_url(results_dir, args.study_name)

    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError as exc:
        raise SystemExit("Optuna is not installed. Install it with: pip install optuna pandas tqdm") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading and preprocessing dataset once for all Optuna trials...")
    data = load_and_preprocess(args)

    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction=args.direction,
        sampler=sampler,
        load_if_exists=args.load_if_exists,
    )

    def objective(trial: Any) -> float:
        params = sample_params(trial, allow_large_search=args.allow_large_search)
        train_args = build_train_args(args, params, trial.number)
        cfg_key = config_key(train_args)
        run_seed = int(trial.number + 1)

        trial.set_user_attr("config", config_dict(train_args))
        trial.set_user_attr("config_key", cfg_key)
        trial.set_user_attr("seed", run_seed)

        try:
            result = run_one_trial(train_args, run_seed, device, "", cfg_key, is_tune=True, data=data)
            value = float(result[args.tune_metric])
            trial.set_user_attr("auc", float(result["auc"]))
            trial.set_user_attr("auprc", float(result["auprc"]))
            trial.set_user_attr("alpha", float(result["alpha"]))
            return value
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=args.show_progress_bar,
        # Do not stop the whole tuning process because one sampled configuration
        # is numerically invalid or runs into a recoverable runtime error.
        catch=(RuntimeError, ValueError),
    )

    best_script = write_best_run_script(args, study.best_params, float(study.best_value))
    best_json = export_best_json(args, study, best_script)
    trials_csv = export_trials_csv(args, study)

    print("\n================ OPTUNA SUMMARY ================")
    print("Study name:", args.study_name)
    print("Storage:", args.storage)
    print("Best trial:", study.best_trial.number)
    print(f"Best {args.tune_metric}:", f"{study.best_value:.8f}")
    print("Best params:", json.dumps(study.best_params, ensure_ascii=False, sort_keys=True))
    print("Best run script:", best_script)
    print("Best params JSON:", best_json)
    print("Trials CSV:", trials_csv)
    print("================================================")


if __name__ == "__main__":
    main()

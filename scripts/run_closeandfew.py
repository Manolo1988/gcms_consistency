#!/usr/bin/env python3
"""Reproducible entry point for the closed-set and few-shot paper experiments."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from evaluation import (  # noqa: E402
    FeatureTable,
    evaluate_method,
    paired_method_differences,
    save_evaluation,
)
from protocol import (  # noqa: E402
    ProtocolSpec,
    audit_metadata,
    build_protocol_manifests,
    load_metadata,
    validate_manifest,
)


def log(message: str) -> None:
    print(f"[closeandfew] {time.strftime('%Y-%m-%d %H:%M:%S')} | {message}", flush=True)


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _read_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _protocol_from_manifest(manifest: dict, args) -> ProtocolSpec:
    fewshot = manifest.get("fewshot", {})
    shots = _csv_ints(args.shots) if getattr(args, "shots", None) else tuple(fewshot.get("shots", (1, 3, 5)))
    episodes = getattr(args, "episodes", None) or int(fewshot.get("episodes", 100))
    return ProtocolSpec(
        shots=shots,
        episodes=episodes,
        min_novel_query_per_class=int(fewshot.get("min_query_per_class", 5)),
        future_query_only=bool(fewshot.get("future_query_only", True)),
        seed=int(manifest.get("seed", 42)),
    )


def _load_validated(metadata: str, manifest_path: str, args):
    manifest = _read_manifest(manifest_path)
    protocol = _protocol_from_manifest(manifest, args)
    df = load_metadata(metadata, protocol)
    validate_manifest(df, manifest, protocol)
    return df, manifest, protocol


def command_audit(args) -> None:
    protocol = ProtocolSpec()
    report = audit_metadata(load_metadata(args.metadata, protocol), protocol)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


def command_build(args) -> None:
    started = time.perf_counter()
    log(
        "building manifests "
        f"metadata={args.metadata} output={Path(args.output).resolve()} "
        f"folds={args.folds} shots={args.shots} episodes={args.episodes} seed={args.seed}"
    )
    protocol = ProtocolSpec(
        novel_classes_per_fold=args.novel_classes,
        max_product_folds=args.folds,
        shots=_csv_ints(args.shots),
        episodes=args.episodes,
        seed=args.seed,
    )
    explicit = None
    if args.novel_products:
        explicit = [_csv_strings(group) for group in args.novel_products]
    manifests = build_protocol_manifests(
        args.metadata, args.output, protocol, novel_product_sets=explicit
    )
    log(f"done: wrote {len(manifests)} manifests in {time.perf_counter() - started:.1f}s")
    print(f"wrote {len(manifests)} manifests to {Path(args.output).resolve()}", flush=True)


def command_baseline(args) -> None:
    from baselines import run_traditional_baselines

    started = time.perf_counter()
    log(f"baseline start manifest={args.manifest} output={Path(args.output).resolve()}")
    df, manifest, protocol = _load_validated(args.metadata, args.manifest, args)
    log(
        f"baseline loaded fold={manifest['fold_id']} "
        f"train={len(manifest['train_idx'])} val={len(manifest['val_idx'])} "
        f"test={len(manifest['test_idx'])} novel={len(manifest['novel_idx'])}"
    )
    paths = run_traditional_baselines(
        df, manifest, args.tensor_root, args.output, protocol,
        pca_components=args.pca_components, seed=args.seed,
    )
    for name, path in paths.items():
        print(f"{name}={path.resolve()}", flush=True)
    log(f"baseline done in {time.perf_counter() - started:.1f}s")


def command_train(args) -> None:
    from training import TrainingSpec, train_deep_method

    started = time.perf_counter()
    log(
        f"train start method={args.method} seed={args.seed} manifest={args.manifest} "
        f"epochs={args.epochs} device={args.device} output={Path(args.output).resolve()}"
    )
    df, manifest, protocol = _load_validated(args.metadata, args.manifest, args)
    training = TrainingSpec(
        method=args.method,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim,
        eval_interval=args.eval_interval,
        early_stop_patience=args.patience,
        num_workers=args.workers,
        device=args.device,
        seed=args.seed,
    )
    path = train_deep_method(
        df, manifest, args.tensor_root, args.output, protocol, training
    )
    log(f"train done method={args.method} seed={args.seed} feature={path.resolve()} in {time.perf_counter() - started:.1f}s")
    print(path.resolve(), flush=True)


def _parse_features(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--feature must use NAME=PATH")
        name, path = value.split("=", 1)
        output[name.strip()] = Path(path)
    return output


def _evaluate_one(
    df, manifest, protocol, name, feature_path, output, bootstrap, training_seed=None,
) -> pd.DataFrame:
    started = time.perf_counter()
    log(f"evaluate start method={name} feature={Path(feature_path).resolve()} output={Path(output).resolve()}")
    result, episodes = evaluate_method(
        df, FeatureTable.load(feature_path), manifest, protocol, name,
        bootstrap_repeats=bootstrap,
    )
    if training_seed is not None:
        result["training_seed"] = int(training_seed)
        episodes["training_seed"] = int(training_seed)
    save_evaluation(result, episodes, output)
    closed = result["closed_set"]
    few = result["few_shot"]
    log(
        f"evaluate done method={name} closed_acc={closed['accuracy']:.4f} "
        f"closed_macro_f1={closed['macro_f1']:.4f} fewshot_rows={len(few)} "
        f"in {time.perf_counter() - started:.1f}s"
    )
    return episodes, result


def command_evaluate(args) -> None:
    started = time.perf_counter()
    log(f"evaluate command start manifest={args.manifest} output={Path(args.output).resolve()}")
    df, manifest, protocol = _load_validated(args.metadata, args.manifest, args)
    all_episodes = []
    for name, path in _parse_features(args.feature).items():
        method_output = Path(args.output) / name
        episodes, _ = _evaluate_one(
            df, manifest, protocol, name, path, method_output, args.bootstrap
        )
        all_episodes.append(episodes)
    combined = pd.concat(all_episodes, ignore_index=True)
    combined.to_csv(Path(args.output) / "all_episodes.csv", index=False)
    if args.reference and len(all_episodes) > 1:
        paired_method_differences(combined, args.reference).to_csv(
            Path(args.output) / "paired_differences.csv", index=False
        )
    log(f"evaluate command done in {time.perf_counter() - started:.1f}s")


def command_matrix(args) -> None:
    from baselines import run_traditional_baselines
    from training import TrainingSpec, VALID_METHODS, train_deep_method

    manifests = sorted(Path(args.manifests).glob("products_*.json"))
    if not manifests:
        raise ValueError(f"no products_*.json manifests in {args.manifests}")
    methods = _csv_strings(args.methods)
    unknown = set(methods) - (VALID_METHODS | {
        "tic_pca_proto", "tic_pca_mahalanobis", "tic_plsda_latent"
    })
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    seeds = _csv_ints(args.seeds)
    started = time.perf_counter()
    log(
        f"matrix start folds={len(manifests)} methods={','.join(methods)} "
        f"seeds={','.join(map(str, seeds))} output={Path(args.output).resolve()}"
    )
    all_episode_tables = []
    all_closed_rows = []
    matrix_record = {
        "metadata": str(Path(args.metadata).resolve()),
        "manifests": [str(path.resolve()) for path in manifests],
        "methods": methods,
        "training_seeds": list(seeds),
        "model_selection": "closed-set validation Macro-F1 only",
        "final_test_used_for_selection": False,
    }
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    total_steps = len(manifests) * (
        len([m for m in methods if m.startswith("tic_")]) * len(seeds)
        + len([m for m in methods if m in VALID_METHODS]) * len(seeds)
    )
    step = 0
    for fold_number, manifest_path in enumerate(manifests, start=1):
        manifest = _read_manifest(manifest_path)
        protocol = _protocol_from_manifest(manifest, args)
        log(f"fold {fold_number}/{len(manifests)} load {manifest_path.name}")
        df = load_metadata(args.metadata, protocol)
        validate_manifest(df, manifest, protocol)
        fold_root = root / manifest["fold_id"]
        classical = [method for method in methods if method.startswith("tic_")]
        feature_paths = {}
        if classical:
            log(f"fold {manifest['fold_id']} baseline feature extraction start methods={','.join(classical)}")
            feature_paths = run_traditional_baselines(
                df, manifest, args.tensor_root, fold_root / "baseline_features",
                protocol, pca_components=args.pca_components, seed=protocol.seed,
            )
            log(f"fold {manifest['fold_id']} baseline feature extraction done")
        for seed in seeds:
            for method in classical:
                step += 1
                method_root = fold_root / method / f"seed_{seed}"
                log(f"step {step}/{total_steps} fold={manifest['fold_id']} method={method} seed={seed} evaluate")
                episodes, result = _evaluate_one(
                    df, manifest, protocol, method, feature_paths[method],
                    method_root, args.bootstrap, training_seed=seed,
                )
                all_episode_tables.append(episodes)
                all_closed_rows.append({
                    "fold_id": manifest["fold_id"], "method": method,
                    "training_seed": seed,
                    **{key: result["closed_set"][key] for key in (
                        "accuracy", "macro_f1", "balanced_accuracy", "n"
                    )},
                })
            for method in [item for item in methods if item in VALID_METHODS]:
                step += 1
                method_root = fold_root / method / f"seed_{seed}"
                log(
                    f"step {step}/{total_steps} fold={manifest['fold_id']} "
                    f"method={method} seed={seed} train+evaluate"
                )
                training = TrainingSpec(
                    method=method, epochs=args.epochs, batch_size=args.batch_size,
                    learning_rate=args.learning_rate, embedding_dim=args.embedding_dim,
                    eval_interval=args.eval_interval,
                    early_stop_patience=args.patience, num_workers=args.workers,
                    device=args.device, seed=seed,
                )
                feature_path = train_deep_method(
                    df, manifest, args.tensor_root, method_root,
                    protocol, training,
                )
                episodes, result = _evaluate_one(
                    df, manifest, protocol, method, feature_path,
                    method_root, args.bootstrap, training_seed=seed,
                )
                all_episode_tables.append(episodes)
                all_closed_rows.append({
                    "fold_id": manifest["fold_id"], "method": method,
                    "training_seed": seed,
                    **{key: result["closed_set"][key] for key in (
                        "accuracy", "macro_f1", "balanced_accuracy", "n"
                    )},
                })
    combined = pd.concat(all_episode_tables, ignore_index=True)
    combined.to_csv(root / "all_episodes.csv", index=False)
    pd.DataFrame(all_closed_rows).to_csv(root / "all_closed_set.csv", index=False)
    log(f"matrix wrote summaries: {root / 'all_episodes.csv'} and {root / 'all_closed_set.csv'}")
    if args.reference in set(combined["method"]):
        paired_method_differences(combined, args.reference).to_csv(
            root / "paired_differences.csv", index=False
        )
    (root / "matrix_config.json").write_text(
        json.dumps(matrix_record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(f"matrix done in {time.perf_counter() - started:.1f}s")


def add_shared_evaluation(parser) -> None:
    parser.add_argument("--shots", default=None, help="comma-separated shots; default from manifest")
    parser.add_argument("--episodes", type=int, default=None, help="default from manifest")


def add_training(parser) -> None:
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit", help="audit metadata and independent groups")
    audit.add_argument("--metadata", required=True)
    audit.add_argument("--output")
    audit.set_defaults(function=command_audit)

    build = commands.add_parser("build", help="create immutable outer-fold manifests")
    build.add_argument("--metadata", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--novel-products", action="append", help="explicit comma-separated fold")
    build.add_argument("--novel-classes", type=int, default=2)
    build.add_argument("--folds", type=int, default=4)
    build.add_argument("--shots", default="1,3,5")
    build.add_argument("--episodes", type=int, default=100)
    build.add_argument("--seed", type=int, default=42)
    build.set_defaults(function=command_build)

    baseline = commands.add_parser("baseline", help="fit train-only traditional baselines")
    baseline.add_argument("--metadata", required=True)
    baseline.add_argument("--manifest", required=True)
    baseline.add_argument("--tensor-root", default="new_prepared_data/tensors")
    baseline.add_argument("--output", required=True)
    baseline.add_argument("--pca-components", type=int, default=64)
    baseline.add_argument("--seed", type=int, default=42)
    add_shared_evaluation(baseline)
    baseline.set_defaults(function=command_baseline)

    train = commands.add_parser("train", help="train one fixed-backbone ablation")
    train.add_argument("--metadata", required=True)
    train.add_argument("--manifest", required=True)
    train.add_argument("--tensor-root", default="new_prepared_data/tensors")
    train.add_argument("--output", required=True)
    train.add_argument("--method", choices=["cnn_ce", "cnn_supcon", "cnn_supcon_batchadv"], required=True)
    train.add_argument("--seed", type=int, default=42)
    add_shared_evaluation(train)
    add_training(train)
    train.set_defaults(function=command_train)

    evaluate = commands.add_parser("evaluate", help="evaluate one or more frozen feature files")
    evaluate.add_argument("--metadata", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--feature", action="append", required=True, help="NAME=PATH")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--bootstrap", type=int, default=2000)
    evaluate.add_argument("--reference")
    add_shared_evaluation(evaluate)
    evaluate.set_defaults(function=command_evaluate)

    matrix = commands.add_parser("matrix", help="run the complete folds x methods x seeds matrix")
    matrix.add_argument("--metadata", required=True)
    matrix.add_argument("--manifests", required=True)
    matrix.add_argument("--tensor-root", default="new_prepared_data/tensors")
    matrix.add_argument("--output", required=True)
    matrix.add_argument("--methods", default="tic_pca_proto,tic_pca_mahalanobis,tic_plsda_latent,cnn_ce,cnn_supcon,cnn_supcon_batchadv")
    matrix.add_argument("--seeds", default="41,42,43,44,45")
    matrix.add_argument("--pca-components", type=int, default=64)
    matrix.add_argument("--bootstrap", type=int, default=2000)
    matrix.add_argument("--reference", default="tic_pca_proto")
    add_shared_evaluation(matrix)
    add_training(matrix)
    matrix.set_defaults(function=command_matrix)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

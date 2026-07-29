import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

METRICS = ["lddt", "bb_lddt", "tm_score", "rmsd"]


def compute_af3_metrics(preds, evals, name, model_ids=None):
    metrics = {}
    model_ids = list(range(5)) if model_ids is None else list(model_ids)

    top_model = None
    top_confidence = -1000
    for model_id in model_ids:
        # Load confidence file
        confidence_file = (
            Path(preds) / f"seed-1_sample-{model_id}" / "summary_confidences.json"
        )
        with confidence_file.open("r") as f:
            confidence_data = json.load(f)
            confidence = confidence_data["ranking_score"]
            if confidence > top_confidence:
                top_model = model_id
                top_confidence = confidence

        # Load eval file
        eval_file = Path(evals) / f"{name}_model_{model_id}.json"
        with eval_file.open("r") as f:
            eval_data = json.load(f)
            for metric_name in METRICS:
                if metric_name in eval_data:
                    metrics.setdefault(metric_name, []).append(eval_data[metric_name])

            if "dockq" in eval_data and eval_data["dockq"] is not None:
                metrics.setdefault("dockq_>0.23", []).append(
                    np.mean(
                        [float(v > 0.23) for v in eval_data["dockq"] if v is not None]
                    )
                )
                metrics.setdefault("dockq_>0.49", []).append(
                    np.mean(
                        [float(v > 0.49) for v in eval_data["dockq"] if v is not None]
                    )
                )
                metrics.setdefault("len_dockq_", []).append(
                    len([v for v in eval_data["dockq"] if v is not None])
                )

        eval_file = Path(evals) / f"{name}_model_{model_id}_ligand.json"
        with eval_file.open("r") as f:
            eval_data = json.load(f)
            if "lddt_pli" in eval_data:
                lddt_plis = [
                    x["score"] for x in eval_data["lddt_pli"]["assigned_scores"]
                ]
                for _ in eval_data["lddt_pli"][
                    "model_ligand_unassigned_reason"
                ].items():
                    lddt_plis.append(0)
                if not lddt_plis:
                    continue
                lddt_pli = np.mean([x for x in lddt_plis])
                metrics.setdefault("lddt_pli", []).append(lddt_pli)
                metrics.setdefault("len_lddt_pli", []).append(len(lddt_plis))

            if "rmsd" in eval_data:
                rmsds = [x["score"] for x in eval_data["rmsd"]["assigned_scores"]]
                for _ in eval_data["rmsd"]["model_ligand_unassigned_reason"].items():
                    rmsds.append(100)
                if not rmsds:
                    continue
                rmsd2 = np.mean([x < 2.0 for x in rmsds])
                rmsd5 = np.mean([x < 5.0 for x in rmsds])
                metrics.setdefault("rmsd<2", []).append(rmsd2)
                metrics.setdefault("rmsd<5", []).append(rmsd5)
                metrics.setdefault("len_rmsd", []).append(len(rmsds))

    # Get oracle
    oracle = {k: min(v) if k == "rmsd" else max(v) for k, v in metrics.items()}
    avg = {k: sum(v) / len(v) for k, v in metrics.items()}
    top_index = model_ids.index(top_model)
    top1 = {k: v[top_index] for k, v in metrics.items()}

    results = {}
    for metric_name in metrics:
        if metric_name.startswith("len_"):
            continue
        if metric_name == "lddt_pli":
            l = metrics["len_lddt_pli"][0]
        elif metric_name == "rmsd<2" or metric_name == "rmsd<5":
            l = metrics["len_rmsd"][0]
        elif metric_name == "dockq_>0.23" or metric_name == "dockq_>0.49":
            l = metrics["len_dockq_"][0]
        else:
            l = 1
        results[metric_name] = {
            "oracle": oracle[metric_name],
            "average": avg[metric_name],
            "top1": top1[metric_name],
            "len": l,
        }

    return results


def compute_chai_metrics(preds, evals, name, model_ids=None):
    metrics = {}
    model_ids = list(range(5)) if model_ids is None else list(model_ids)

    top_model = None
    top_confidence = -float("inf")
    for model_id in model_ids:
        # Load confidence file
        confidence_file = Path(preds) / f"scores.model_idx_{model_id}.npz"
        confidence_data = np.load(confidence_file)
        confidence = confidence_data["aggregate_score"].item()
        if confidence > top_confidence:
            top_model = model_id
            top_confidence = confidence

        # Load eval file
        eval_file = Path(evals) / f"{name}_model_{model_id}.json"
        with eval_file.open("r") as f:
            eval_data = json.load(f)
            for metric_name in METRICS:
                if metric_name in eval_data:
                    metrics.setdefault(metric_name, []).append(eval_data[metric_name])

            if "dockq" in eval_data and eval_data["dockq"] is not None:
                metrics.setdefault("dockq_>0.23", []).append(
                    np.mean(
                        [float(v > 0.23) for v in eval_data["dockq"] if v is not None]
                    )
                )
                metrics.setdefault("dockq_>0.49", []).append(
                    np.mean(
                        [float(v > 0.49) for v in eval_data["dockq"] if v is not None]
                    )
                )
                metrics.setdefault("len_dockq_", []).append(
                    len([v for v in eval_data["dockq"] if v is not None])
                )

        eval_file = Path(evals) / f"{name}_model_{model_id}_ligand.json"
        with eval_file.open("r") as f:
            eval_data = json.load(f)
            if "lddt_pli" in eval_data:
                lddt_plis = [
                    x["score"] for x in eval_data["lddt_pli"]["assigned_scores"]
                ]
                for _ in eval_data["lddt_pli"][
                    "model_ligand_unassigned_reason"
                ].items():
                    lddt_plis.append(0)
                if not lddt_plis:
                    continue
                lddt_pli = np.mean([x for x in lddt_plis])
                metrics.setdefault("lddt_pli", []).append(lddt_pli)
                metrics.setdefault("len_lddt_pli", []).append(len(lddt_plis))

            if "rmsd" in eval_data:
                rmsds = [x["score"] for x in eval_data["rmsd"]["assigned_scores"]]
                for _ in eval_data["rmsd"]["model_ligand_unassigned_reason"].items():
                    rmsds.append(100)
                if not rmsds:
                    continue
                rmsd2 = np.mean([x < 2.0 for x in rmsds])
                rmsd5 = np.mean([x < 5.0 for x in rmsds])
                metrics.setdefault("rmsd<2", []).append(rmsd2)
                metrics.setdefault("rmsd<5", []).append(rmsd5)
                metrics.setdefault("len_rmsd", []).append(len(rmsds))

    # Get oracle
    oracle = {k: min(v) if k == "rmsd" else max(v) for k, v in metrics.items()}
    avg = {k: sum(v) / len(v) for k, v in metrics.items()}
    top_index = model_ids.index(top_model)
    top1 = {k: v[top_index] for k, v in metrics.items()}

    results = {}
    for metric_name in metrics:
        if metric_name.startswith("len_"):
            continue
        if metric_name == "lddt_pli":
            l = metrics["len_lddt_pli"][0]
        elif metric_name == "rmsd<2" or metric_name == "rmsd<5":
            l = metrics["len_rmsd"][0]
        elif metric_name == "dockq_>0.23" or metric_name == "dockq_>0.49":
            l = metrics["len_dockq_"][0]
        else:
            l = 1
        results[metric_name] = {
            "oracle": oracle[metric_name],
            "average": avg[metric_name],
            "top1": top1[metric_name],
            "len": l,
        }

    return results


def compute_boltz_metrics(preds, evals, name, model_ids=None):
    metrics = {}
    model_ids = list(range(5)) if model_ids is None else list(model_ids)

    top_model = None
    top_confidence = -float("inf")
    for model_id in model_ids:
        # Load confidence file
        confidence_file = (
            Path(preds) / f"confidence_{Path(preds).name}_model_{model_id}.json"
        )
        with confidence_file.open("r") as f:
            confidence_data = json.load(f)
            confidence = confidence_data["confidence_score"]
            if confidence > top_confidence:
                top_model = model_id
                top_confidence = confidence

        # Load eval file
        eval_file = Path(evals) / f"{name}_model_{model_id}.json"
        with eval_file.open("r") as f:
            eval_data = json.load(f)
            for metric_name in METRICS:
                if metric_name in eval_data:
                    metrics.setdefault(metric_name, []).append(eval_data[metric_name])

            if "dockq" in eval_data and eval_data["dockq"] is not None:
                metrics.setdefault("dockq_>0.23", []).append(
                    np.mean(
                        [float(v > 0.23) for v in eval_data["dockq"] if v is not None]
                    )
                )
                metrics.setdefault("dockq_>0.49", []).append(
                    np.mean(
                        [float(v > 0.49) for v in eval_data["dockq"] if v is not None]
                    )
                )
                metrics.setdefault("len_dockq_", []).append(
                    len([v for v in eval_data["dockq"] if v is not None])
                )

        eval_file = Path(evals) / f"{name}_model_{model_id}_ligand.json"
        with eval_file.open("r") as f:
            eval_data = json.load(f)
            if "lddt_pli" in eval_data:
                lddt_plis = [
                    x["score"] for x in eval_data["lddt_pli"]["assigned_scores"]
                ]
                for _ in eval_data["lddt_pli"][
                    "model_ligand_unassigned_reason"
                ].items():
                    lddt_plis.append(0)
                if not lddt_plis:
                    continue
                lddt_pli = np.mean([x for x in lddt_plis])
                metrics.setdefault("lddt_pli", []).append(lddt_pli)
                metrics.setdefault("len_lddt_pli", []).append(len(lddt_plis))

            if "rmsd" in eval_data:
                rmsds = [x["score"] for x in eval_data["rmsd"]["assigned_scores"]]
                for _ in eval_data["rmsd"]["model_ligand_unassigned_reason"].items():
                    rmsds.append(100)
                if not rmsds:
                    continue
                rmsd2 = np.mean([x < 2.0 for x in rmsds])
                rmsd5 = np.mean([x < 5.0 for x in rmsds])
                metrics.setdefault("rmsd<2", []).append(rmsd2)
                metrics.setdefault("rmsd<5", []).append(rmsd5)
                metrics.setdefault("len_rmsd", []).append(len(rmsds))

    # Get oracle
    oracle = {k: min(v) if k == "rmsd" else max(v) for k, v in metrics.items()}
    avg = {k: sum(v) / len(v) for k, v in metrics.items()}
    top_index = model_ids.index(top_model)
    top1 = {k: v[top_index] for k, v in metrics.items()}

    results = {}
    for metric_name in metrics:
        if metric_name.startswith("len_"):
            continue
        if metric_name == "lddt_pli":
            l = metrics["len_lddt_pli"][0]
        elif metric_name == "rmsd<2" or metric_name == "rmsd<5":
            l = metrics["len_rmsd"][0]
        elif metric_name == "dockq_>0.23" or metric_name == "dockq_>0.49":
            l = metrics["len_dockq_"][0]
        else:
            l = 1
        results[metric_name] = {
            "oracle": oracle[metric_name],
            "average": avg[metric_name],
            "top1": top1[metric_name],
            "len": l,
        }

    return results


def eval_models(
    chai_preds,
    chai_evals,
    af3_preds,
    af3_evals,
    boltz_preds,
    boltz_evals,
    model_ids=None,
):
    model_ids = list(range(5)) if model_ids is None else list(model_ids)
    # Load preds and make sure we have predictions for all models
    chai_preds_names = {
        x.name.lower(): x
        for x in Path(chai_preds).iterdir()
        if x.is_dir() and not x.name.lower().startswith(".")
    }
    af3_preds_names = {
        x.name.lower(): x
        for x in Path(af3_preds).iterdir()
        if x.is_dir() and not x.name.lower().startswith(".")
    }
    boltz_preds_names = {
        x.name.lower(): x
        for x in Path(boltz_preds).iterdir()
        if x.is_dir() and not x.name.lower().startswith(".")
    }


    print("Chai preds", len(chai_preds_names))
    print("Af3 preds", len(af3_preds_names))
    print("Boltz preds", len(boltz_preds_names))


    common = (
        set(chai_preds_names.keys())
        & set(af3_preds_names.keys())
        & set(boltz_preds_names.keys())
    )

    # Remove examples in the validation set
    keys_to_remove = ["t1133", "h1134", "r1134s1", "t1134s2", "t1121", "t1123", "t1159"]
    for key in keys_to_remove:
        if key in common:
            common.remove(key)
    print("Common", len(common))

    # Create a dataframe with the following schema:
    # tool, name, metric, oracle, average, top1
    results = []
    for name in tqdm(common):
        try:
            af3_results = compute_af3_metrics(
                af3_preds_names[name],
                af3_evals,
                name,
                model_ids,
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error evaluating AF3 {name}: {e}")
            continue
        try:
            chai_results = compute_chai_metrics(
                chai_preds_names[name],
                chai_evals,
                name,
                model_ids,
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error evaluating Chai {name}: {e}")
            continue
        try:
            boltz_results = compute_boltz_metrics(
                boltz_preds_names[name],
                boltz_evals,
                name,
                model_ids,
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error evaluating Boltz {name}: {e}")
            continue

        for metric_name in af3_results:
            if metric_name in chai_results and metric_name in boltz_results:
                if (
                    (
                        af3_results[metric_name]["len"]
                        == chai_results[metric_name]["len"]
                    )
                    and (
                        af3_results[metric_name]["len"]
                        == boltz_results[metric_name]["len"]
                    )
                ):
                    results.append(
                        {
                            "tool": "AF3 oracle",
                            "target": name,
                            "metric": metric_name,
                            "value": af3_results[metric_name]["oracle"],
                        }
                    )
                    results.append(
                        {
                            "tool": "AF3 top-1",
                            "target": name,
                            "metric": metric_name,
                            "value": af3_results[metric_name]["top1"],
                        }
                    )
                    results.append(
                        {
                            "tool": "Chai-1 oracle",
                            "target": name,
                            "metric": metric_name,
                            "value": chai_results[metric_name]["oracle"],
                        }
                    )
                    results.append(
                        {
                            "tool": "Chai-1 top-1",
                            "target": name,
                            "metric": metric_name,
                            "value": chai_results[metric_name]["top1"],
                        }
                    )
                    results.append(
                        {
                            "tool": "Boltz-1 oracle",
                            "target": name,
                            "metric": metric_name,
                            "value": boltz_results[metric_name]["oracle"],
                        }
                    )
                    results.append(
                        {
                            "tool": "Boltz-1 top-1",
                            "target": name,
                            "metric": metric_name,
                            "value": boltz_results[metric_name]["top1"],
                        }
                    )

                else:
                    print(
                        "Different lengths",
                        name,
                        metric_name,
                        af3_results[metric_name]["len"],
                        chai_results[metric_name]["len"],
                        boltz_results[metric_name]["len"],
                    )
            else:
                print(
                    "Missing metric",
                    name,
                    metric_name,
                    metric_name in chai_results,
                    metric_name in boltz_results,
                )

    # Write the results to a file, ensure we only keep the target & metrics where we have all tools
    df = pd.DataFrame(results)
    return df


def eval_validity_checks(df):
    df = df[df["tool"].isin(["af3", "chai", "boltz1"])].copy()
    # Filter the dataframe to only include the targets in the validity checks
    name_mapping = {
        "af3": "AF3 top-1",
        "chai": "Chai-1 top-1",
        "boltz1": "Boltz-1 top-1",
    }
    top1 = df[df["model_idx"] == 0]
    top1 = top1[["tool", "pdb_id", "valid"]]
    top1["tool"] = top1["tool"].apply(lambda x: name_mapping[x])
    top1 = top1.rename(columns={"tool": "tool", "pdb_id": "target", "valid": "value"})
    top1["metric"] = "physical validity"
    top1["target"] = top1["target"].apply(lambda x: x.lower())
    top1 = top1[["tool", "target", "metric", "value"]]

    name_mapping = {
        "af3": "AF3 oracle",
        "chai": "Chai-1 oracle",
        "boltz1": "Boltz-1 oracle",
    }
    oracle = df[["tool", "model_idx", "pdb_id", "valid"]]
    oracle = oracle.groupby(["tool", "pdb_id"])["valid"].max().reset_index()
    oracle = oracle.rename(
        columns={"tool": "tool", "pdb_id": "target", "valid": "value"}
    )
    oracle["tool"] = oracle["tool"].apply(lambda x: name_mapping[x])
    oracle["metric"] = "physical validity"
    oracle = oracle[["tool", "target", "metric", "value"]]
    oracle["target"] = oracle["target"].apply(lambda x: x.lower())
    out = pd.concat([top1, oracle])
    return out


def bootstrap_ci(series, n_boot=1000, alpha=0.05):
    """
    Compute 95% bootstrap confidence intervals for the mean of 'series'.
    """
    n = len(series)
    boot_means = []
    # Perform bootstrap resampling
    for _ in range(n_boot):
        sample = series.sample(n, replace=True)
        boot_means.append(sample.mean())

    boot_means = np.array(boot_means)
    mean_val = np.mean(series)
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return mean_val, lower, upper


def plot_data(desired_tools, desired_metrics, df, dataset, filename):
    import matplotlib.pyplot as plt

    filtered_df = df[
        df["tool"].isin(desired_tools) & df["metric"].isin(desired_metrics)
    ]

    # Apply bootstrap to each (tool, metric) group
    boot_stats = filtered_df.groupby(["tool", "metric"])["value"].apply(bootstrap_ci)

    # boot_stats is a Series of tuples (mean, lower, upper). Convert to DataFrame:
    boot_stats = boot_stats.apply(pd.Series)
    boot_stats.columns = ["mean", "lower", "upper"]

    # Unstack to get a DataFrame suitable for plotting
    plot_data = boot_stats["mean"].unstack("tool")
    plot_data = plot_data.reindex(desired_metrics)

    lower_data = boot_stats["lower"].unstack("tool")
    lower_data = lower_data.reindex(desired_metrics)

    upper_data = boot_stats["upper"].unstack("tool")
    upper_data = upper_data.reindex(desired_metrics)

    # If you need a specific order of tools:
    tool_order = [
        "AF3 oracle",
        "AF3 top-1",
        "Chai-1 oracle",
        "Chai-1 top-1",
        "Boltz-1 oracle",
        "Boltz-1 top-1",
    ]
    plot_data = plot_data[tool_order]
    lower_data = lower_data[tool_order]
    upper_data = upper_data[tool_order]

    # Rename metrics
    renaming = {
        "lddt_pli": "Mean LDDT-PLI",
        "rmsd<2": "L-RMSD < 2A",
        "lddt": "Mean LDDT",
        "dockq_>0.23": "DockQ > 0.23",
        "physical validity": "Physical Validity",
    }
    plot_data = plot_data.rename(index=renaming)
    lower_data = lower_data.rename(index=renaming)
    upper_data = upper_data.rename(index=renaming)
    mean_vals = plot_data.values

    # Colors
    tool_colors = [
        "#994C00",  # AF3 oracle
        "#FFB55A",  # AF3 top-1
        "#931652",  # Chai-1 oracle
        "#FC8AD9",  # Chai-1 top-1
        "#188F52",  # Boltz-1 oracle
        "#86E935",  # Boltz-1 top-1
        "#004D80",  # Boltz-1x oracle
        "#55C2FF",  # Boltz-1x top-1
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(plot_data.index))
    bar_spacing = 0.015
    total_width = 0.7
    # Adjust width to account for the spacing
    width = (total_width - (len(tool_order) - 1) * bar_spacing) / len(tool_order)

    for i, tool in enumerate(tool_order):
        # Each subsequent bar moves over by width + bar_spacing
        offsets = x - (total_width - width) / 2 + i * (width + bar_spacing)
        # Extract the means and errors for this tool
        tool_means = plot_data[tool].values
        tool_yerr_lower = mean_vals[:, i] - lower_data.values[:, i]
        tool_yerr_upper = upper_data.values[:, i] - mean_vals[:, i]
        # Construct yerr array specifically for this tool
        tool_yerr = np.vstack([tool_yerr_lower, tool_yerr_upper])

        ax.bar(
            offsets,
            tool_means,
            width=width,
            color=tool_colors[i],
            label=tool,
            yerr=tool_yerr,
            capsize=2,
            error_kw={"elinewidth": 0.75},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_data.index, rotation=0)
    ax.set_ylabel("Value")
    ax.set_title(f"Performances on {dataset} with 95% CI (Bootstrap)")

    plt.tight_layout()
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.85), ncols=4, frameon=False)

    plt.savefig(filename)
    plt.show()


def _require_directory(path, label):
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"{label} directory not found: {path}")
    return path


def _resolve_split_paths(args, split):
    root = args.shared_root
    paths = {
        "chai_preds": args.chai_preds or root / "outputs" / split / "chai",
        "chai_evals": args.chai_evals or root / "evals" / split / "chai",
        "af3_preds": args.af3_preds or root / "outputs" / split / "af3",
        "af3_evals": args.af3_evals or root / "evals" / split / "af3",
        "boltz_preds": (
            args.boltz_preds
            or root / "outputs" / split / "boltz" / "predictions"
        ),
        "boltz_evals": (
            args.boltz_evals
            or args.local_eval_root / split / "boltz"
        ),
    }
    return {
        name: _require_directory(path, name.replace("_", " "))
        for name, path in paths.items()
    }


def _aggregate_split(args, split):
    paths = _resolve_split_paths(args, split)
    model_ids = list(range(args.num_samples))
    df = eval_models(
        paths["chai_preds"],
        paths["chai_evals"],
        paths["af3_preds"],
        paths["af3_evals"],
        paths["boltz_preds"],
        paths["boltz_evals"],
        model_ids=model_ids,
    )
    if df.empty:
        raise RuntimeError(
            f"No complete {split} targets could be aggregated. "
            "Check target names, model counts, confidence files, and eval JSON files."
        )

    if args.physical_checks is not None:
        physical_path = args.physical_checks.expanduser().resolve()
        if not physical_path.is_file():
            raise FileNotFoundError(
                f"Physical checks CSV not found: {physical_path}"
            )
        physical_df = eval_validity_checks(pd.read_csv(physical_path))
        df = pd.concat([df, physical_df], ignore_index=True)

    output_name = "results_test.csv" if split == "test" else "results_casp.csv"
    plot_name = "plot_test.pdf" if split == "test" else "plot_casp.pdf"
    dataset_name = "PDB Test" if split == "test" else "CASP15"
    df = df.reset_index(drop=True)
    df.to_csv(args.output / output_name, index=False)

    desired_tools = [
        "AF3 oracle",
        "AF3 top-1",
        "Chai-1 oracle",
        "Chai-1 top-1",
        "Boltz-1 oracle",
        "Boltz-1 top-1",
    ]
    desired_metrics = ["lddt", "dockq_>0.23", "lddt_pli", "rmsd<2"]
    if args.physical_checks is not None:
        desired_metrics.append("physical validity")
    available = set(df["metric"])
    desired_metrics = [metric for metric in desired_metrics if metric in available]
    if not desired_metrics:
        raise RuntimeError(f"No plottable metrics were produced for {split}")
    plot_data(
        desired_tools,
        desired_metrics,
        df,
        dataset_name,
        args.output / plot_name,
    )
    print(  # noqa: T201
        f"Wrote {split} aggregate results to {args.output}",
        flush=True,
    )


def parse_args():
    legacy_root = os.environ.get("BOLTZ_SHARED_RESULTS_ROOT") or os.environ.get(
        "BOLTZ_RESULTS_ROOT"
    )
    legacy_output = os.environ.get("BOLTZ_AGGREGATE_OUTPUT")
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate local Boltz predictions/evaluations with shared "
            "Chai and AF3 benchmark results"
        )
    )
    parser.add_argument(
        "--shared-root",
        type=Path,
        default=Path(legacy_root) if legacy_root else None,
    )
    parser.add_argument(
        "--local-eval-root",
        type=Path,
        default=(
            Path(os.environ["BOLTZ_LOCAL_EVAL_ROOT"])
            if os.environ.get("BOLTZ_LOCAL_EVAL_ROOT")
            else None
        ),
    )
    parser.add_argument("--split", choices=("test", "casp15", "all"), default="all")
    parser.add_argument("--boltz-preds", type=Path)
    parser.add_argument("--boltz-evals", type=Path)
    parser.add_argument("--chai-preds", type=Path)
    parser.add_argument("--chai-evals", type=Path)
    parser.add_argument("--af3-preds", type=Path)
    parser.add_argument("--af3-evals", type=Path)
    parser.add_argument("--physical-checks", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(legacy_output) if legacy_output else None,
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=int(os.environ.get("BOLTZ_AGGREGATE_NUM_SAMPLES", "5")),
    )
    args = parser.parse_args()

    if args.shared_root is None:
        parser.error("--shared-root or BOLTZ_SHARED_RESULTS_ROOT is required")
    args.shared_root = args.shared_root.expanduser().resolve()
    if not args.shared_root.is_dir():
        parser.error(f"shared results root not found: {args.shared_root}")
    if args.local_eval_root is None:
        args.local_eval_root = args.shared_root / "evals_local"
    args.local_eval_root = args.local_eval_root.expanduser().resolve()
    if args.num_samples < 1:
        parser.error("--num-samples must be at least 1")

    if args.split != "all":
        env_overrides = {
            "boltz_preds": os.environ.get("BOLTZ_PREDICTIONS_DIR"),
            "boltz_evals": os.environ.get("BOLTZ_LOCAL_EVAL_DIR"),
            "chai_preds": os.environ.get("BOLTZ_CHAI_PREDICTIONS_DIR"),
            "chai_evals": os.environ.get("BOLTZ_CHAI_EVAL_DIR"),
            "af3_preds": os.environ.get("BOLTZ_AF3_PREDICTIONS_DIR"),
            "af3_evals": os.environ.get("BOLTZ_AF3_EVAL_DIR"),
            "physical_checks": os.environ.get("BOLTZ_PHYSICAL_OUTPUT"),
        }
        for name, value in env_overrides.items():
            if getattr(args, name) is None and value:
                setattr(args, name, Path(value))

    overrides = (
        args.boltz_preds,
        args.boltz_evals,
        args.chai_preds,
        args.chai_evals,
        args.af3_preds,
        args.af3_evals,
        args.physical_checks,
    )
    if args.split == "all" and any(path is not None for path in overrides):
        parser.error(
            "explicit prediction/evaluation paths require --split test or --split casp15"
        )

    if args.output is None:
        args.output = args.shared_root / "aggregate_local"
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    return args


def main():
    args = parse_args()
    splits = ("test", "casp15") if args.split == "all" else (args.split,)
    for split in splits:
        _aggregate_split(args, split)


if __name__ == "__main__":
    main()

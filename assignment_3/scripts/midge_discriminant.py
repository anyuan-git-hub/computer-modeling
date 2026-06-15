from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import chi2

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "midges.csv"
OUTPUT_DIR = ROOT / "output"


CLASS_TO_CODE = {"Apf": 1, "Af": 2}
CODE_TO_CLASS = {1: "Apf", 2: "Af"}
LOSS_RATIOS = [1, 2, 3, 5]


@dataclass
class DistanceModel:
    mean_apf: np.ndarray
    mean_af: np.ndarray
    pooled_cov: np.ndarray
    inv_cov: np.ndarray
    w: np.ndarray
    c: float

    def score(self, x: np.ndarray) -> np.ndarray:
        z = np.atleast_2d(x)
        return z @ self.w + self.c

    def squared_distances(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = np.atleast_2d(x)
        d1 = z - self.mean_apf
        d2 = z - self.mean_af
        dist_apf = np.einsum("ij,jk,ik->i", d1, self.inv_cov, d1)
        dist_af = np.einsum("ij,jk,ik->i", d2, self.inv_cov, d2)
        return dist_apf, dist_af

    def predict(self, x: np.ndarray) -> np.ndarray:
        dist_apf, dist_af = self.squared_distances(x)
        return np.where(dist_apf <= dist_af, 1, 2)


def read_data() -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray]:
    rows: list[dict[str, str]] = []
    with DATA_FILE.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    apf = np.array(
        [[float(r["antenna"]), float(r["wing"])] for r in rows if r["group"] == "Apf"],
        dtype=float,
    )
    af = np.array(
        [[float(r["antenna"]), float(r["wing"])] for r in rows if r["group"] == "Af"],
        dtype=float,
    )
    unknown = np.array(
        [
            [float(r["antenna"]), float(r["wing"])]
            for r in rows
            if r["group"] == "Unknown"
        ],
        dtype=float,
    )
    return rows, apf, af, unknown


def covariance(x: np.ndarray) -> np.ndarray:
    return np.cov(x, rowvar=False, ddof=1)


def train_distance_model(apf: np.ndarray, af: np.ndarray) -> DistanceModel:
    mean_apf = apf.mean(axis=0)
    mean_af = af.mean(axis=0)
    cov_apf = covariance(apf)
    cov_af = covariance(af)
    pooled = ((len(apf) - 1) * cov_apf + (len(af) - 1) * cov_af) / (
        len(apf) + len(af) - 2
    )
    inv_cov = np.linalg.inv(pooled)
    w = inv_cov @ (mean_apf - mean_af)
    c = -0.5 * (mean_apf + mean_af) @ w
    return DistanceModel(mean_apf, mean_af, pooled, inv_cov, w, float(c))


def box_m_test(groups: list[np.ndarray]) -> dict[str, float]:
    k = len(groups)
    p = groups[0].shape[1]
    n_total = sum(len(g) for g in groups)
    covs = [covariance(g) for g in groups]
    pooled = sum((len(g) - 1) * s for g, s in zip(groups, covs)) / (n_total - k)

    m_stat = (n_total - k) * math.log(np.linalg.det(pooled))
    m_stat -= sum((len(g) - 1) * math.log(np.linalg.det(s)) for g, s in zip(groups, covs))
    correction = ((2 * p * p + 3 * p - 1) / (6 * (p + 1) * (k - 1))) * (
        sum(1 / (len(g) - 1) for g in groups) - 1 / (n_total - k)
    )
    chi_square = (1 - correction) * m_stat
    df = p * (p + 1) * (k - 1) / 2
    p_value = 1 - chi2.cdf(chi_square, df)
    return {
        "M": float(m_stat),
        "correction": float(correction),
        "chi_square": float(chi_square),
        "df": float(df),
        "p_value": float(p_value),
    }


def leave_one_out(apf: np.ndarray, af: np.ndarray) -> dict[str, object]:
    x_all = np.vstack([apf, af])
    y_all = np.array([1] * len(apf) + [2] * len(af), dtype=int)
    predictions = []

    for i in range(len(x_all)):
        mask = np.ones(len(x_all), dtype=bool)
        mask[i] = False
        model = train_distance_model(x_all[mask & (y_all == 1)], x_all[mask & (y_all == 2)])
        predictions.append(int(model.predict(x_all[i])[0]))

    predictions_arr = np.array(predictions, dtype=int)
    wrong = np.where(predictions_arr != y_all)[0]
    return {
        "predictions": predictions_arr,
        "truth": y_all,
        "wrong_indices": wrong,
        "error_count": int(len(wrong)),
        "error_rate": float(len(wrong) / len(y_all)),
    }


def bayes_predictions(
    model: DistanceModel,
    x: np.ndarray,
    prior_apf: float,
    prior_af: float,
    loss_ratio: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    dist_apf, dist_af = model.squared_distances(x)
    log_posterior_odds = math.log(prior_apf / prior_af) - 0.5 * (dist_apf - dist_af)
    threshold = math.log(1 / loss_ratio)
    predictions = np.where(log_posterior_odds >= threshold, 1, 2)
    return predictions, log_posterior_odds, float(threshold)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def label_axes(ax: plt.Axes) -> None:
    ax.set_xlabel("触角长")
    ax.set_ylabel("翅长")
    ax.grid(True, alpha=0.25, linewidth=0.8)


def plot_raw_scatter(apf: np.ndarray, af: np.ndarray, unknown: np.ndarray, model: DistanceModel) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.scatter(apf[:, 0], apf[:, 1], s=72, marker="o", color="#1f77b4", label="Apf 样本")
    ax.scatter(af[:, 0], af[:, 1], s=72, marker="s", color="#d62728", label="Af 样本")
    ax.scatter(
        unknown[:, 0],
        unknown[:, 1],
        s=120,
        marker="*",
        color="#2ca02c",
        edgecolor="black",
        linewidth=0.7,
        label="待判样本",
    )
    ax.scatter(*model.mean_apf, s=160, marker="X", color="#08306b", label="Apf 均值")
    ax.scatter(*model.mean_af, s=160, marker="X", color="#7f0000", label="Af 均值")
    for i, point in enumerate(unknown, 1):
        ax.annotate(f"U-{i}", point + np.array([0.008, 0.008]), fontsize=10)
    label_axes(ax)
    ax.set_title("样本分布")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "raw_scatter.png", dpi=220)
    plt.close(fig)


def plot_distance_boundary(
    apf: np.ndarray,
    af: np.ndarray,
    unknown: np.ndarray,
    model: DistanceModel,
    unknown_pred: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    x_min, x_max = 1.08, 1.60
    xs = np.linspace(x_min, x_max, 200)
    ys = -(model.w[0] * xs + model.c) / model.w[1]
    ax.plot(xs, ys, color="#111111", linewidth=2.0, label="距离判别边界")
    ax.fill_between(xs, ys, 2.12, color="#1f77b4", alpha=0.08, label="Apf 区域")
    ax.fill_between(xs, 1.58, ys, color="#d62728", alpha=0.08, label="Af 区域")
    ax.scatter(apf[:, 0], apf[:, 1], s=70, marker="o", color="#1f77b4", label="Apf 样本")
    ax.scatter(af[:, 0], af[:, 1], s=70, marker="s", color="#d62728", label="Af 样本")
    for i, point in enumerate(unknown, 1):
        color = "#1f77b4" if unknown_pred[i - 1] == 1 else "#d62728"
        ax.scatter(point[0], point[1], s=135, marker="*", color=color, edgecolor="black")
        ax.annotate(f"U-{i}: {CODE_TO_CLASS[int(unknown_pred[i-1])]}", point + np.array([0.008, 0.008]), fontsize=9)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(1.58, 2.12)
    label_axes(ax)
    ax.set_title("线性距离判别边界")
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "distance_boundary.png", dpi=220)
    plt.close(fig)


def add_cov_ellipse(
    ax: plt.Axes,
    mean: np.ndarray,
    cov: np.ndarray,
    color: str,
    label: str,
    n_std: float = 1.5,
) -> None:
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        edgecolor=color,
        facecolor=color,
        alpha=0.12,
        linewidth=2,
        label=label,
    )
    ax.add_patch(ell)


def plot_covariance_ellipses(apf: np.ndarray, af: np.ndarray, model: DistanceModel) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.scatter(apf[:, 0], apf[:, 1], s=70, color="#1f77b4", label="Apf 样本")
    ax.scatter(af[:, 0], af[:, 1], s=70, color="#d62728", marker="s", label="Af 样本")
    add_cov_ellipse(ax, model.mean_apf, covariance(apf), "#1f77b4", "Apf 协方差椭圆")
    add_cov_ellipse(ax, model.mean_af, covariance(af), "#d62728", "Af 协方差椭圆")
    add_cov_ellipse(ax, (model.mean_apf + model.mean_af) / 2, model.pooled_cov, "#444444", "合并协方差椭圆", n_std=1.2)
    ax.scatter(*model.mean_apf, marker="X", s=160, color="#08306b")
    ax.scatter(*model.mean_af, marker="X", s=160, color="#7f0000")
    label_axes(ax)
    ax.set_title("协方差椭圆")
    ax.legend(fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "covariance_ellipses.png", dpi=220)
    plt.close(fig)


def plot_mahalanobis_distances(distance_rows: list[dict[str, object]]) -> None:
    labels = [str(r["sample_id"]) for r in distance_rows]
    d_apf = np.array([float(r["dist_apf"]) for r in distance_rows])
    d_af = np.array([float(r["dist_af"]) for r in distance_rows])
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.bar(x - width / 2, d_apf, width=width, color="#1f77b4", label="到 Apf")
    ax.bar(x + width / 2, d_af, width=width, color="#d62728", label="到 Af")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("平方 Mahalanobis 距离")
    ax.set_title("待判样本距离比较")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)
    for i, (a, b) in enumerate(zip(d_apf, d_af)):
        winner = "Apf" if a <= b else "Af"
        ax.text(i, max(a, b) + 0.2, winner, ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mahalanobis_distances.png", dpi=220)
    plt.close(fig)


def plot_loocv(
    apf: np.ndarray,
    af: np.ndarray,
    loo: dict[str, object],
    model: DistanceModel,
) -> None:
    x_all = np.vstack([apf, af])
    y_all = np.array([1] * len(apf) + [2] * len(af), dtype=int)
    pred = np.array(loo["predictions"], dtype=int)
    wrong = np.array(loo["wrong_indices"], dtype=int)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    xs = np.linspace(1.08, 1.60, 200)
    ys = -(model.w[0] * xs + model.c) / model.w[1]
    ax.plot(xs, ys, color="#111111", linewidth=1.8, label="全样本边界")
    ax.scatter(apf[:, 0], apf[:, 1], s=70, color="#1f77b4", marker="o", label="Apf 样本")
    ax.scatter(af[:, 0], af[:, 1], s=70, color="#d62728", marker="s", label="Af 样本")

    for idx in wrong:
        ax.scatter(
            x_all[idx, 0],
            x_all[idx, 1],
            s=260,
            facecolors="none",
            edgecolors="#ff7f0e",
            linewidth=2.5,
            label="留一误判" if idx == wrong[0] else None,
        )
        truth = CODE_TO_CLASS[int(y_all[idx])]
        assigned = CODE_TO_CLASS[int(pred[idx])]
        ax.annotate(
            f"{truth} -> {assigned}",
            x_all[idx] + np.array([0.008, 0.010]),
            fontsize=9,
            color="#8c4a00",
        )
    label_axes(ax)
    ax.set_xlim(1.08, 1.60)
    ax.set_ylim(1.58, 2.12)
    ax.set_title("留一交叉验证")
    ax.legend(fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "loocv_result.png", dpi=220)
    plt.close(fig)


def plot_bayes_sensitivity(bayes_rows: list[dict[str, object]]) -> None:
    ratios = [str(r["loss_ratio"]) for r in bayes_rows]
    sample_ids = ["U-1", "U-2", "U-3"]
    matrix = np.array(
        [[1 if r[sample] == "Apf" else 0 for sample in sample_ids] for r in bayes_rows],
        dtype=float,
    ).T

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    im = ax.imshow(matrix, cmap=matplotlib.colors.ListedColormap(["#d62728", "#1f77b4"]), aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(ratios)))
    ax.set_xticklabels(ratios)
    ax.set_yticks(np.arange(len(sample_ids)))
    ax.set_yticklabels(sample_ids)
    ax.set_xlabel("损失倍数：Apf->Af / Af->Apf")
    ax.set_title("Bayes 分类敏感性")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            label = "Apf" if matrix[i, j] == 1 else "Af"
            ax.text(j, i, label, ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, ticks=[0.25, 0.75], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(["Af", "Apf"])
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "bayes_sensitivity.png", dpi=220)
    plt.close(fig)


def plot_bayes_scores_thresholds(
    unknown_rows: list[dict[str, object]],
    bayes_rows: list[dict[str, object]],
) -> None:
    sample_ids = [str(r["sample_id"]) for r in unknown_rows]
    log_odds = np.array([float(r["bayes_log_odds"]) for r in unknown_rows])
    ratios = [float(r["loss_ratio_value"]) for r in bayes_rows]
    thresholds = [math.log(1 / r) for r in ratios]
    threshold_styles = {
        1: {"color": "#6b6b6b", "linestyle": "-", "linewidth": 1.6},
        2: {"color": "#1f77b4", "linestyle": "--", "linewidth": 1.8},
        3: {"color": "#ff7f0e", "linestyle": "-.", "linewidth": 1.8},
        5: {"color": "#d62728", "linestyle": ":", "linewidth": 2.2},
    }

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    x = np.arange(len(sample_ids))
    ax.scatter(x, log_odds, s=110, color="#2ca02c", marker="D", label="待判样本得分")
    for i, value in enumerate(log_odds):
        ax.text(i + 0.03, value + 0.05, f"{value:.2f}", fontsize=9)
    for ratio, threshold in zip(ratios, thresholds):
        style = threshold_styles[int(ratio)]
        ax.axhline(threshold, label=f"阈值 {ratio:.0f}:1", **style)
    ax.axhline(0, color="#999999", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(sample_ids)
    ax.set_ylabel("Bayes 后验优势对数")
    ax.set_title("Bayes 得分与移动阈值")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "bayes_scores_thresholds.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    rows, apf, af, unknown = read_data()
    model = train_distance_model(apf, af)
    box = box_m_test([apf, af])

    x_train = np.vstack([apf, af])
    y_train = np.array([1] * len(apf) + [2] * len(af), dtype=int)
    train_pred = model.predict(x_train)
    resub_error_count = int(np.sum(train_pred != y_train))
    resub_error_rate = float(resub_error_count / len(y_train))

    loo = leave_one_out(apf, af)
    unknown_pred = model.predict(unknown)
    unknown_dist_apf, unknown_dist_af = model.squared_distances(unknown)

    prior_apf = len(apf) / (len(apf) + len(af))
    prior_af = len(af) / (len(apf) + len(af))
    bayes_unknown_equal, bayes_log_odds, _ = bayes_predictions(
        model, unknown, prior_apf, prior_af, loss_ratio=1
    )

    distance_rows = []
    for i, point in enumerate(unknown, 1):
        distance_rows.append(
            {
                "sample_id": f"U-{i}",
                "antenna": f"{point[0]:.2f}",
                "wing": f"{point[1]:.2f}",
                "dist_apf": f"{unknown_dist_apf[i-1]:.6f}",
                "dist_af": f"{unknown_dist_af[i-1]:.6f}",
                "distance_prediction": CODE_TO_CLASS[int(unknown_pred[i - 1])],
                "bayes_log_odds": f"{bayes_log_odds[i-1]:.6f}",
                "bayes_equal_loss_prediction": CODE_TO_CLASS[int(bayes_unknown_equal[i - 1])],
            }
        )
    write_csv(OUTPUT_DIR / "distance_unknown.csv", distance_rows)

    bayes_rows = []
    for ratio in LOSS_RATIOS:
        preds, _, threshold = bayes_predictions(model, unknown, prior_apf, prior_af, ratio)
        bayes_rows.append(
            {
                "loss_ratio": f"{ratio}:1",
                "loss_ratio_value": ratio,
                "threshold": f"{threshold:.6f}",
                "U-1": CODE_TO_CLASS[int(preds[0])],
                "U-2": CODE_TO_CLASS[int(preds[1])],
                "U-3": CODE_TO_CLASS[int(preds[2])],
            }
        )
    write_csv(OUTPUT_DIR / "bayes_sensitivity.csv", bayes_rows)

    covariance_rows = [
        {
            "group": "Apf",
            "mean_antenna": f"{model.mean_apf[0]:.6f}",
            "mean_wing": f"{model.mean_apf[1]:.6f}",
            "cov_11": f"{covariance(apf)[0,0]:.8f}",
            "cov_12": f"{covariance(apf)[0,1]:.8f}",
            "cov_22": f"{covariance(apf)[1,1]:.8f}",
        },
        {
            "group": "Af",
            "mean_antenna": f"{model.mean_af[0]:.6f}",
            "mean_wing": f"{model.mean_af[1]:.6f}",
            "cov_11": f"{covariance(af)[0,0]:.8f}",
            "cov_12": f"{covariance(af)[0,1]:.8f}",
            "cov_22": f"{covariance(af)[1,1]:.8f}",
        },
    ]
    write_csv(OUTPUT_DIR / "class_statistics.csv", covariance_rows)

    summary = {
        "n_apf": len(apf),
        "n_af": len(af),
        "n_unknown": len(unknown),
        "mean_apf": model.mean_apf.tolist(),
        "mean_af": model.mean_af.tolist(),
        "pooled_cov": model.pooled_cov.tolist(),
        "linear_w": model.w.tolist(),
        "linear_c": model.c,
        "box_m_chi_square": box["chi_square"],
        "box_m_df": box["df"],
        "box_m_p_value": box["p_value"],
        "resub_error_count": resub_error_count,
        "resub_error_rate": resub_error_rate,
        "loocv_error_count": loo["error_count"],
        "loocv_error_rate": loo["error_rate"],
        "loocv_wrong_indices_1_based": (np.array(loo["wrong_indices"], dtype=int) + 1).tolist(),
        "distance_unknown_labels": ", ".join(CODE_TO_CLASS[int(x)] for x in unknown_pred),
        "bayes_equal_loss_unknown_labels": ", ".join(CODE_TO_CLASS[int(x)] for x in bayes_unknown_equal),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(
        OUTPUT_DIR / "results_summary.csv",
        [
            {"metric": "Box M chi-square", "value": f"{summary['box_m_chi_square']:.6f}"},
            {"metric": "Box M p-value", "value": f"{summary['box_m_p_value']:.6f}"},
            {"metric": "Resubstitution error", "value": f"{resub_error_count}/15"},
            {"metric": "LOOCV error", "value": f"{loo['error_count']}/15"},
            {"metric": "Distance unknown labels", "value": summary["distance_unknown_labels"]},
        ],
    )

    plot_raw_scatter(apf, af, unknown, model)
    plot_distance_boundary(apf, af, unknown, model, unknown_pred)
    plot_covariance_ellipses(apf, af, model)
    plot_mahalanobis_distances(distance_rows)
    plot_loocv(apf, af, loo, model)
    plot_bayes_sensitivity(bayes_rows)
    plot_bayes_scores_thresholds(distance_rows, bayes_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

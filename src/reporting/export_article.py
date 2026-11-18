"""Export article evidence, Markdown tables and figures from existing experiment files."""

import csv
import hashlib
import importlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from src.paths import ROOT, storage_root


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    return "\n".join(lines + ["| " + " | ".join(map(str, row)) + " |" for row in rows]) + "\n"


def main() -> None:
    storage = storage_root()
    data = ROOT / "data/unified"
    final = storage / "runs/final_test_all"
    output = ROOT / "docs/artigo"
    output.mkdir(parents=True, exist_ok=True)
    manifest = load(data / "manifest.json")
    plan, ranking = load(final / "evaluation_plan.json"), load(final / "ranking.json")
    assert sha(data / "manifest.json") == plan["manifest_sha256"]
    splits = ("train", "val", "test")
    sources = sorted({item["source"] for item in manifest})
    source_ids = {source: f"S{i}" for i, source in enumerate(sources, 1)}
    group_sets = [{m["group"] for m in manifest if m["split"] == split} for split in splits]
    assert all(not group_sets[i] & group_sets[j] for i in range(3) for j in range(i + 1, 3))
    sizes = []
    for item in manifest:
        with Image.open(data / item["image"]) as image:
            sizes.append(image.size)
        labels = data / item["split"] / "labels" / (Path(item["image"]).stem + ".txt")
        assert len(labels.read_text().splitlines()) == item["objects"]
    stats = {}
    for split in splits:
        entries = [m for m in manifest if m["split"] == split]
        stats[split] = {
            "images": len(entries),
            "boxes": sum(m["objects"] for m in entries),
            "groups": len({m["group"] for m in entries}),
            "negative_images": sum(m["objects"] == 0 for m in entries),
            "max_boxes_per_image": max(m["objects"] for m in entries),
        }
    assert stats["test"]["images"] == plan["test_images"]
    evidence = {
        "date": "2026-11-17",
        "sources": source_ids,
        "split_statistics": stats,
        "image_dimensions": {
            "min_width": min(w for w, h in sizes),
            "max_width": max(w for w, h in sizes),
            "min_height": min(h for w, h in sizes),
            "max_height": max(h for w, h in sizes),
            "most_common": Counter(sizes).most_common(10),
        },
        "evaluation_plan": plan,
        "ranking": ranking,
        "training": {},
        "input_hashes": {
            "data/unified/manifest.json": sha(data / "manifest.json"),
            "runs/final_test_all/ranking.json": sha(final / "ranking.json"),
        },
        "configs": {
            name: load(ROOT / "configs" / name)
            for name in (
                "yolov8_grid.json",
                "yolov8_preprocessing.json",
                "yolov8_preprocessing_local.json",
            )
        },
    }
    blocks = [
        "# Tabelas verificadas para o artigo\n\n"
        "Geradas de arquivos de resultados e do manifesto, sem novo treinamento ou inferência. "
        "Métricas em porcentagem; diferenças em pontos percentuais (pp).\n"
    ]
    blocks += [
        "## 1. Dataset final\n",
        table(
            [
                "Conjunto",
                "Imagens",
                "% imagens",
                "Caixas",
                "Grupos",
                "Negativas",
                "Máx. caixas/foto",
            ],
            [
                [
                    split,
                    s["images"],
                    f"{100 * s['images'] / len(manifest):.2f}",
                    s["boxes"],
                    s["groups"],
                    s["negative_images"],
                    s["max_boxes_per_image"],
                ]
                for split, s in stats.items()
            ],
        ),
    ]
    blocks += [
        f"Total: **{len(manifest)} imagens e {sum(m['objects'] for m in manifest)} caixas**.\n"
    ]
    by_source = []
    for source in sources:
        by_source.append(
            [source_ids[source]]
            + [
                sum(m["source"] == source and m["split"] == split for m in manifest)
                for split in splits
            ]
            + [
                sum(m["source"] == source for m in manifest),
                sum(m["objects"] for m in manifest if m["source"] == source),
            ]
        )
    blocks += [
        "## 2. Distribuição final por origem\n",
        table(
            ["Origem", "Treino", "Validação", "Teste", "Total imagens", "Total caixas"], by_source
        ),
    ]
    blocks += ["\n".join(f"- **{source_ids[source]}:** `{source}`" for source in sources) + "\n"]
    baseline = next(
        row["test_mAP50_95"] for row in ranking if row["experiment"] == "preproc__original"
    )
    blocks += [
        "## 3. Ranking final no teste\n",
        table(
            [
                "Rank",
                "Modelo",
                "P",
                "R",
                "mAP50",
                "mAP50–95",
                "Val mAP50–95",
                "Δ teste vs RGB local (pp)",
            ],
            [
                [
                    r["rank"],
                    r["experiment"],
                    *[
                        f"{100 * r[k]:.2f}"
                        for k in (
                            "test_precision",
                            "test_recall",
                            "test_mAP50",
                            "test_mAP50_95",
                            "val_mAP50_95",
                        )
                    ],
                    f"{100 * (r['test_mAP50_95'] - baseline):+.2f}"
                    if r["family"] == "preprocessing"
                    else "não controlado",
                ]
                for r in ranking
            ],
        ),
    ]
    training_rows, latency_rows = [], []
    histories = {}
    for row in ranking:
        name = row["experiment"]
        if name.startswith("preproc__"):
            folder = storage / "runs/preprocessing_local/grid" / name.removeprefix("preproc__")
            result = load(folder / "result.json")
            with (folder / "results.csv").open(encoding="utf-8", newline="") as handle:
                history = [
                    {k.strip(): float(v) for k, v in entry.items()}
                    for entry in csv.DictReader(handle)
                ]
            histories[name] = history
            best_epoch = max(history, key=lambda entry: entry["metrics/mAP50-95(B)"])["epoch"]
            training_rows.append(
                [
                    name,
                    int(history[-1]["epoch"]),
                    int(best_epoch),
                    f"{history[-1]['time'] / 3600:.2f}",
                    *[
                        f"{100 * result['metrics'][key]:.2f}"
                        for key in (
                            "metrics/precision(B)",
                            "metrics/recall(B)",
                            "metrics/mAP50(B)",
                            "metrics/mAP50-95(B)",
                        )
                    ],
                ]
            )
            args = "\n".join(
                line.split(":", 1)[0] + ": <runtime path omitted>"
                if line.split(":", 1)[0] in {"data", "project", "save_dir"}
                else line
                for line in (folder / "args.yaml").read_text(encoding="utf-8").splitlines()
            )
            evidence["training"][name] = {
                "request": result["request"],
                "validation_metrics": result["metrics"],
                "effective_args_yaml": args,
                "history": history,
            }
            checkpoint = folder / "weights/best.pt"
            logical = f"runs/preprocessing_local/grid/{folder.name}"
            for filename in ("result.json", "results.csv", "args.yaml"):
                evidence["input_hashes"][f"{logical}/{filename}"] = sha(folder / filename)
        else:
            folder = storage / "models" / name
            result = load(folder / "result.json")
            checkpoint = folder / "best.pt"
            evidence["training"][name] = {k: v for k, v in result.items() if k != "best"}
            evidence["input_hashes"][f"models/{name}/result.json"] = sha(folder / "result.json")
        assert sha(checkpoint) == plan["checkpoints"][name], f"Checkpoint changed: {name}"
        test_result = load(final / name / "test_metrics.json")
        evidence["training"][name]["test_result"] = test_result
        for key, value in test_result["metrics"].items():
            if key == "metrics/mAP50-95(B)":
                assert value == row["test_mAP50_95"]
        latency_rows.append(
            [
                name,
                *[
                    f"{test_result['speed_ms_per_image'][key]:.3f}"
                    for key in ("preprocess", "inference", "postprocess")
                ],
            ]
        )
    blocks += [
        "## 4. Histórico local e validação do best.pt\n",
        "Época de pico = maior mAP50–95 registrado em results.csv. Não confundir com "
        "uma nova avaliação do best.pt. Tempo = coluna time do treinador; não é tempo "
        "ativo de GPU e pode incluir pausas/suspensões.\n",
        table(
            [
                "Modelo",
                "Épocas",
                "Época de pico",
                "Tempo registrado (h)",
                "Val P",
                "Val R",
                "Val mAP50",
                "Val mAP50–95",
            ],
            training_rows,
        ),
        "O histórico de épocas e duração do exp_05 não está disponível neste snapshot.\n",
    ]
    blocks += [
        "## 5. Tempos da avaliação no teste (ms/imagem)\n",
        "Tempos internos do validador, batch 1, GPU device=0. Não incluem exportação dos "
        "filtros, leitura total do dataset, carregamento do modelo ou servidor HTTP. "
        "Não converter diretamente em FPS da aplicação em CPU.\n",
        table(["Modelo", "Preprocessamento YOLO", "Inferência", "Pós-processamento"], latency_rows),
    ]
    (output / "tabelas.md").write_text("\n".join(blocks), encoding="utf-8")
    (output / "dados_verificados.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    mpl = importlib.import_module("matplotlib")
    mpl.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    labels = [r["experiment"].replace("preproc__", "") for r in ranking][::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(labels, [100 * r["test_mAP50_95"] for r in ranking][::-1], color="#346c62")
    ax.set(xlim=(0, 100), xlabel="mAP50–95 no teste (%)", title="Ranking dos 12 modelos avaliados")
    for i, row in enumerate(ranking[::-1]):
        ax.text(100 * row["test_mAP50_95"] + 1, i, f"{100 * row['test_mAP50_95']:.2f}", va="center")
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output / f"ranking_teste.{extension}", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 6))
    for index, (name, history) in enumerate(sorted(histories.items())):
        ax.plot(
            [e["epoch"] for e in history],
            [100 * e["metrics/mAP50-95(B)"] for e in history],
            label=name.removeprefix("preproc__"),
            color=plt.get_cmap("tab20")(index),
            linestyle="--" if index % 2 else "-",
            linewidth=1.5,
        )
    ax.set(
        xlabel="Época",
        ylabel="mAP50–95 de validação (%)",
        ylim=(0, 100),
        title="Evolução da validação — grade local (uma semente)",
    )
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output / f"curvas_validacao.{extension}", dpi=180)
    plt.close(fig)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Article evidence exported to {output}")


if __name__ == "__main__":
    main()

"""Flag annotation risks and generate a local, filterable visual review queue."""

import collections
import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/unified"
OUT = ROOT / "data/annotation_review"


def main() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    duplicates = json.loads((DATA / "duplicates.json").read_text(encoding="utf-8"))
    conflict = {d["kept"] for d in duplicates if d["annotations_differ"]}
    OUT.mkdir(exist_ok=True)
    rows = []
    for m in manifest:
        path = DATA / m["image"]
        label = path.parent.parent / "labels" / f"{path.stem}.txt"
        raw = [list(map(float, line.split())) for line in label.read_text().splitlines()]
        flags = []
        boxes = []
        with Image.open(path) as im:
            width, height = im.size
        if not raw:
            flags.append("empty_annotation")
        for line in raw:
            if len(line) != 5 or line[0] != 0 or not np.isfinite(line).all():
                flags.append("invalid")
                continue
            x, y, w, h = line[1:]
            corners = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
            if w <= 0 or h <= 0 or min(corners) < -1e-6 or max(corners) > 1 + 1e-6:
                flags.append("invalid")
            scale = 640 / max(width, height)
            if min(w * width, h * height) * scale < 3:
                flags.append("tiny_at_640")
            if w * h > 0.75:
                flags.append("large_box")
            boxes.append(line[1:])
        for i, (x, y, w, h) in enumerate(boxes):
            for a, b, c, d in boxes[i + 1 :]:
                inter = max(0, min(x + w / 2, a + c / 2) - max(x - w / 2, a - c / 2)) * max(
                    0, min(y + h / 2, b + d / 2) - max(y - h / 2, b - d / 2)
                )
                union = w * h + c * d - inter
                if union > 0 and inter / union > 0.85:
                    flags.append("high_overlap")
        if m["id"] in conflict:
            flags.append("duplicate_annotation_difference")
        rows.append(
            {**m, "width": width, "height": height, "boxes": boxes, "flags": sorted(set(flags))}
        )
    selected: set[int] = set()
    rng = random.Random(42)
    for source in sorted({r["source"] for r in rows}):
        for split in ["train", "val", "test"]:
            candidates = [r for r in rows if r["source"] == source and r["split"] == split]
            selected.update(r["id"] for r in rng.sample(candidates, min(2, len(candidates))))
    for r in rows:
        r["sample"] = r["id"] in selected
    counts = collections.Counter(flag for r in rows for flag in r["flags"])
    summary = {
        "images": len(rows),
        "boxes": sum(len(r["boxes"]) for r in rows),
        "flagged_images": sum(bool(r["flags"]) for r in rows),
        "flags": dict(counts),
        "random_sample": sorted(selected),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (OUT / "queue.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "source", "split", "image", "flags", "decision", "notes"])
        for r in rows:
            if r["flags"] or r["sample"]:
                writer.writerow(
                    [r["id"], r["source"], r["split"], r["image"], ";".join(r["flags"]), "", ""]
                )
    # Embed metadata so the review works from file:// without a web server.
    html = """<!doctype html><meta charset="utf-8"><title>Revisão das anotações</title>
<style>body{font:16px system-ui;margin:24px;background:#eee}svg{max-width:100%;
height:70vh;background:#111}button,select{padding:8px;margin:4px}pre{white-space:pre-wrap}</style>
<h1>Revisão das anotações</h1><p>Sinais são candidatos, não erros confirmados.
Registre decisões e observações em queue.csv. Os arquivos não são alterados por esta página.</p>
<select id="filter"><option value="sample">Amostra por origem e divisão</option>
<option value="all">Todas as imagens</option><option value="flagged">Com sinais</option></select>
<button id="prev">Anterior</button><button id="next">Próxima</button>
<label><input type="checkbox" id="boxes" checked> Mostrar caixas</label>
<pre id="info"></pre><div id="viewer"></div><script>
const rows=DATA;let index=0;const select=document.querySelector('#filter');
for(const flag of [...new Set(rows.flatMap(r=>r.flags))]){
 const o=document.createElement('option');o.value=flag;o.textContent=flag;select.append(o)}
function render(){const f=select.value;const list=rows.filter(r=>f==='all'||
 (f==='sample'?r.sample:f==='flagged'?r.flags.length:r.flags.includes(f)));
 if(!list.length){document.querySelector('#info').textContent='Nenhum caso';
 document.querySelector('#viewer').innerHTML='';return}
 index=(index+list.length)%list.length;const r=list[index];
 document.querySelector('#info').textContent=`${index+1}/${list.length} | ID ${r.id}\n`+
 `${r.source}\n${r.split} | ${r.objects} caixas | ${r.flags.join(', ')}\n${r.image}`;
 let boxes=document.querySelector('#boxes').checked?r.boxes.map(([x,y,w,h])=>
 `<rect x="${(x-w/2)*r.width}" y="${(y-h/2)*r.height}" width="${w*r.width}" `+
 `height="${h*r.height}" fill="none" stroke="#00ff55" stroke-width="1.5"/>`).join(''):'';
 document.querySelector('#viewer').innerHTML=`<svg viewBox="0 0 ${r.width} ${r.height}">
 <image href="../unified/${r.image}" width="${r.width}" height="${r.height}"/>${boxes}</svg>`;}
select.onchange=()=>{index=0;render()};document.querySelector('#boxes').onchange=render;
document.querySelector('#prev').onclick=()=>{index--;render()};
document.querySelector('#next').onclick=()=>{index++;render()};render();</script>"""
    (OUT / "index.html").write_text(html.replace("DATA", json.dumps(rows)), encoding="utf-8")
    samples = [r for r in rows if r["sample"]]
    samples.extend(r for r in rows if r["id"] in {209, 222, 285, 251, 100, 177})
    for page in range(0, len(samples), 12):
        sheet = Image.new("RGB", (1200, 900), "white")
        draw = ImageDraw.Draw(sheet)
        for k, r in enumerate(samples[page : page + 12]):
            with Image.open(DATA / r["image"]) as original:
                im = original.convert("RGB")
            im.thumbnail((390, 265))
            overlay = ImageDraw.Draw(im)
            for x, y, w, h in r["boxes"]:
                overlay.rectangle(
                    (
                        (x - w / 2) * im.width,
                        (y - h / 2) * im.height,
                        (x + w / 2) * im.width,
                        (y + h / 2) * im.height,
                    ),
                    outline="lime",
                )
            left, top = (k % 3) * 400, (k // 3) * 225
            im.thumbnail((390, 195))
            sheet.paste(im, (left, top + 25))
            draw.text((left, top), f"ID {r['id']} {r['split']} boxes={r['objects']}", fill="black")
        sheet.save(OUT / f"sample_{page // 12}.jpg")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

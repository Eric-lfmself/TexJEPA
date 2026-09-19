"""Build Tables IV-XIII from measured result records; never fill paper numbers.

Table VI is long-form model x frequency condition. Tables IV/V/VII/IX/X/XII
are wide matrices. Missing values remain NR, unsupported classes remain visible
in raw records. Synthetic provenance is repeated in every table and manifest.
"""
import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path

SIGMAS = (0.05, 0.1, 0.2, 0.3)
_TABLE_NAMES = tuple(f"table_{number}" for number in
                     ("IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII")) + ("robustness_long",)
_MANAGED_FILES = {f"{name}.{suffix}" for name in _TABLE_NAMES for suffix in ("csv", "md")}
_MANAGED_FILES.update(("manifest.json", "INDEX.md"))


def _sigma(value):
    sigma = float(value)
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("Gaussian table severity must be finite and nonnegative")
    return 0.0 if sigma == 0 else sigma


def _sigma_column(value):
    sigma = _sigma(value)
    # Preserve paper column names; repr preserves distinct custom floats such
    # as .051/.052 that two-decimal formatting would silently merge.
    return f"sigma={sigma:.2f}" if sigma in SIGMAS else f"sigma={sigma!r}"


def _finite(value):
    return not isinstance(value, float) or math.isfinite(value)


def _format(value):
    if value is None or not _finite(value):
        return "NR"
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _format_cell(field, value):
    if field == "noise_sigma" and value is not None:
        sigma = _sigma(value)
        rounded = f"{sigma:.6f}"
        return rounded if float(rounded) == sigma else repr(sigma)
    return _format(value)


def sanitize(value):
    """Strict JSON: nonfinite numerical estimates become null, not invented zeros."""
    if isinstance(value, dict):
        return {k:sanitize(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [sanitize(v) for v in value]
    return None if isinstance(value,float) and not math.isfinite(value) else value


def _matrix(rows, metric, include_clean=True, include_drift=False, extra_sigmas=()):
    groups = {}
    sigmas = set(SIGMAS) | {_sigma(value) for value in extra_sigmas}
    for row in rows:
        if row["perturbation"] not in ("clean", "gaussian_noise"):
            continue
        key = (row["model"], row["protocol"])
        result = groups.setdefault(key, {"Model":key[0], "Protocol":key[1], "Evidence":row["evidence"]})
        if row["perturbation"] == "clean":
            if include_clean:
                result["Clean"] = row[metric]
        else:
            severity = _sigma(row["severity"])
            sigmas.add(severity)
            result[_sigma_column(severity)] = row[metric]
            if include_drift and float(row["severity"]) == 0.05:
                result["Drift@0.05"] = row["drift"]
    fields = ["Model", "Protocol"] + (["Clean"] if include_clean else [])
    fields += [_sigma_column(sigma) for sigma in sorted(sigmas)]
    if include_drift:
        fields += ["Drift@0.05"]
    return fields + ["Evidence"], list(groups.values())


def _write_table(folder, name, title, fields, rows, evidence):
    with (folder/f"{name}.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        writer.writerows({k:_format_cell(k, row.get(k)) for k in fields} for row in rows)
    lines=[f"# {title}", "", f"Evidence: **{evidence}**. NR = unavailable/undefined; no paper values substituted.", ""]
    if evidence == "synthetic_smoke":
        lines += ["Tiny random models and dummy images: these values validate the pipeline only.", ""]
    lines += ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(k, row.get(k)).replace("|", "\\|").replace("\n", " ") for k in fields) + " |")
    (folder/f"{name}.md").write_text("\n".join(lines)+"\n")
    return {"table":name,"title":title,"rows":len(rows),"csv":f"{name}.csv","markdown":f"{name}.md"}



def token_summary_rows(records):
    """Export measured token summaries without inventing Table XIII conclusions.

    Maps and signed saliency remain in raw results; mean patch-token drift is
    kept separate from pooled-feature drift and carries its image/class context.
    """
    rows = []
    for record in records:
        base = {key: record.get(key) for key in ("model", "protocol", "image_id", "class_index", "evidence")}
        if record.get("status"):
            rows.append({**base, "intervention": "token_drift", "status": record["status"]})
            continue
        for condition in ("noise", "lesion_occlusion"):
            detail = record.get(condition)
            if not isinstance(detail, dict):
                continue
            rows.append({**base, "intervention": f"token_drift_{condition}",
                         "noise_sigma": record.get("noise_sigma") if condition == "noise" else None,
                         "mean_token_drift": detail.get("mean_drift"),
                         "valid_tokens": detail.get("valid_tokens"),
                         "total_tokens": detail.get("total_tokens"),
                         "status": "measured_weak_spatial_diagnostic"})
    return rows


def _generate_tables(results, output):
    metadata = results["metadata"]
    evidence=metadata["evidence"]
    if evidence not in ("synthetic_smoke", "measured_local"):
        raise ValueError("Unknown/missing evidence provenance")
    for section in ("robustness", "lesion", "interventions", "mitigation", "token_diagnostics"):
        for record in results.get(section, []):
            if record.get("evidence") != evidence:
                raise ValueError(f"Mixed or missing evidence in {section}")
    lesion_keys = [(r["model"], r.get("protocol", "linear")) for r in results.get("lesion", [])]
    if len(set(lesion_keys)) != len(lesion_keys):
        raise ValueError("Duplicate lesion model/protocol condition")
    if any(protocol != "linear" for _, protocol in lesion_keys):
        raise ValueError("Current Table VIII/XI schema requires linear-probe lesion scores")
    interventions = [*results.get("interventions", []), *token_summary_rows(results.get("token_diagnostics", []))]
    intervention_keys = []
    for record in interventions:
        sigma = record.get("noise_sigma")
        intervention_keys.append((record["model"], record.get("protocol"),
                                  record["intervention"], None if sigma is None else _sigma(sigma),
                                  record.get("image_id"), record.get("class_index")))
    if len(intervention_keys) != len(set(intervention_keys)):
        raise ValueError("Duplicate intervention model/protocol/method/sigma condition")
    rows=results.get("robustness",[])
    keys=set()
    for row in rows:
        if row.get("evidence") != evidence:
            raise ValueError("Mixed evidence cannot be collapsed into one result table")
        severity = row["severity"]
        if row["perturbation"] == "gaussian_noise":
            severity = _sigma(severity)
        elif row["perturbation"] == "clean":
            severity = 0.0
        key=(row["model"],row["protocol"],row["perturbation"],str(severity))
        if key in keys:
            raise ValueError(f"Duplicate experimental condition: {key}")
        keys.add(key)
    configured_sigmas = metadata.get("config", {}).get("perturbations", {}).get("gaussian_noise", [])
    extra_sigmas = {_sigma(value) for value in configured_sigmas}
    extra_sigmas.update(_sigma(row["severity"]) for row in rows if row["perturbation"] == "gaussian_noise")
    tables=[]
    def emit(name,title,fields,data):
        tables.append(_write_table(output,name,title,fields,data,evidence))
    linear=[r for r in rows if r["protocol"]=="linear"]
    main=[r for r in linear if r.get("group")=="main"]
    post=[r for r in linear if r.get("group") in ("post","external") or r.get("paper_model")=="MAE-H/300"]
    emit("table_IV","Table IV — Gaussian noise AUROC",*_matrix(main,"auroc",extra_sigmas=extra_sigmas))
    emit("table_V","Table V — Gaussian noise cosine drift",*_matrix(main,"drift",False,extra_sigmas=extra_sigmas))
    frequency=[{"Model":r["model"],"Condition":r["perturbation"],"Severity":r["severity"],
                "AUROC":r["auroc"],"Delta AUROC":r["delta_auroc"],"Delta status":r.get("delta_status"),"Drift":r["drift"],
                "Parameters":r["parameters"],"Evidence":r["evidence"]}
               for r in main if r["perturbation"] in ("clean","low_pass","band_stop","band_pass")]
    emit("table_VI","Table VI — Frequency conditions",["Model","Condition","Severity","AUROC","Delta AUROC","Delta status","Drift","Parameters","Evidence"],frequency)
    emit("table_VII","Table VII — Probe capacity and partial FT",*_matrix([r for r in rows if r.get("group")=="main"],"auroc",True,True,extra_sigmas=extra_sigmas))
    lesions=results.get("lesion",[])
    lesion_rows=[{"Model":r["model"],"Delta lesion":r.get("delta_lesion"),"CI lower":r.get("ci_low"),
                  "CI upper":r.get("ci_high"),"Images":r.get("n_images"),"Pairs":r.get("n_pairs"),
                  "Skipped":r.get("n_skipped"),"Evidence":evidence} for r in lesions]
    emit("table_VIII","Table VIII — Classification-aligned lesion occlusion",["Model","Delta lesion","CI lower","CI upper","Images","Pairs","Skipped","Evidence"],lesion_rows)
    emit("table_IX","Table IX — Post-training Gaussian AUROC",*_matrix(post,"auroc",extra_sigmas=extra_sigmas))
    emit("table_X","Table X — Post-training cosine drift",*_matrix(post,"drift",False,extra_sigmas=extra_sigmas))
    lesion_lookup={r["model"]:r for r in lesions}
    trade=[]
    for model in dict.fromkeys(r["model"] for r in post):
        model_rows=[r for r in post if r["model"]==model]
        noise=[r for r in model_rows if r["perturbation"]=="gaussian_noise"]
        low=[r["auroc"] for r in noise if float(r["severity"]) in (0.05,0.1) and r["auroc"] is not None and _finite(r["auroc"])]
        drift=next((r["drift"] for r in noise if float(r["severity"])==0.2),None)
        trade.append({"Model":model,"Delta lesion":lesion_lookup.get(model,{}).get("delta_lesion"),
                      "Best low-noise AUROC":max(low) if low else None,"Drift@0.20":drift,"Evidence":evidence})
    emit("table_XI","Table XI — Measured trade-off matrix",["Model","Delta lesion","Best low-noise AUROC","Drift@0.20","Evidence"],trade)
    emit("table_XII","Table XII — Golden-checkpoint-shaped matrix (synthetic surrogates when flagged)",*_matrix(post,"auroc",extra_sigmas=extra_sigmas))
    fields=["model","protocol","intervention","noise_sigma","clean_auroc","noise_auroc",
            "raw_drift","adapted_drift","image_id","class_index","mean_token_drift",
            "valid_tokens","total_tokens","status","evidence"]
    emit("table_XIII","Table XIII — Additional diagnostic measurements",fields,interventions)
    if rows:
        fields=list(rows[0])
        emit("robustness_long","All robustness conditions and validity counts",fields,rows)
    manifest={"metadata":metadata,"tables":tables,
              "result_sha256":hashlib.sha256(json.dumps(sanitize(results),sort_keys=True,ensure_ascii=False).encode()).hexdigest()}
    (output/"manifest.json").write_text(json.dumps(sanitize(manifest),indent=2,ensure_ascii=False,allow_nan=False))
    (output/"INDEX.md").write_text("# Generated tables\n\nEvidence: **"+evidence+"**\n\n"+"\n".join(f"- [{t['title']}]({t['markdown']}) — {t['rows']} rows" for t in tables)+"\n")
    return manifest


def build_tables(results, output_dir):
    """Publish one complete table set, retaining unrelated output-directory files.

    Generate and copy retained files in a sibling staging directory first. A
    failed generation leaves the previous directory untouched; a failed directory
    promotion restores it. Only this tool's fixed filenames are replaced/removed.
    """
    output = Path(output_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    staged, previous = temporary / "tables", temporary / "previous"
    staged.mkdir()
    preserve_previous = False
    try:
        manifest = _generate_tables(results, staged)
        if output.exists():
            for item in output.iterdir():
                if item.name in _MANAGED_FILES:
                    continue
                destination = staged / item.name
                if item.is_dir() and not item.is_symlink():
                    shutil.copytree(item, destination, symlinks=True)
                else:
                    shutil.copy2(item, destination, follow_symlinks=False)
            shutil.copystat(output, staged, follow_symlinks=False)
            output.replace(previous)
        try:
            staged.replace(output)
        except BaseException:
            if previous.exists():
                preserve_previous = True
                try:
                    previous.replace(output)
                except OSError as exc:
                    raise RuntimeError(f"Table publication and rollback failed; previous output remains at {previous}") from exc
                preserve_previous = False
            raise
        return manifest
    finally:
        if not preserve_previous:
            shutil.rmtree(temporary, ignore_errors=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",type=Path)
    parser.add_argument("--output",type=Path,default=Path("outputs/tables"))
    args=parser.parse_args()
    result=build_tables(json.loads(args.input.read_text()),args.output)
    print(json.dumps({"tables":len(result["tables"]),"output":str(args.output)},indent=2))

if __name__=="__main__":
    main()

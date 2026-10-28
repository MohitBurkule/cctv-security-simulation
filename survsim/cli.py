"""survsim command line."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .core.geometry import pixel_height
from .core.loader import load_scenario
from .simulator import Simulator

app = typer.Typer(add_completion=False, help="Multi-camera surveillance simulator")
console = Console()

# IEC 62676-4 DORI bands: pixels per metre of target, converted here to the pixel
# height of a 1.7 m person so the number matches what a spec actually asks for.
DORI_BANDS = ((400 * 1.7, "identify"), (250 * 1.7, "recognise"), (125 * 1.7, "observe"),
              (25 * 1.7, "detect"), (12 * 1.7, "monitor"))


def _ground_probe(cam, distance_m: float):
    """A ground point `distance_m` in front of the camera, along its own heading."""
    import numpy as np

    yaw = np.radians(cam.mount.yaw)
    return (
        cam.mount.pos[0] + distance_m * float(np.cos(yaw)),
        cam.mount.pos[1] + distance_m * float(np.sin(yaw)),
        0.0,
    )


def _dori(px_height: float) -> str:
    for threshold, name in DORI_BANDS:
        if px_height >= threshold:
            return name
    return "-"


@app.command()
def validate(scenario: Path) -> None:
    """Load a scenario, check every reference resolves, and print a summary."""
    s = load_scenario(scenario)
    console.print(f"[green]OK[/] {s.name}  seed={s.seed}  engine={s.engine}")

    t = Table("camera", "type", "res", "fps", "hfov", "px/person @20m", "DORI")
    for cam in s.cameras:
        ph = pixel_height(cam, _ground_probe(cam, 20.0), 1.7)
        t.add_row(
            cam.id, cam.type, "x".join(map(str, cam.capture.res)),
            f"{cam.capture.fps:g}", f"{cam.hfov_deg:g}deg", f"{ph:.0f}", _dori(ph),
        )
    console.print(t)
    console.print(
        f"zones={len(s.environment.zones)} hero={len(s.population.hero)} "
        f"ticks={s.n_ticks} duration={s.duration_s:g}s"
    )


@app.command()
def run(
    scenario: Path,
    seed: int | None = typer.Option(None),
    engine: str = typer.Option("mock"),
    out: Path = typer.Option(Path("out/run")),
    video: bool = typer.Option(True, help="write a top-down MP4 (mock backend only)"),
) -> None:
    """Run a scenario and write ground truth."""
    sim = Simulator.from_yaml(scenario, seed=seed, engine=engine)
    result = sim.record(out, topdown_video=video)
    console.print(f"[green]done[/] ticks={result['ticks']} events={result['events']}")
    for name, path in result["paths"].items():
        console.print(f"  {name:22s} {path}")


@app.command()
def rig(scenario: Path, cell_m: float = typer.Option(1.0), out: Path | None = typer.Option(None)) -> None:
    """Analyse camera coverage: overlap, blind spots and transit priors."""
    import json

    from .director import rig as rig_tools

    s = load_scenario(scenario)
    report = rig_tools.summary(s.cameras, s.environment, cell_m=cell_m)

    console.print(
        f"[bold]{s.name}[/]  blind={report['blind_fraction']:.1%}  "
        f"mean cameras/cell={report['mean_cameras_per_cell']:.2f}  "
        f"2+ cameras={report['cells_seen_by_2plus']:.1%}"
    )

    ids = report["cameras"]
    t = Table("overlap", *ids)
    for i, row in enumerate(report["overlap_matrix"]):
        t.add_row(ids[i], *[f"{v:.2f}" for v in row])
    console.print(t)

    if report["blind_spots"]:
        b = Table("blind spot", "area m2", "centre", "can change clothes here")
        for spot in report["blind_spots"][:5]:
            b.add_row(spot["id"], f"{spot['area_m2']:.0f}", str(spot["centre"]),
                      "yes" if spot["large_enough_to_change_clothes"] else "no")
        console.print(b)

    if out:
        out.write_text(json.dumps(report, indent=2))
        console.print(f"written {out}")






@app.command()
def evasion(
    pairs: int = typer.Option(20, help="matched hostile/benign pairs to generate"),
    techniques: int = typer.Option(2, help="evasion techniques per persona"),
    seed: int = typer.Option(0),
    no_protected: bool = typer.Option(False, help="exclude protected characteristics"),
    out: Path | None = typer.Option(None, help="write personas as JSON"),
) -> None:
    """Generate evasion personas and audit that appearance does not predict intent."""
    import json

    from .director.evasion import ALL, BY_CATEGORY, audit_confounds, generate_matched_population

    people = generate_matched_population(
        n_pairs=pairs, seed=seed, n_techniques=techniques, allow_protected=not no_protected
    )
    audit = audit_confounds(people)

    cat = Table("category", "techniques")
    for name, items in BY_CATEGORY.items():
        cat.add_row(name, ", ".join(t.id for t in items))
    console.print(cat)
    console.print(f"[bold]{len(ALL)} techniques[/] across {len(BY_CATEGORY)} categories\n")

    sample = Table("uid", "intent", "techniques", "face", "gait", "body", "innocent reason")
    for p in people[:12]:
        r = p.residual
        sample.add_row(p.uid, p.intent, ",".join(p.techniques),
                       f"{r['face']:.2f}", f"{r['gait']:.2f}", f"{r['body_shape']:.2f}",
                       p.rationale[:44])
    console.print(sample)

    verdict = "[green]balanced[/]" if audit["balanced"] else "[red]LEAKS INTENT[/]"
    console.print(
        f"appearance/intent confound: {verdict}  worst ratio={audit['worst_imbalance']}  "
        f"protected clean={audit['protected_clean']}"
    )
    console.print(
        "[dim]Every hostile persona has a benign twin with identical appearance, so a "
        "model cannot shortcut from clothing to intent.[/]"
    )
    if out:
        out.write_text(json.dumps(
            {"personas": [p.as_dict() for p in people], "audit": audit}, indent=2))
        console.print(f"written {out}")


@app.command()
def generate(
    count: int = typer.Option(20, help="how many diverse scenes"),
    seed: int = typer.Option(0),
    duration_s: float = typer.Option(240.0),
    engine: str = typer.Option("mock"),
    out: Path = typer.Option(Path("out/scenes")),
) -> None:
    """Generate N diverse, validated scenes (stratified, not random)."""
    from .generator import generate_batch, write_batch

    result = generate_batch(n=count, seed=seed, duration_s=duration_s, engine=engine)
    write_batch(result, out)

    t = Table("scene", "site", "cams", "entr", "people", "coverage", "narratives", "valid")
    for sc in result.scenes:
        s_ = sc.summary()
        t.add_row(s_["name"][:26], s_["site"].split("_")[0], str(s_["cameras"]),
                  str(s_["entrances"]), str(s_["people"]),
                  f"{(s_['coverage'] or 0):.2f}",
                  ",".join(sorted({a.narrative for a in sc.arcs if a.intent == "hostile"}))[:30],
                  "[green]yes[/]" if s_["valid"] else "[red]no[/]")
    console.print(t)

    d = result.diversity()
    console.print(
        f"[bold]{d['n_scenes']} scenes[/] | site types {d['distinct_site_types']} | "
        f"cameras {d['camera_count_range']} | people {d['people_range']} | "
        f"narratives {d['distinct_narratives']} | all valid: "
        + ("[green]yes[/]" if d["all_valid"] else "[red]no[/]")
    )
    console.print(f"attempts={result.attempts} repairs={result.repaired} -> {out}")




@app.command()
def events(run_dir: Path) -> None:
    """Summarise the events recorded in a run."""
    import json
    from collections import Counter

    path = run_dir / "gt" / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    counts = Counter(r["label"] for r in rows)
    t = Table("label", "count", "total seconds")
    for label, n in counts.most_common():
        secs = sum((r["t_end"] or 0) - r["t_start"] for r in rows if r["label"] == label)
        t.add_row(label, str(n), f"{secs:.1f}")
    console.print(t)




if __name__ == "__main__":
    app()


@app.command("export")
def export_unreal(
    run_dir: Path,
    out: Path | None = typer.Option(None, help="manifest path (default <run>/unreal/manifest.json)"),
) -> None:
    """Export a completed run as a manifest Unreal Editor can rebuild and render.

    This is the seam between the simulator and the renderer. The Director has
    already decided everything — who was where, facing which way, wearing what —
    so the manifest is a statement of fact rather than a request. Rendering it
    later, or at higher quality, changes pixels and not one value of ground
    truth.
    """

    import pandas as pd

    from .export.unreal_scene import build_manifest, write_manifest

    world_path = run_dir / "gt" / "world.parquet"
    scenario_path = run_dir / "meta" / "scenario.resolved.json"
    if not world_path.is_file() or not scenario_path.is_file():
        console.print(f"[red]{run_dir} does not look like a run "
                      "(needs gt/world.parquet and meta/scenario.resolved.json)")
        raise typer.Exit(1)

    scenario = load_scenario(scenario_path)
    rows = pd.read_parquet(world_path).to_dict("records")

    identities = []
    identities_path = run_dir / "gt" / "identities.parquet"
    if identities_path.is_file():
        identities = pd.read_parquet(identities_path).to_dict("records")

    manifest = build_manifest(scenario, rows, identities)
    destination = out or (run_dir / "unreal" / "manifest.json")
    write_manifest(destination, manifest)

    table = Table(title=f"exported {run_dir.name}")
    table.add_column("field")
    table.add_column("value", justify="right")
    table.add_row("cameras", str(len(manifest["cameras"])))
    table.add_row("people", str(len(manifest["people"])))
    table.add_row("tracks", str(len(manifest["tracks"])))
    table.add_row("keyframes", str(sum(len(t["keys"]) for t in manifest["tracks"])))
    table.add_row("duration_s", str(manifest["duration_s"]))
    table.add_row("manifest", str(destination))
    console.print(table)


@app.command()
def dataset(
    scenarios: list[Path] = typer.Argument(None, help="scenario YAMLs (omit with --mot)"),
    seeds: int = typer.Option(3, help="seeds per scenario; split by seed (last=test, 2nd-last=val)"),
    out: Path = typer.Option(Path("out/dataset")),
    frames_dir: Path | None = typer.Option(None, help="rendered frames (<dir>/<run>/<cam>/NNNN.png)"),
    profile: str | None = typer.Option(None, help="degrade profile applied to rendered frames"),
    max_frames: int = typer.Option(300, help="frames per camera (0 = all)"),
    mot: Path | None = typer.Option(None, help="wrap a MOTChallenge root instead (e.g. ~/data/mot/extracted/MOT20)"),
) -> None:
    """Package seeded runs (or MOTChallenge sequences) into a baseline dataset."""
    from .datasets import build_dataset, from_motchallenge

    if mot:
        m = from_motchallenge(mot, out, max_frames=max_frames or None)
    else:
        if not scenarios:
            raise typer.BadParameter("give scenario YAMLs or --mot")
        m = build_dataset(scenarios, seeds, out, frames_dir=frames_dir, profile=profile,
                          max_frames=max_frames or None)
    n = sum(len(c["frames"]) for r in m["runs"] for c in r["cameras"].values())
    console.print(f"[green]dataset[/] {out}: {len(m['runs'])} runs, {n} frames, "
                  f"synthetic_topdown={m['synthetic_topdown']}")






@app.command()
def video(
    render_dir: Path,
    profile: str = typer.Option("cctv_day_720p"),
    out: Path = typer.Option(Path("out/video")),
    fps: float = typer.Option(12.5),
    crf: int = typer.Option(30, help="final H.264 CRF; 28-32 looks like a DVR"),
    seed: int = typer.Option(0),
    real: Path = typer.Option(Path("~/data/real_cctv"), help="real CCTV frames for the report"),
    report: bool = typer.Option(True, help="write realism.md and contact_sheet.png"),
) -> None:
    """Degrade rendered frames (<dir>/<cam>/NNNN.png) into one DVR-style MP4 per camera."""
    from .video import make_videos

    res = make_videos(render_dir, out, profile, fps, crf, seed, real, report)
    for cam, mp4 in res["videos"].items():
        console.print(f"  {cam:10s} {res['frames'][cam]:4d} frames  {mp4}")
    for k in ("contact_sheet", "realism"):
        if k in res:
            console.print(f"  {k:22s} {res[k]}")
    if "table" in res:
        console.print(res["table"], markup=False, highlight=False)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def realism(ctx: typer.Context) -> None:
    """Frame-statistics comparison vs real CCTV (args as `python -m survsim.realism`)."""
    from .realism.compare import main

    raise typer.Exit(main(list(ctx.args)))

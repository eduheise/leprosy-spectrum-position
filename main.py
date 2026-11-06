"""Run the pipeline: preprocess, experiments, reports.

    python main.py preprocess              # raw spreadsheets -> data/processed
    python main.py experiments             # data/processed -> results
    python main.py reports                 # results -> reports
    python main.py all                     # the three stages in order
    python main.py experiments --only confidence   # one step of a stage

Each step is a module with ``INPUTS``, ``OUTPUTS`` and ``run()``. A step
refuses to start when one of its inputs is missing, and names the step that
writes it, so stages can be rerun independently from their files on disk.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path
from types import ModuleType

from leprosy_similarity import paths

# Steps of each stage, in the order they run.
STAGES: dict[str, tuple[str, ...]] = {
    "preprocess": (
        "preprocessing.load_raw_cohort",
        "preprocessing.apply_exclusion_criteria",
        "preprocessing.build_reference_cases",
        "preprocessing.make_folds",
        "preprocessing.screen_variables",
    ),
    "experiments": (
        "experiments.configuration_search",
        "experiments.select_configuration",
        "experiments.position_rescaling",
        "experiments.spectrum_position",
        "experiments.attribution",
        "experiments.confidence",
        "experiments.final_model",
    ),
    "reports": (
        "leprosy_similarity.reports.exclusion_attrition",
        "leprosy_similarity.reports.cohort_description",
        "leprosy_similarity.reports.feature_screen",
        "leprosy_similarity.reports.search_front",
        "leprosy_similarity.reports.search_diagnostics",
        "leprosy_similarity.reports.position_rescaling",
        "leprosy_similarity.reports.spectrum_position",
        "leprosy_similarity.reports.attribution",
        "leprosy_similarity.reports.confidence",
    ),
}


def step_name(module_path: str) -> str:
    return module_path.rsplit(".", 1)[-1]


def load_steps(stage: str) -> list[ModuleType]:
    return [importlib.import_module(module) for module in STAGES[stage]]


def producers() -> dict[Path, str]:
    """Which step writes each file, for the missing-input message."""
    owner: dict[Path, str] = {}
    for stage in STAGES:
        for module in load_steps(stage):
            for output in getattr(module, "OUTPUTS", ()):
                owner[Path(output)] = f"python main.py {stage} --only {step_name(module.__name__)}"
    return owner


def check_inputs(module: ModuleType) -> None:
    missing = [Path(p) for p in getattr(module, "INPUTS", ()) if not Path(p).exists()]
    if not missing:
        return
    owner = producers()
    lines = [f"step '{step_name(module.__name__)}' cannot run; missing input(s):"]
    for path in missing:
        hint = owner.get(path, f"see {paths.RAW / 'README.md'}")
        lines.append(f"  {path.relative_to(paths.ROOT)}  <- {hint}")
    raise SystemExit("\n".join(lines))


def run_stage(stage: str, only: str | None) -> None:
    steps = load_steps(stage)
    if only is not None:
        steps = [m for m in steps if step_name(m.__name__) == only]
        if not steps:
            known = ", ".join(step_name(m) for m in STAGES[stage]) or "none yet"
            raise SystemExit(f"no step '{only}' in stage '{stage}' (steps: {known})")
    for module in steps:
        check_inputs(module)
        print(f"== {stage}: {step_name(module.__name__)}", flush=True)
        start = time.time()
        module.run()
        print(f"== done in {time.time() - start:.1f} s\n", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=[*STAGES, "all"])
    parser.add_argument("--only", help="run one step of the stage, by name")
    args = parser.parse_args(argv)
    if args.stage == "all":
        if args.only:
            raise SystemExit("--only needs a single stage, not 'all'")
        for stage in STAGES:
            run_stage(stage, None)
    else:
        run_stage(args.stage, args.only)


if __name__ == "__main__":
    main(sys.argv[1:])

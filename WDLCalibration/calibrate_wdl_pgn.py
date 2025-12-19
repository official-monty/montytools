#!/usr/bin/env python3
"""
Calibrate post-MCTS WDL (Win/Draw/Loss) probabilities found in PGN comments.

PGN comment format expected (examples):
  {+18.43/5378 1.816s}
  {-1.88/7472 0.143s}
  {+M7/170 0.049s}   <-- mate; ignored by default

Encoding note (from user):
  cp = round((win - loss) * 10000)
  output_depth = round(draw * 10000)

In the PGN, cp is printed as cp/100 (so "+18.43" corresponds to cp=1843).
Therefore:
  v = (win - loss) = cp_printed / 100
  d = draw = output_depth / 10000

We assume the evaluation is from the *player to move before the move* (UCI-style),
and the comment appears after that move in the PGN. So "mover" alternates from the
starting side in the FEN (or White if no FEN).
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


# --- Regex helpers ---
MOVE_WITH_COMMENT_RE = re.compile(r'(?P<move>\S+)\s*\{(?P<comment>[^}]*)\}')
SCORE_RE = re.compile(r'(?P<cp>[+-]?\d+(?:\.\d+)?)/(?P<d>\d+)')
MATE_RE = re.compile(r'([+-]?M\d+|#\d+|mate)', re.IGNORECASE)


# --- Parsing PGN into (tags, move_text) games ---
def iter_pgn_games(path: str) -> Iterator[Tuple[int, Dict[str, str], str]]:
    """Yield (game_idx, tags, moves_text) from a PGN file."""
    game_idx = 0
    tags: Dict[str, str] = {}
    moves_lines = []
    in_tags = False
    in_moves = False

    def flush():
        nonlocal game_idx, tags, moves_lines, in_tags, in_moves
        if tags:
            moves_text = " ".join([ln.strip() for ln in moves_lines if ln.strip()])
            yield game_idx, tags, moves_text
            game_idx += 1
        tags = {}
        moves_lines = []
        in_tags = False
        in_moves = False

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("[Event "):
                # new game starts; flush previous
                yield from flush()
                in_tags = True
                in_moves = False

            if line.startswith("[") and in_tags:
                m = re.match(r'\[(\w+)\s+"(.*)"\]\s*', line.strip())
                if m:
                    tags[m.group(1)] = m.group(2)
                continue

            if line.strip() == "":
                if in_tags:
                    # end of tags; next non-empty line begins moves
                    in_tags = False
                    in_moves = True
                elif in_moves and moves_lines:
                    # blank line after moves -> end game
                    yield from flush()
                continue

            if in_moves:
                moves_lines.append(line)

        # EOF
        yield from flush()


def start_side_to_move(tags: Dict[str, str]) -> str:
    fen = tags.get("FEN")
    if fen:
        parts = fen.split()
        if len(parts) >= 2 and parts[1] in ("w", "b"):
            return parts[1]
    return "w"


def flip_side(side: str) -> str:
    return "b" if side == "w" else "w"


# --- WDL conversion ---
def wdl_from_cp_draw(cp_printed: float, draw_int: int) -> np.ndarray:
    """
    Convert (cp_printed, draw_int) -> raw probs [W, D, L] in mover's perspective.

    cp_printed is the "+18.43" value shown in PGN.
    draw_int is the "/5378" value shown in PGN (draw * 10000).
    """
    v = cp_printed / 100.0          # v = W - L
    d = draw_int / 10000.0          # D
    d = float(np.clip(d, 0.0, 1.0))

    # Feasible range: v in [-(1-d), +(1-d)]
    max_abs_v = max(0.0, 1.0 - d)
    v = float(np.clip(v, -max_abs_v, max_abs_v))

    w = 0.5 * ((1.0 - d) + v)
    l = 0.5 * ((1.0 - d) - v)

    p = np.array([w, d, l], dtype=np.float64)
    # numerical guard
    p = np.clip(p, 1e-15, 1.0)
    p = p / p.sum()
    return p


def label_for_mover(result_tag: str, mover: str) -> Optional[int]:
    """
    Return label index {0:W,1:D,2:L} from mover's perspective.
    """
    if result_tag == "1/2-1/2":
        return 1
    if result_tag == "1-0":
        return 0 if mover == "w" else 2
    if result_tag == "0-1":
        return 0 if mover == "b" else 2
    return None


@dataclass
class EvalRow:
    game_idx: int
    ply: int
    mover: str              # 'w' or 'b' (player to move *before* the move)
    result: str             # PGN Result tag
    move: str               # SAN token as written
    cp_printed: float
    draw_int: int
    raw_w: float
    raw_d: float
    raw_l: float


def iter_eval_rows(game_idx: int, tags: Dict[str, str], moves_text: str) -> Iterator[EvalRow]:
    result = tags.get("Result", "*")
    mover = start_side_to_move(tags)
    ply = 0

    for m in MOVE_WITH_COMMENT_RE.finditer(moves_text):
        ply += 1
        move = m.group("move")
        comment = m.group("comment")

        # Always advance mover per move-token/comment pair.
        current_mover = mover
        mover = flip_side(mover)

        if MATE_RE.search(comment):
            continue

        sm = SCORE_RE.search(comment)
        if not sm:
            continue

        cp_printed = float(sm.group("cp"))
        draw_int = int(sm.group("d"))
        p = wdl_from_cp_draw(cp_printed, draw_int)

        yield EvalRow(
            game_idx=game_idx,
            ply=ply,
            mover=current_mover,
            result=result,
            move=move,
            cp_printed=cp_printed,
            draw_int=draw_int,
            raw_w=float(p[0]),
            raw_d=float(p[1]),
            raw_l=float(p[2]),
        )


# --- Calibration ---
def fit_full_matrix_calibrator(
    X_logp: np.ndarray,
    y: np.ndarray,
    game_ids: np.ndarray,
    seed: int = 0,
    val_fraction: float = 0.2,
    C_grid: Tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0),
) -> Tuple[LogisticRegression, float, float, float]:
    """
    Fit q = softmax(W log(p) + b) using multinomial logistic regression.

    Returns:
      model_refit_on_all, best_C, baseline_val_logloss, calibrated_val_logloss
    """
    rng = np.random.default_rng(seed)
    unique_games = np.unique(game_ids)
    rng.shuffle(unique_games)

    split = int((1.0 - val_fraction) * len(unique_games))
    train_games = set(unique_games[:split])

    train_mask = np.array([gid in train_games for gid in game_ids], dtype=bool)
    X_train, y_train = X_logp[train_mask], y[train_mask]
    X_val, y_val = X_logp[~train_mask], y[~train_mask]

    # Per-position weight so each game contributes equally.
    counts = np.bincount(game_ids)
    sw = 1.0 / counts[game_ids]
    sw_train, sw_val = sw[train_mask], sw[~train_mask]

    # Baseline: uncalibrated probs correspond to exp(logp) but we pass original p to log_loss.
    P_val = np.exp(X_val)
    P_val = P_val / P_val.sum(axis=1, keepdims=True)
    baseline_ll = log_loss(y_val, P_val, labels=[0, 1, 2], sample_weight=sw_val)

    best_C = None
    best_ll = None

    for C in C_grid:
        model = LogisticRegression(
            solver="lbfgs",
            C=C,
            max_iter=500,
        )
        model.fit(X_train, y_train, sample_weight=sw_train)
        Q_val = model.predict_proba(X_val)
        ll = log_loss(y_val, Q_val, labels=[0, 1, 2], sample_weight=sw_val)
        if best_ll is None or ll < best_ll:
            best_ll = ll
            best_C = C

    assert best_C is not None and best_ll is not None

    # Refit on all data with best C.
    model_all = LogisticRegression(
        solver="lbfgs",
        C=best_C,
        max_iter=500,
    )
    model_all.fit(X_logp, y, sample_weight=sw)

    return model_all, float(best_C), float(baseline_ll), float(best_ll)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn", help="Input PGN with comments like {+18.43/5378 1.816s}")
    ap.add_argument("--out_csv", default="", help="Output CSV path. If empty, writes to stdout.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_fraction", type=float, default=0.2)
    ap.add_argument("--max_games", type=int, default=0, help="If >0, only use first N games.")
    args = ap.parse_args()

    # Pass 1: collect training data
    X_list = []
    y_list = []
    gid_list = []

    total_games = 0
    total_rows = 0

    for game_idx, tags, moves_text in iter_pgn_games(args.pgn):
        total_games += 1
        if args.max_games and game_idx >= args.max_games:
            break

        result = tags.get("Result", "*")
        if result not in ("1-0", "0-1", "1/2-1/2"):
            continue

        for row in iter_eval_rows(game_idx, tags, moves_text):
            y = label_for_mover(row.result, row.mover)
            if y is None:
                continue
            p = np.array([row.raw_w, row.raw_d, row.raw_l], dtype=np.float64)
            X_list.append(np.log(np.clip(p, 1e-12, 1.0)))
            y_list.append(y)
            gid_list.append(row.game_idx)
            total_rows += 1

    if not X_list:
        raise SystemExit("No usable eval rows found (check PGN format).")

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    game_ids = np.asarray(gid_list, dtype=np.int64)

    model, best_C, baseline_ll, calibrated_ll = fit_full_matrix_calibrator(
        X_logp=X,
        y=y,
        game_ids=game_ids,
        seed=args.seed,
        val_fraction=args.val_fraction,
    )

    W = model.coef_
    b = model.intercept_

    # Print a compact summary to stderr so CSV stays clean
    import sys
    print(f"# games_seen={total_games} rows_used={total_rows}", file=sys.stderr)
    print(f"# best_C={best_C}", file=sys.stderr)
    print(f"# val_logloss_raw={baseline_ll:.6f} val_logloss_cal={calibrated_ll:.6f}", file=sys.stderr)
    print(f"# W (3x3):\n{W}", file=sys.stderr)
    print(f"# b (3,): {b}", file=sys.stderr)

    # Pass 2: output calibrated mapping
    out_f = open(args.out_csv, "w", newline="", encoding="utf-8") if args.out_csv else sys.stdout
    writer = csv.writer(out_f)
    writer.writerow([
        "game_idx","ply","mover","result","move",
        "cp_printed","draw_int",
        "raw_w","raw_d","raw_l",
        "cal_w","cal_d","cal_l",
        "cal_cp_printed","cal_draw_int",
    ])

    def cal_from_raw(raw_w: float, raw_d: float, raw_l: float) -> Tuple[float, float, float]:
        p = np.array([raw_w, raw_d, raw_l], dtype=np.float64)
        x = np.log(np.clip(p, 1e-12, 1.0)).reshape(1, -1)
        q = model.predict_proba(x)[0]
        return float(q[0]), float(q[1]), float(q[2])

    emitted = 0
    for game_idx, tags, moves_text in iter_pgn_games(args.pgn):
        if args.max_games and game_idx >= args.max_games:
            break

        result = tags.get("Result", "*")
        if result not in ("1-0", "0-1", "1/2-1/2"):
            continue

        for row in iter_eval_rows(game_idx, tags, moves_text):
            cal_w, cal_d, cal_l = cal_from_raw(row.raw_w, row.raw_d, row.raw_l)

            # Convert calibrated probs back into the same compact (cp_printed, draw_int) encoding
            cal_v = cal_w - cal_l
            cal_cp_printed = cal_v * 100.0
            cal_draw_int = int(round(cal_d * 10000.0))

            writer.writerow([
                row.game_idx, row.ply, row.mover, row.result, row.move,
                f"{row.cp_printed:.2f}", row.draw_int,
                f"{row.raw_w:.6f}", f"{row.raw_d:.6f}", f"{row.raw_l:.6f}",
                f"{cal_w:.6f}", f"{cal_d:.6f}", f"{cal_l:.6f}",
                f"{cal_cp_printed:.2f}", cal_draw_int,
            ])
            emitted += 1

    if args.out_csv:
        out_f.close()

    print(f"# wrote {emitted} rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

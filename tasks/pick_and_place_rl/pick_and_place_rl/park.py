"""Park the base so the object and pad are inside the region the policy was trained on.

Yaw comes from the table's near edge (depth), the shift from the perceived object and pad. The base only moves
while the arm is tucked. Each iteration: survey (base still) -> plan -> move -> survey again, at most `max_iter`.

    python3 -m pick_and_place_rl.park            # real robot: surveys, prints the plan, asks, drives
    python3 -m pick_and_place_rl.park --dry-run  # surveys and prints only
"""

import argparse
import os
import sys
from dataclasses import dataclass, field

import numpy as np

from pick_and_place_rl.env import OBJ_REGION, PAD_REGION

YAW_TOL = np.radians(1.5)  # square to the table within this
MARGIN = 0.02  # objects should be at least this far inside the region edges, else the base moves
MOVE_TOL = 0.01
MIN_EDGE = 0.40  # never end closer than this to the table edge (the base is ~0.28 m in front of its centre)
MAX_SHIFT = 0.25  # one plan never moves further than this
MAX_YAW = np.radians(20.0)


@dataclass
class ParkPlan:
    dyaw: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    done: bool = False
    ok: bool = True
    msg: str = ""

    def describe(self) -> str:
        parts = []
        if abs(self.dyaw) > 1e-4:
            parts.append(f"turn {np.degrees(self.dyaw):+.1f} deg ({'left' if self.dyaw > 0 else 'right'})")
        if np.hypot(self.dx, self.dy) > 1e-4:
            parts.append(f"shift {self.dx * 100:+.1f} cm forward, {self.dy * 100:+.1f} cm left")
        return ", ".join(parts) or "no move"


def _interval(v_obj, v_pad, r_obj, r_pad, margin=0.0):
    lo = max(v_obj - (r_obj[1] - margin), v_pad - (r_pad[1] - margin))
    hi = min(v_obj - (r_obj[0] + margin), v_pad - (r_pad[0] + margin))
    return lo, hi


def plan_park(table, obj, pad) -> ParkPlan:
    """table: TableFit or None (no usable edge: yaw and edge distance are then not checked); obj, pad in the base frame."""
    yaw_err = 0.0 if table is None else table.yaw_err
    if abs(yaw_err) > MAX_YAW:
        return ParkPlan(ok=False, msg=f"robot is {np.degrees(yaw_err):+.0f} deg off the table; turn it closer to square by hand first")
    if abs(yaw_err) > YAW_TOL:
        return ParkPlan(dyaw=yaw_err, msg="square up to the table edge")
    (ox0, ox1), (oy0, oy1) = OBJ_REGION
    (px0, px1), (py0, py1) = PAD_REGION
    xl, xh = _interval(obj[0], pad[0], (ox0, ox1), (px0, px1))
    yl, yh = _interval(obj[1], pad[1], (oy0, oy1), (py0, py1))
    if xl > xh:
        return ParkPlan(ok=False, msg=f"the object and pad are {(xl - xh) * 100:.0f} cm too far apart front-to-back; place them at similar distances from the robot")
    if yl > yh:
        gap = obj[1] - pad[1]
        want = (oy0 - py1, oy1 - py0)
        return ParkPlan(ok=False, msg=f"object-to-pad spacing is {gap * 100:.0f} cm; the trained layout needs {want[0] * 100:.0f}-{want[1] * 100:.0f} cm")
    xs, xt = _interval(obj[0], pad[0], (ox0, ox1), (px0, px1), MARGIN)
    ys, yt = _interval(obj[1], pad[1], (oy0, oy1), (py0, py1), MARGIN)
    dx = 0.0 if xs <= 0.0 <= xt else 0.5 * (xl + xh)
    dy = 0.0 if ys <= 0.0 <= yt else 0.5 * (yl + yh)
    if np.hypot(dx, dy) < MOVE_TOL:
        dx = dy = 0.0
    if dx == 0.0 and dy == 0.0:
        return ParkPlan(done=True, msg="in position")
    if abs(dx) > MAX_SHIFT or abs(dy) > MAX_SHIFT:
        return ParkPlan(ok=False, msg=f"needs a {np.hypot(dx, dy) * 100:.0f} cm shift (limit {MAX_SHIFT * 100:.0f}); reposition the robot or the objects by hand")
    if table is not None and table.edge_x - dx < MIN_EDGE:
        return ParkPlan(ok=False, msg=f"moving {dx * 100:.0f} cm forward would leave the base only {(table.edge_x - dx) * 100:.0f} cm from the table edge")
    return ParkPlan(dx=dx, dy=dy, msg="shift into the trained region")


@dataclass
class ParkResult:
    ok: bool
    msg: str
    iterations: int = 0
    history: list = field(default_factory=list)


def survey(sc, settle=2.5, timeout=25.0, height_tol=0.04, need_table=True):
    """Base still: fresh table fit, object and pad (base frame). Returns (TableFit, obj, pad) or raises.

    Waits until torso and head have stopped, and ignores table fits whose height is not within `height_tol` of
    the expected table height (sc.z_offset + 0.815): that is a view taken while something was still moving."""
    t0 = sc.now()
    while not sc.joints_still(["torso_lift_joint", "head_1_joint", "head_2_joint"]):
        if sc.now() - t0 > timeout:
            raise RuntimeError("survey failed: torso/head never stopped moving")
    sc.sleep(settle)
    sc.unlatch_object()
    sc.unlatch_pad()
    sc.clear_table()
    expected = 0.815 + sc.z_offset
    t0 = sc.now()
    last = "no table fit yet"
    while sc.now() - t0 < timeout:
        fit = sc.table_fit(max_age=1.0)
        if fit is not None and abs(fit.height - expected) > height_tol:
            last = f"table measured at {fit.height:.3f} m but expected {expected:.3f} m (check --table-height)"
            sc.clear_table()
            fit = None
            if need_table:
                sc.sleep(0.2)
                continue
        if (fit is not None or not need_table) and sc.wait_perception(timeout=0.2, max_age=1.5):
            sc.sleep(1.0)  # a second of frames so the node's median window is clean
            fit = sc.table_fit(max_age=1.0) or fit
            if fit is not None and abs(fit.height - expected) > height_tol:
                fit = None
            return fit, sc.model_pos("pick_object") + np.array([0, 0, sc.z_offset]), sc.model_pos("place_pad") + np.array([0, 0, sc.z_offset])
        sc.sleep(0.2)
    raise RuntimeError(f"survey failed: the camera does not see the table edge, the object and the pad together ({last})")


def auto_park(sc, base, confirm=None, max_iter=5, log=print, dry_run=False, need_table=True) -> ParkResult:
    """Iterate survey -> plan -> move. `confirm(text)` -> bool is asked once before the first move."""
    hist = []
    asked = False
    for it in range(max_iter + 1):
        try:
            table, obj, pad = survey(sc, need_table=need_table)
        except RuntimeError as e:
            return ParkResult(False, str(e), it, hist)
        plan = plan_park(table, obj, pad)
        tdesc = "no table edge" if table is None else (f"table h {table.height:.3f} m, edge {table.edge_x:.3f} m, yaw {np.degrees(table.yaw_err):+.2f} deg "
                                                       f"(rms {table.rms * 1000:.1f} mm)")
        log(f"[park {it}] {tdesc} | object {obj[:2].round(3)} pad {pad[:2].round(3)} -> {plan.describe()}: {plan.msg}")
        hist.append({"table_edge": None if table is None else table.edge_x,
                     "yaw_deg": None if table is None else float(np.degrees(table.yaw_err)), "obj": obj[:2].tolist(),
                     "pad": pad[:2].tolist(), "plan": plan.describe()})
        if plan.done:
            return ParkResult(True, "parked", it, hist)
        if not plan.ok:
            return ParkResult(False, plan.msg, it, hist)
        if dry_run:
            return ParkResult(True, "dry run: " + plan.describe(), it, hist)
        if it == max_iter:
            return ParkResult(False, f"still not in position after {max_iter} corrections ({plan.describe()})", it, hist)
        if not asked:
            if confirm is not None and not confirm(
                f"The base will drive ({plan.describe()}, then small corrections, within {base.travel_left * 100:.0f} cm). "
                "Arm tucked? Area around the base clear? Hand on the e-stop?"
            ):
                return ParkResult(False, "declined", it, hist)
            asked = True
        try:
            if abs(plan.dyaw) > 1e-4:
                base.turn(plan.dyaw)
            else:
                base.move_relative(plan.dx, plan.dy)
        except RuntimeError as e:
            base.stop()
            return ParkResult(False, f"base motion aborted: {e}", it, hist)
    return ParkResult(False, "unreachable", max_iter, hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=None)
    ap.add_argument("--dry-run", action="store_true", help="survey and print the plan, do not move")
    ap.add_argument("--yes", action="store_true", help="do not ask before driving")
    ap.add_argument("--table-height", type=float, default=0.815)
    args = ap.parse_args()

    import rclpy

    from pick_and_place_rl.base import BaseDriver
    from pick_and_place_rl.head import solve_head
    from pick_and_place_rl.preflight import fetch_urdf
    from pick_and_place_rl.robot import TiagoScene

    rclpy.init()
    if args.urdf:
        urdf = open(args.urdf).read()
    else:
        tmp = rclpy.create_node("park_urdf")
        urdf = fetch_urdf(tmp)
        tmp.destroy_node()
        if not urdf:
            sys.exit("no robot_description; pass --urdf")
    sc = TiagoScene(urdf, node_name="pick_place_park", sim=False)
    sc.z_offset = args.table_height - 0.815
    pan, tilt, _, _ = solve_head(urdf, 0.35, (0.63, 0.0, args.table_height + 0.03))
    sc.send_torso(0.35, 8.0)
    sc.send_head(pan, tilt, 4.0)
    sc.sleep(8.5)
    base = BaseDriver(sc)
    ans = (lambda t: True) if args.yes else (lambda t: input(f"{t} [y/N] ").strip().lower() in ("y", "yes"))
    res = auto_park(sc, base, confirm=ans, dry_run=args.dry_run)
    print(f"PARK: {'OK' if res.ok else 'FAILED'} - {res.msg}")
    os._exit(0 if res.ok else 1)


if __name__ == "__main__":
    main()

# Closed-loop mode: Gazebo results

Produced by `python3 -m pick_and_place_rl.closed_loop_test <scenarios>` on the real-robot code path (`sim=False`).
Finger contact (`learn`, `flow`) and the hard table (`touch_emul`) are **emulated**; see `../../REAL_ROBOT.md`.

| File | What |
| :--- | :--- |
| `finger_probe.json` | Finger opening vs time, closing on the can and on empty table, at close commands 0.0245 and 0.016: identical. Gazebo's fingers give no grasp signal. |
| `early_run_arm_does_not_stall.json` | `touch` scenario: the arm tracks the command straight through the table height (no stall), so contact cannot be seen physically in Gazebo. Its `track`/`stale` rows come from an earlier version of the code. |
| `touch_emul_deadband.json` | Touchdown with an emulated surface, including pad-height errors that leave the hand blocked 7-10 mm above its target (the former dead band). |
| `final_suite.json` | `track stale carry touch_emul learn` on the final code. |
| `flow.json` | Approach, grasp judgement, carry, place, release in one closed-loop episode, with 0 and 2 cm object-estimate bias. |

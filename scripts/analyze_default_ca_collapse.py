#!/usr/bin/env python
"""Explain why the Default method's accuracy in results/pdf/videomme.pdf is low.

That figure plots END-TO-END (CA-stage) accuracy. Every method's SA accuracy on
VideoMME-long 601-620 is similar (~53-58%), but the CA stage -- which re-answers
each question from resampled video frames of the top SA candidate nodes -- flips
many correct SA answers to wrong. Default suffers the worst CA collapse, so its
plotted accuracy is lowest.

This computes, per method, SA vs CA accuracy and the SA->CA transition counts
(how many SA-correct answers CA destroys vs rescues), and writes a small CSV.
Reads each method's records_{sa,ca}.json from outputs/.
"""
import collections
import csv
import json
import os

MAIN = "/export/home/yulin/wenyan/github/Project-Ava"
METHODS = [
    ("Default",      f"{MAIN}/outputs/videomme_long_601_620"),
    ("KF-Merge",     f"{MAIN}/outputs/videomme_keyframe_merge_601_605"),
    ("KF-Direct 7s", f"{MAIN}/outputs/videomme_keyframe_601_605"),
    ("KF-Direct 3s", f"{MAIN}/outputs/videomme_keyframe_gop3s_601_620"),
]
OUT_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "csv", "videomme_sa_ca_transitions.csv")


def main():
    rows = []
    for name, d in METHODS:
        sa = json.load(open(f"{d}/records_sa.json"))
        ca = json.load(open(f"{d}/records_ca.json"))
        ok = lambda r: int(r.get("answer") == r.get("gt"))
        keys = [k for k in sa if k in ca
                and sa[k].get("status") == "ok" and ca[k].get("status") == "ok"]
        sok = {k: ok(sa[k]) for k in keys}
        cok = {k: ok(ca[k]) for k in keys}
        t = collections.Counter((sok[k], cok[k]) for k in keys)
        n = len(keys)
        rec = {
            "method": name, "n": n,
            "sa_acc": round(100 * sum(sok.values()) / n, 1),
            "ca_acc": round(100 * sum(cok.values()) / n, 1),
            "both_correct": t[(1, 1)],
            "destroyed_sa_ok_ca_wrong": t[(1, 0)],
            "rescued_sa_wrong_ca_ok": t[(0, 1)],
            "both_wrong": t[(0, 0)],
            "net_ca_change": t[(0, 1)] - t[(1, 0)],
        }
        rows.append(rec)
        print(f"{name:14s} SA {rec['sa_acc']:.1f}%  CA {rec['ca_acc']:.1f}%  "
              f"(destroyed {rec['destroyed_sa_ok_ca_wrong']}, "
              f"rescued {rec['rescued_sa_wrong_ca_ok']}, net {rec['net_ca_change']:+d})")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()

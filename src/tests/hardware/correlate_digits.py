#!/usr/bin/env python3
"""
run_optical_jtc.py
──────────────────
Command-line utility that performs one binary- or analogue
Joint-Transform-Correlation step between two MNIST digits on **real
hardware** (Thorlabs Exulus HD SLM + IDS/Thorlabs camera).

Everything is configured through a single Python dictionary `cfg`.
No plotting/GUI code lives inside the correlator – all visualisation
is done here.

Requires:

* `ExulusSLM`  (see exulus_driver.py)
* `UC480Controller`  (thin wrapper around uEye SDK)
* `OpticalJTCorrelator`  (see optical_jtc_correlator_exulus.py)
* `get_test_data()`  for MNIST/F-MNIST digits
"""

# ───────────────────────── imports ──────────────────────────────
from pathlib import Path
from datetime import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt

from ...data.data import get_test_data
from ...hardware.devices.SLM import ExulusSLM
from ...hardware.devices.Camera import UC480Controller
from ...distance.OpticalJTCorrelator import OpticalJTCorrelator

# ─────────────────────── configuration ─────────────────────────
cfg: dict = {
    # I/O
    "dataset":       "mnist",      # "mnist" | "fashion"
    "ref_digit":     1,
    "obj_digit":     2,

    # SLM
    "stroke_mode":   "half",       # "half" | "full"
    "m":             0.0174,       # calibration slope  (rad / grey)
    "b":             0.3639,       # calibration offset (rad)
    "checkerboard":  True,         # apply ±1 checkerboard

    # camera
    "cam_exposure":  0.005,          # [ms] – adjust for your laser power
    "cam_serial":    None,         # set if you have >1 IDS camera

    # JTC algorithm
    "binary_input":  True,
    "binary_jps":    True,
    "display_scale": 0.05,
    "sleep":         0.10,         # s between uploads
    "blocking_frac": 0.005,
}

# ─────────────────────── helper: load digits ───────────────────
def load_two_digits(dataset: str, d_ref: int, d_obj: int):
    X, y = get_test_data(dataset_name=dataset)
    ref = X[np.where(y == d_ref)[0][0]].reshape(28, 28) / 255.0
    obj = X[np.where(y == d_obj)[0][0]].reshape(28, 28) / 255.0
    return ref, obj

# ─────────────────────── main routine ──────────────────────────
def main(cfg):

    # — open hardware ——————————————————————————
    slm = ExulusSLM(device_index=0,
                    m=cfg["m"], b=cfg["b"],
                    stroke_mode=cfg["stroke_mode"])
    cam = UC480Controller(serial=cfg["cam_serial"])
    cam.setExposure(cfg["cam_exposure"])        # ms

    jtc = OpticalJTCorrelator(
        slm=slm,
        cam=cam,
        binary_input=cfg["binary_input"],
        binary_jps=cfg["binary_jps"],
        checkerboard=cfg["checkerboard"],
        display_scale=cfg["display_scale"],
        sleep_time=cfg["sleep"],
        blocking_fraction=cfg["blocking_frac"],
    )

    # — prepare digits ————————————————————————
    ref, obj = load_two_digits(cfg["dataset"],
                               cfg["ref_digit"],
                               cfg["obj_digit"])

    # — run correlation ————————————————————————
    pk, dc, shift, a0_8bit, jps_8bit, corr, masked = jtc.correlate(
        ref, obj, return_planes=True
    )
    norm = pk / (dc + 1e-6)

    # — simple plots ———————————————————————————
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    ax[0, 0].imshow(ref, cmap="gray"); ax[0, 0].set_title(f"Reference {cfg['ref_digit']}")
    ax[0, 1].imshow(obj, cmap="gray"); ax[0, 1].set_title(f"Object {cfg['obj_digit']}")
    ax[0, 2].imshow(a0_8bit, cmap="gray"); ax[0, 2].set_title("SLM input (8-bit)")

    ax[1, 0].imshow(jps_8bit, cmap="gray"); ax[1, 0].set_title("JPS → SLM (8-bit)")
    im = ax[1, 1].imshow(corr, cmap="hot"); ax[1, 1].set_title("Correlation plane")
    fig.colorbar(im, ax=ax[1, 1])
    ax[1, 2].imshow(masked, cmap="hot"); ax[1, 2].set_title("Masked corr (peak)")
    for a in ax.ravel(): a.axis("off")
    plt.suptitle(f"peak={pk:.3e}  DC={dc:.3e}  norm={norm:.4f}  shift={shift}")
    plt.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"JTC_{cfg['ref_digit']}vs{cfg['obj_digit']}_{ts}.png"
    plt.savefig(out, dpi=150)
    print("saved →", out)
    plt.show()

    # — tidying up ————————————————————————————
    jtc.close()

# ───────────────────── CLI wrapper (optional) ──────────────────
if __name__ == "__main__":
    main(cfg)

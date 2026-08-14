"""Sustained T4 GPU heartbeat. Writes /content/t4_heartbeat.log until killed."""

import datetime
import subprocess
import time

import taichi as ti

LOG = "/content/t4_heartbeat.log"
t0 = time.time()

ti.init(arch=ti.cuda)
x = ti.field(dtype=ti.f32, shape=8_000_000)


@ti.kernel
def burn():
    for i in x:
        x[i] = x[i] * 1.0001 + 0.001


def gpu_line() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,temperature.gpu",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def log(msg: str) -> None:
    line = f"{datetime.datetime.now(datetime.timezone.utc).isoformat()} elapsed_s={time.time() - t0:.1f} {msg}\n"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
    print(line, end="", flush=True)


log("start " + gpu_line())
burn()
log("warmup_done " + gpu_line())

while True:
    inner_t0 = time.time()
    for _ in range(400):
        burn()
    inner = time.time() - inner_t0
    log(f"batch_s={inner:.3f} {gpu_line()}")

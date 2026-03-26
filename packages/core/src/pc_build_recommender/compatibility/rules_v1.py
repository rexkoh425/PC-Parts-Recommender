"""First-pass compatibility checks.

Hard-coded booleans with no rule identity or version, so a failure could not be
explained or audited. Replaced by the versioned compat_v2 engine.
"""


def socket_matches(cpu: dict, motherboard: dict) -> bool:
    return cpu.get("socket") == motherboard.get("socket")


def psu_is_big_enough(psu: dict, estimated_watts: float) -> bool:
    return float(psu.get("wattage", 0)) >= estimated_watts * 1.2


def gpu_fits(gpu: dict, case: dict) -> bool:
    return float(gpu.get("length_mm", 0)) <= float(case.get("max_gpu_length_mm", 0))

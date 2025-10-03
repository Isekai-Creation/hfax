import torch
import torch_xla
import torch_xla.core.xla_model as xm


def _tpuinfo_totals():
    try:
        from tpu_info import device as tpu_device  # type: ignore
        from tpu_info import metrics  # type: ignore
    except Exception as e:
        raise SystemExit(
            'Please install tpu-info for TPU memory metrics: '
            'https://github.com/AI-Hypercomputer/cloud-accelerator-diagnostics/tree/main/tpu_info'
        ) from e
    chip_type, count = tpu_device.get_local_chips()
    if not chip_type or not count:
        print('No TPU devices found.')
        return 0, 0
    used = 0
    total = 0
    for chip in metrics.get_chip_usage(chip_type):
        used += int(chip.memory_usage)
        total += int(chip.total_memory)
    # SPMD adjustment (v3-8)
    used //= 3
    total //= 3
    return used, total


devices = xm.get_xla_supported_devices()
print(f"Devices: {devices}")

used_b, total_b = _tpuinfo_totals()
print(f"TPU memory (per-replica approx): used={used_b/1e9:.2f} / total={total_b/1e9:.2f} GB")

# Allocate a tensor and show updated totals
dev = xm.xla_device()
t = torch.randn(2, 4, 144, 720, 1280, device=dev)
xm.mark_step()
used_b2, total_b2 = _tpuinfo_totals()
print(f"After alloc: used={used_b2/1e9:.2f} / total={total_b2/1e9:.2f} GB")

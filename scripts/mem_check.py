import torch
import torch_xla
import torch_xla.core.xla_model as xm


devices = xm.get_xla_supported_devices()
print(f"Devices: {devices}")
total = {0: 0, 1: 0}
for device in devices:
    mem = round(xm.get_memory_info(device)["bytes_limit"] / 1e9, 2)
    total[1] += mem
    print(f"Total TPU device: {device} memory: {mem} GB")

print(f"Total TPU memory: {total[0]} / {total[1]} GB")

for device in devices:
    mem = round(xm.get_memory_info(device)["bytes_limit"] / 1e9, 2)
    t = torch.randn(torch.randint(1, 8, (1,)), 4, 144, 720, 1280).to(device)
    mem_used = round(xm.get_memory_info(device)["bytes_used"] / 1e9, 2)
    total[0] += mem_used
    print(f"Total TPU device: {device} memory: {mem_used} / {mem} GB")
    xm.mark_step()

print(f"Total TPU memory: {total[0]} / {total[1]} GB")

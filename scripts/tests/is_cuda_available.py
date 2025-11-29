import os
import sys

import torch

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
try:
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())
    props = torch.cuda.get_device_properties(0)
    print("device name:", props.name)
    print("compute capability:", props.major, props.minor)
except Exception as e:
    print("cuda probe error:", repr(e))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

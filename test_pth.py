import torch
from tqdm import tqdm

def print_structure(obj, indent=0, max_items=20):
    prefix = " " * indent

    if isinstance(obj, dict):
        print(f"{prefix}dict with {len(obj)} keys")
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_items:
                print(f"{prefix}  ... ({len(obj) - max_items} more keys)")
                break
            print(f"{prefix}  [{k}] -> {type(v)}")
            print_structure(v, indent + 4, max_items)

    elif isinstance(obj, list):
        print(f"{prefix}list of length {len(obj)}")
        for i, item in enumerate(obj):
            if i >= max_items:
                print(f"{prefix}  ... ({len(obj) - max_items} more items)")
                break
            print(f"{prefix}  [{i}] -> {type(item)}")
            print_structure(item, indent + 4, max_items)

    elif isinstance(obj, tuple):
        print(f"{prefix}tuple of length {len(obj)}")
        for i, item in enumerate(obj):
            print(f"{prefix}  [{i}] -> {type(item)}")
            print_structure(item, indent + 4, max_items)

    else:
        # leaf node
        if hasattr(obj, "shape"):
            print(f"{prefix}Tensor shape: {obj.shape}, dtype: {obj.dtype}")
        else:
            print(f"{prefix}{obj}")


# ---- load file ----
path = "./activation_outs/layer_22/activations_l22_idx6590.pt"
data = torch.load(path, map_location="cpu")

print("Top-level type:", type(data))
print_structure(data)
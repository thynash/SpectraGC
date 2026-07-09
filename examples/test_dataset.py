from spectragc import *

set_seed()

device = get_device()

print(device)

data = load_cora(device)

print(data)

print(f"Nodes      : {data.num_nodes}")

print(f"Edges      : {data.num_edges}")

print(f"Features   : {data.num_features}")

print(f"Classes    : {data.y.max().item()+1}")

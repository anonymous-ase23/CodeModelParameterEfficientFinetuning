import glob
import json
import numpy as np
import os
files = list(glob.glob("checkpoints-cj/*"))
files = set([v.split(".seed")[0] for v in files])
results = []
times = []
methods = []
for file in files:
    temp1 = [[],[],[]]
    temp2 = []
    print(file)
    model = file.split("/")[1].split("_")[0]
    method = "prefix"
    if "sequential" in file:
        method = "adapter"
    if "am_lora" in file:
        method = "lora"
    if "am_prefix" in file and "fm_adapter" in file:
        method = "MHM"
    elif "parallel" in file:
        method = "PA"
    if "am_none" in file and "fm_none" in file:
        method = "Full-FT"
    for postfix in ['.seed_42','.seed_36','.seed_32']:
        if not os.path.exists(file + postfix + "/summary.log"):
            continue
        lines = open(file + postfix + "/summary.log").readlines()
        line = lines[-2]
        data = line.split(", ")
        data = [float(item.split(": ")[1]) for item in data]
        temp1[0].append(data[0])
        temp1[1].append(data[1])
        temp1[2].append(data[2])
        # temp2.append(js['train_samples_per_second'])
    results.append('%.2f & %.2f & %.2f'%(np.mean(temp1[0]), np.mean(temp1[1]), np.mean(temp1[2]) ))

    print(temp1)
    # times.append("%d"%(np.mean(temp2)))
    methods.append(model + "_" + method)
values = []
for m, r in zip(methods, results):
    values.append((m, r))
for m, r in sorted(values):
    print(m, r)

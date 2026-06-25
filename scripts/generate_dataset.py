# CHANGED: lowered range to 10-120 (was 10-300)
# CHANGED: thresholds adjusted for ~33% each class (was heavily HIGH-skewed)
# CHANGED: 10000 records with balanced thresholds

import pandas as pd
import random
import os

random.seed(42)

modules  = ['FI', 'MM', 'SD', 'HR']
stages   = ['Development', 'Quality', 'Production']
statuses = ['Approved', 'Pending', 'Rejected']

data = []

for i in range(10000):                         
    module    = random.choice(modules)
    objects   = random.randint(1, 6)
    lines     = random.randint(10, 120)        # CHANGED: was randint(10, 300)
    conflicts = random.randint(0, 2)
    failures  = random.randint(0, 3)           # CHANGED: was randint(0, 4)
    stage     = random.choice(stages)
    status    = random.choice(statuses)

    # CHANGED: New balanced thresholds
    # HIGH   ~33%: conflicts==2 OR failures==3
    # MEDIUM ~33%: lines>80 OR failures==2 OR (FI & lines>60)
    # LOW    ~33%: everything else
    if conflicts == 2 or failures == 3:
        risk = "HIGH"
    elif lines > 80 or failures == 2 or (module == "FI" and lines > 60):
        risk = "MEDIUM"
    else:
        risk = "LOW"

    # 8% real-world noise simulation
    if random.random() < 0.08:
        risk = random.choice(["LOW", "MEDIUM", "HIGH"])

    data.append([f"TR{i+1:05d}", module, objects, lines,
                 conflicts, failures, stage, status, risk])

df = pd.DataFrame(data, columns=[
    "transport_id", "module", "objects_changed", "lines_changed",
    "conflicts", "history_failures", "transport_stage",
    "change_request_status", "risk_level"
])

os.makedirs("data", exist_ok=True)
df.to_csv("data/sap_transport_dataset.csv", index=False)

print("✅ 10000 balanced SAP records created!")
print(df["risk_level"].value_counts())
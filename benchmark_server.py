import requests
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

url = "http://workspace.featurize.cn:64014/process"
image_path = "./images/img1.jpg"
objects = ["person", "cat"]
n_requests = 100

records = []

for i in range(n_requests):
    with open(image_path, "rb") as img:
        files = {"image": img}
        data = [("objects", obj) for obj in objects]

        start = time.time()
        response = requests.post(url, files=files, data=data)
        end = time.time()

        latency = end - start
        timestamp = datetime.now()

        status = "success" if response.status_code == 200 else "error"

        records.append({
            "timestamp": timestamp,
            "latency_s": latency,
            "status": status
        })

        print(f"[{i+1}/{n_requests}] {status.upper()} - {latency:.3f}s")

df = pd.DataFrame(records)
df.to_csv("llava_latency_log.csv", index=False)

successful = df[df["status"] == "success"]
mean_latency = successful["latency_s"].mean()
std_latency = successful["latency_s"].std()
max_latency = successful["latency_s"].max()
min_latency = successful["latency_s"].min()
sla_threshold = 1.5
sla_success_rate = (successful["latency_s"] < sla_threshold).mean() * 100

plt.figure(figsize=(12, 6))
sns.lineplot(x="timestamp", y="latency_s", data=successful, marker="o", label="Latency (s)")
plt.ylim(0, 5)
plt.title("LLaVA Latency Over Time", fontsize=16)
plt.xlabel("Request Time")
plt.ylabel("Latency (seconds)")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("llava_latency_plot.png", dpi=300)
plt.show()

print("\n--- Summary Statistics ---")
print(f"Mean latency: {mean_latency:.3f}s")
print(f"Standard deviation: {std_latency:.3f}s")
print(f"Min latency: {min_latency:.3f}s")
print(f"Max latency: {max_latency:.3f}s")
print(f"Requests under 1.5s: {sla_success_rate:.1f}%")

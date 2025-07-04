from collections import defaultdict
import os
import shutil
import random

input_dir = "C:/Users/Vignesh M/PycharmProjects/Thesis/dataset/train"
output_dir = "C:/Users/Vignesh M/PycharmProjects/Thesis/dataset/lfw_test_clean"

# Group files by identity
identity_dict = defaultdict(list)
for f in os.listdir(input_dir):
    if f.endswith(".jpg"):
        identity = "_".join(f.split("_")[:-1])
        identity_dict[identity].append(f)

print("Total identities found:", len(identity_dict))  # <- should be ~5749

# Pick 20% for test
test_ids = random.sample(list(identity_dict.keys()), int(0.2 * len(identity_dict)))
print("Selected test identities:", len(test_ids))  # <- should be ~1150

os.makedirs(output_dir, exist_ok=True)
for identity in test_ids:
    for fname in identity_dict[identity]:
        shutil.copy(os.path.join(input_dir, fname), os.path.join(output_dir, fname))

print("✅ Test set created in:", output_dir)


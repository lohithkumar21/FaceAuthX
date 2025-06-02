import os
import random
import shutil
from itertools import islice

# Configuration
outputFolderPath = "Dataset/SplitData"
inputFolderPath = "Dataset/All"
splitRatio = {"train": 0.7, "val": 0.2, "test": 0.1}
classes = ["fake", "real"]

# Prepare Output Folder
shutil.rmtree(outputFolderPath, ignore_errors=True)
os.makedirs(outputFolderPath, exist_ok=True)

for split in ["train", "val", "test"]:
    os.makedirs(f"{outputFolderPath}/{split}/images", exist_ok=True)
    os.makedirs(f"{outputFolderPath}/{split}/labels", exist_ok=True)

# Fetch and Shuffle Unique Names
uniqueNames = list({name.split('.')[0] for name in os.listdir(inputFolderPath)})
random.shuffle(uniqueNames)

# Calculate Split Sizes
lenData = len(uniqueNames)
lenTrain = int(lenData * splitRatio['train'])
lenVal = int(lenData * splitRatio['val'])
lenTest = lenData - (lenTrain + lenVal)

# Split the Data
splits = [uniqueNames[:lenTrain], uniqueNames[lenTrain:lenTrain + lenVal], uniqueNames[-lenTest:]]
for split, folder in zip(splits, ["train", "val", "test"]):
    for name in split:
        shutil.copy(f"{inputFolderPath}/{name}.jpg", f"{outputFolderPath}/{folder}/images/{name}.jpg")
        shutil.copy(f"{inputFolderPath}/{name}.txt", f"{outputFolderPath}/{folder}/labels/{name}.txt")

# Create data.yaml
dataYaml = f"path: ../Data\ntrain: ../train/images\nval: ../val/images\ntest: ../test/images\nnc: {len(classes)}\nnames: {classes}"
with open(f"{outputFolderPath}/data.yaml", 'w') as f:
    f.write(dataYaml)

print("Data Split and YAML Creation Completed.")
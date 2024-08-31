
import sys
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader

task = "<MORE_DETAILED_CAPTION>"
name="yayayaaa/florence-2-large-ft-moredetailed"
model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, device_map="auto")
processor = AutoProcessor.from_pretrained(name, trust_remote_code=True, device_map="auto")

#model.language_model.generate = torch.compile(model.language_model.generate)

# BATCH SIZE
batch_size=12

directory = Path(sys.argv[1])
patterns = ['**/*.jpg', '**/*.jpeg', '**/*.png']
patterns += [p.upper() for p in patterns]
filenames = [str(fn) for pattern in patterns for fn in directory.glob(pattern)]

class Data(Dataset):
    def __init__(self, data):
        self.name="Data"
        self.data=data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

data = DataLoader(Data(filenames), batch_size=batch_size, num_workers=0, shuffle=False)

@torch.inference_mode()
def process_images(images):
    inputs = processor(text=[task]*len(images), images=images, return_tensors="pt").to("cuda")

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
    )

    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_texts

for batch in tqdm(data):
    images = [Image.open(filename).convert("RGB") for filename in batch]

    generated_texts = process_images(images)

    for i, caption in enumerate(generated_texts):
        filename = Path(batch[i])
        #print(caption)
        with open(filename.with_suffix(".txt"), "w") as text_file:
            text_file.write(caption)

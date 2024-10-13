# Load data
import pandas as pd
import os
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel, PeftConfig
import torch

TEST_IMG_DIR = "/scratch/data/m23csa016/test_images/"
FINETUNING_DIR = "/scratch/data/m23csa016/finetuning"
CSV_DIR = "/iitjhome/m23csa016/meesho_code/1_missing_attrs"

MAX_PIXELS = 1280*28*28

# Load the base model (e.g., Qwen2-VL-7B-Instruct)
model_name = "Qwen/Qwen2-VL-7B-Instruct"
print("Loading model")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    cache_dir=FINETUNING_DIR
)


adapter_paths = {
    'cs_kurtis': {
        'model': os.path.join(FINETUNING_DIR, "cs_kurtis"),
        'csv_path': os.path.join(CSV_DIR, "cs_kurtis_1_missing.csv")
    },
    'cs_men_tshirts': {
        'model': os.path.join(FINETUNING_DIR, "cs_men_tshirts"),
        'csv_path': os.path.join(CSV_DIR, "cs_men_tshirts_1_missing.csv")
    },
    'cs_sarees': {
        'model': os.path.join(FINETUNING_DIR, "cs_sarees"),
        'csv_path': os.path.join(CSV_DIR, "cs_sarees_1_missing.csv")
    },
    'cs_women_tops': {
        'model': os.path.join(FINETUNING_DIR, "cs_women_tops"),
        'csv_path': os.path.join(CSV_DIR, "cs_women_tops_1_missing.csv")
    },
    'universal_attrs': {
        'model': os.path.join(FINETUNING_DIR, "universal_attrs"),
        'csv_path': os.path.join(CSV_DIR, "universal_attrs_1_missing.csv")
    },
    'cs_women_tshirts': {
        'model': os.path.join(FINETUNING_DIR, "cs_women_tshirts"),
        'csv_path': os.path.join(CSV_DIR, "cs_women_tshirts_1_missing.csv")
    },
    'women_group_attrs': {
        'model': os.path.join(FINETUNING_DIR, "women_group_attrs"),
        'csv_path': os.path.join(CSV_DIR, "women_group_attrs_1_missing.csv")
    },
}


for dataset, values in adapter_paths.items():
    adapter_path = values['model']
    csv_path = values['csv_path']

    # Load the PEFT config
    peft_config = PeftConfig.from_pretrained(adapter_path)

    # Load the model with the adapter
    model = PeftModel.from_pretrained(model, adapter_path)

    # Activate the adapter (this step is typically not needed for PEFT models as they're active by default)
    model.set_adapter("default")
    print("PEFT adapter loaded and set active")

    # Load the processor
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=FINETUNING_DIR, max_pixels=MAX_PIXELS)
    print("Loaded processor")

    print("Loading data...")
    df = pd.read_csv(csv_path)
    print(f"Data loaded. Total rows: {len(df)}")

    output_file = f"predicted_{dataset}.csv"
    fieldnames = ['id', 'Category']

    for col in df.columns[2:]:
        fieldnames.append(col)

    # Initialize CSV file with headers if it doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def process_chunk(chunk):
        messages = []
        for c in chunk.itertuples():
            image_name = str(c.id).zfill(6) + '.jpg'

            # Get the column names starting from the third one onward (ignoring 'id' and 'Category')
            attributes = [col for col in chunk.columns[2:]] 
            
            # Join the column names with a comma
            attributes_str = ', '.join(attributes)
            
            # Generate the prompt using column names
            prompt = f"Analyze this {c.Category} image and identify {attributes_str}."

            message = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": os.path.join(TEST_IMG_DIR, image_name)},
                    {"type": "text", "text": prompt}
                ]
            }]
            messages.append(message)

        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to("cuda")
        model = model.to("cuda")

        with torch.no_grad():  # Disable gradient calculation for inference
            generated_ids = model.generate(**inputs, max_new_tokens=128)  # Adjust token length based on needs

        # Trim input tokens from generated output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        results = []
        for idx, category, txt in zip(chunk["id"], chunk['Category'], output_texts):
            result = {'id': idx, 'Category': category}

            predictions = txt.split(",")
            for i, col in enumerate(chunk.columns[2:]):
                result[col] = predictions[i]

            results.append(result)
       
        return results

    def write_to_file(results, output_file, fieldnames):
        with open(output_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(results)

    # Example usage of processing chunks and writing to a file
    chunk_size = 1  # Adjust based on your GPU memory
    results_buffer = []

    print("Starting processing...")
    with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers based on CPU/GPU capacity
        future_to_chunk = {executor.submit(process_chunk, df.iloc[i:i + chunk_size]): i for i in range(0, len(df), chunk_size)}

        for future in as_completed(future_to_chunk):
            try:
                chunk_results = future.result()
                results_buffer.extend(chunk_results)
                print(f"Processed {len(results_buffer)} items so far.")

                # Write to file every 20 processed items
                if len(results_buffer) >= 500:
                    write_to_file(results_buffer, output_file, fieldnames)
                    print(f"Wrote {len(results_buffer)} items to file.")
                    results_buffer = []

            except Exception as e:
                print(f"Error processing chunk: {e}")
                continue

    # Write any remaining results
    if results_buffer:
        write_to_file(results_buffer, output_file, fieldnames)
        print(f"Wrote final {len(results_buffer)} items to file.")

    print("Processing completed.")

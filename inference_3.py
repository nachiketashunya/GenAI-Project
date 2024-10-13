# Load data
import pandas as pd
import os
import csv
import torch
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel, PeftConfig

# Define directories and constants
TRAIN_IMG_DIR = "/scratch/data/m23csa016/meesho_data/train_images"
FINETUNING_DIR = "/scratch/data/m23csa016/meesho_data/finetuning"
CSV_DIR = "/iitjhome/m23csa016/meesho_code/1_missing_attrs"
MAX_PIXELS = 1280 * 28 * 28

# Adapter paths (only one adapter here, can add more as needed)
adapter_paths = {
    'cs_men_tshirts': {
        'model': os.path.join(FINETUNING_DIR, "cs_men_tshirts/checkpoint-150"),
        'csv_path': os.path.join(CSV_DIR, "cs_men_tshirts_1_missing.csv"),
        'pred_dir': "/iitjhome/m23csa016/meesho_code/cat_specific_attrs_fine/pred"
    },
    'cs_women_tshirts': {
        'model': os.path.join(FINETUNING_DIR, "cs_women_tshirts/checkpoint-50"),
        'csv_path': os.path.join(CSV_DIR, "cs_women_tshirts_1_missing.csv"),
        'pred_dir': "/iitjhome/m23csa016/meesho_code/cat_specific_attrs_fine/pred"
    },
    'cs_sarees': {
        'model': os.path.join(FINETUNING_DIR, "cs_sarees/checkpoint-250"),
        'csv_path': os.path.join(CSV_DIR, "cs_sarees_1_missing.csv"),
        'pred_dir': "/iitjhome/m23csa016/meesho_code/cat_specific_attrs_fine/pred"
    },
    'cs_women_tops': {
        'model': os.path.join(FINETUNING_DIR, "cs_women_tops/checkpoint-300"),
        'csv_path': os.path.join(CSV_DIR, "cs_women_tops_1_missing.csv"),
        'pred_dir': "/iitjhome/m23csa016/meesho_code/cat_specific_attrs_fine/pred"
    },
    'cs_kurtis': {
        'model': os.path.join(FINETUNING_DIR, "cs_kurtis/checkpoint-200"),
        'csv_path': os.path.join(CSV_DIR, "cs_kurtis_1_missing.csv"),
        'pred_dir': "/iitjhome/m23csa016/meesho_code/cat_specific_attrs_fine/pred"
    }    
}

category_prompts = {
    "Women Tshirts": "Given this product image of 'Women Tshirts' category, what are surface_styling of the product?",
    "Women Tops & Tunics": "Given this product image of 'Women Tops & Tunics' category, what are occasion, neck_collar, surface_styling of the product?",
    "Kurtis": "Given this product image of 'Kurtis' category, what are occasion, ornamentation of the product?",
    "Men Tshirts": "Given this product image of 'Men Tshirts' category, what are neck, sleeve_length of the product?",
    "Sarees": "Given this product image of 'Sarees' category, what are occasion, ornamentation, blouse_pattern, border, border_width, pallu_details, transparency of the product?"
}


# Define a function to load model and process data chunks
def process_chunk(chunk, model_name, adapter_path):
    # Prepare messages for each row in the chunk
    messages = []
    for c in chunk.itertuples():
        image_name = str(c.id).zfill(6) + '.jpg'
    
        prompt = category_prompts[c.Category]
        
        message = [{
            "role": "user",
            "content": [
                {"type": "image", "image": os.path.join(TRAIN_IMG_DIR, image_name)},
                {"type": "text", "text": prompt}
            ]
        }]
        messages.append(message)

    # Process inputs for the model
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

    # Generate predictions
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    # Post-process outputs
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    results = []
    for idx, category, txt in zip(chunk["id"], chunk['Category'], output_texts):
        result = {'id': idx, 'Category': category}

        predictions = txt.replace("'", "").split(",")

        for i, col in enumerate(chunk.columns[2:]):
            result[col] = predictions[i].strip()
        results.append(result)
    
    print(f"Results: {results}")

    # Clear CUDA cache
    torch.cuda.empty_cache()

    return results

# Function to write results to file
def write_to_file(results, output_file, fieldnames):
    with open(output_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(results)

# Main processing loop
for dataset, values in adapter_paths.items():
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    adapter_path = values['model']
    csv_path = values['csv_path']

    pred_dir = values['pred_dir']
    os.makedirs(pred_dir, exist_ok=True)

    # Load and configure model
    print("Loading model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map="cuda",
        cache_dir=FINETUNING_DIR
    )

    # Load PEFT adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    # Load processor
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=FINETUNING_DIR, max_pixels=MAX_PIXELS)

    # Load CSV data
    print(f"Loading data for {dataset}...")
    df = pd.read_csv(csv_path)
    print(f"Data loaded. Total rows: {len(df)}")

    output_file = os.path.join(pred_dir, f"predicted_{dataset}.csv")
    fieldnames = ['id', 'Category'] + list(df.columns[2:])

    # Initialize CSV file with headers if it doesn't exist
    if not os.path.exists(output_file):
        with open(os.path.join(pred_dir, output_file), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    # Set chunk size and buffer for results
    chunk_size = 1  # Adjust based on GPU memory
    results_buffer = []

    # Process the CSV file in chunks sequentially
    print("Starting sequential processing...")
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        try:
            chunk_results = process_chunk(chunk, model_name, adapter_path)
            results_buffer.extend(chunk_results)
            print(f"Processed {len(results_buffer)} items so far.")

            # Write to file every 500 processed items
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

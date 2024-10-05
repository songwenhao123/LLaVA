import os
import argparse
import torch
import pandas as pd

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
from tqdm import tqdm

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    data = pd.read_csv(args.csv_file)
    answers = [None] * len(data)
    batch_size = args.batch_size

    # Split data into batches
    num_batches = (len(data) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        batch_data = data[batch_idx * batch_size:(batch_idx + 1) * batch_size]

        # Process each sample individually
        for index, row in batch_data.iterrows():
            image_path = os.path.join(args.image_folder, row['Path'])
            question = row['Questions']

            if not os.path.exists(image_path):
                print(f"Warning: Image {row['Path']} not found.")
                answers[index] = ""
                continue

            # Load image
            image = load_image(image_path)
            image_tensor = process_images([image], image_processor, model.config)
            if type(image_tensor) is list:
                image_tensor = image_tensor[0].to(model.device, dtype=torch.float16)
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)

            # Prepare prompt
            conv = conv_templates[args.conv_mode].copy()
            if "mpt" in model_name.lower():
                roles = ('user', 'assistant')
            else:
                roles = conv.roles

            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + question

            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

            with torch.no_grad():
                try:
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0),
                        image_sizes=[image.size],
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True
                    )
                except RuntimeError as e:
                    print(f"RuntimeError during generation: {e}")
                    answers[index] = "Error during generation"
                    continue

            # Decode output and store answer
            outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            answers[index] = outputs

            if args.debug:
                print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

    # Save answers back to CSV
    data['Answers'] = answers
    data.to_csv(args.csv_file, index=False)
    print(f"Answers saved to {args.csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--csv-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of samples to read in a batch")
    args = parser.parse_args()
    main(args)
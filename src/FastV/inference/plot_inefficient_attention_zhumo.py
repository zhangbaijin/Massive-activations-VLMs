import os
import argparse
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import json
from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str, required=True, default="llava-v1.5-13b")
    parser.add_argument('--image-path', type=str, required=True, help='figs/example.jpg')
    parser.add_argument('--prompt', type=str, required=True, help='Describe the image in detail')
    parser.add_argument('--output-path', type=str, required=True, help='The path to save the output json file')
    pargs = parser.parse_args()

    class InferenceArgs:
        model_path = pargs.model_path
        model_base = None
        image_file = None
        device = "cuda"
        conv_mode = None
        temperature = 0.2
        max_new_tokens = 512
        load_8bit = False
        load_4bit = False
        debug = False
        image_aspect_ratio = 'pad'

    args = InferenceArgs()

    # Load the pretrained model and tokenizer
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
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

    model.config.use_fast_v = False
    model.model.reset_fastv()

    total_layers = model.config.num_hidden_layers

    def inference_and_analyze(prompts, images, append_output="", num_generation_steps=3):
        outputs = []
        outputs_attention = []
        hidden_states_analysis = []

        for prompt, image in tqdm(zip(prompts, images), total=len(prompts)):
            image = load_image(image)
            image_tensor = process_images([image], image_processor, args)
            conv = conv_templates[args.conv_mode].copy()
            if isinstance(image_tensor, list):
                image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)

            inp = prompt

            if image is not None:
                # First message with image tokens
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt() + append_output

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            for step in range(num_generation_steps):
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor,
                        attention_mask=None,
                        do_sample=False,
                        max_new_tokens=256,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria],
                        output_attentions=True,
                        output_hidden_states=True,  # Output hidden states
                        output_scores=True,
                        return_dict_in_generate=True
                    )

                output = tokenizer.decode(output_ids['sequences'][0, input_ids.shape[1]:], skip_special_tokens=True).strip().replace("</s>", "")
                outputs.append(output)
                print(f"Step {step + 1} output: {output}")

                # Extract attention and hidden states
                outputs_attention.append(output_ids['attentions'])
                hidden_states = output_ids['hidden_states']
                
                # Analyze hidden states for max activations and weights
                activation_analysis = []
                for layer_idx, hidden_state in enumerate(hidden_states[:-1]):
                    next_layer_weight = model.model.layers[layer_idx].self_attn.q_proj.weight
                    max_value, max_index = torch.max(hidden_state, dim=-1)

                    for batch_idx in range(max_index.size(0)):
                        for seq_idx in range(max_index.size(1)):
                            max_channel = torch.argmax(hidden_state[batch_idx, seq_idx, :]).item()
                            max_weight_value = next_layer_weight[:, max_channel].max().item()

                            analysis_entry = {
                                "generation_step": step,
                                "layer": layer_idx,
                                "batch": batch_idx,
                                "sequence": seq_idx,
                                "max_value": max_value[batch_idx, seq_idx].item(),
                                "max_value_channel": max_channel,
                                "weight": max_weight_value
                            }
                            activation_analysis.append(analysis_entry)

                hidden_states_analysis.append(activation_analysis)

                # Generate new token for the next step
                new_token_id = torch.argmax(output_ids.logits[:, -1, :], dim=-1)
                input_ids = torch.cat((input_ids, new_token_id.unsqueeze(-1)), dim=-1)

        return outputs, outputs_attention, hidden_states_analysis

    # %%

    prompts = [pargs.prompt]
    images = [pargs.image_path]

    model_output_ori, outputs_attention, hidden_states_analysis = inference_and_analyze(prompts, images, num_generation_steps=3)

    output_path = pargs.output_path

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path + "/attn_maps", exist_ok=True)

    with open(output_path + "/output.json", "w") as f:
        json.dump({"prompt": pargs.prompt, "image": pargs.image_path, "output": model_output_ori}, f, indent=4)

    with open(output_path + "/hidden_states_analysis.json", "w") as f:
        json.dump(hidden_states_analysis, f, indent=4)

    # Draw attention maps (can be added back if needed)

# %%
import os
# %%
import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.yxs.generation_utils import build_logits_processor, build_model_kwargs
from PIL import Image
import json
from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import os
from datasets import load_from_disk,load_dataset
import torch
import json
from tqdm import tqdm
import re	

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as Colormap
from matplotlib.colors import LogNorm
# 使用自定义色图
from matplotlib.colors import LinearSegmentedColormap
colors = ["white", "blue", "green", "red"]  # 定义颜色列表，从背景色到最深色。
cmap_name = "my_cmap"
n_bins = 100  # 颜色级别越多，过渡越平滑
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)



def visualize_attention(multihead_attention,output_path="atten_map_1.png",title="Layer 5"):
    # Assuming the input is a numpy array of shape (1, num_heads, n_tokens, n_tokens)
    # First, we average the attention scores over the multiple heads
    averaged_attention = torch.mean(multihead_attention, axis=1)[0].float()# Shape: (n_tokens, n_tokens)
    
    # pooling the attention scores  with stride 20
    averaged_attention = torch.nn.functional.avg_pool2d(averaged_attention.unsqueeze(0).unsqueeze(0), 10, stride=10).squeeze(0).squeeze(0)
    
    cmap = plt.cm.get_cmap("viridis")
    plt.figure(figsize=(5, 5),dpi=400)

    # Log normalization
    # log_norm = LogNorm(vmin=0.0007, vmax=averaged_attention.max())
    # Log normalization
    log_norm = LogNorm(vmin=0.0001, vmax=averaged_attention.max())  # 降低 vmin 值以增加颜色深度
    # set the x and y ticks to 20x of the original
    ax = sns.heatmap(
    averaged_attention,
    cmap=cmap,  # 应用新的色图
    norm=log_norm,
)
    # remove the x and y ticks
    
    # replace the x and y ticks with string
    x_ticks = [str(i*10) for i in range(0,averaged_attention.shape[0])]
    y_ticks = [str(i*10) for i in range(0,averaged_attention.shape[0])]
    ax.set_xticks([i for i in range(0,averaged_attention.shape[0])])
    ax.set_yticks([i for i in range(0,averaged_attention.shape[0])])
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)
    # change the x tinks font size
    plt.xticks(fontsize=3)
    plt.yticks(fontsize=3)
    
    # make y label vertical
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)     
    
    plt.title(title)
    # tight layout
    plt.savefig(output_path, bbox_inches='tight')
    # plt.show()

    top_five_attentions = []
    for row in averaged_attention:
        # Use torch.topk to get the top 5 values and their indices
        top_values, top_indices = torch.topk(row, 10)
        # Convert to lists and append to the overall list
        top_five_line = list(zip(top_indices.tolist(), top_values.tolist()))
        top_five_attentions.append(top_five_line)
        
    return top_five_attentions,averaged_attention    

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str, required=True, default="llava-v1.5-13b")
    parser.add_argument('--image-path', type=str, required=True, help='figs/example.jpg')
    parser.add_argument('--prompt', type=str, default="<image>\nContext: Below is a food web from a tundra ecosystem in Nunavut, a territory in Northern Canada.\nA food web models how the matter eaten by organisms moves through an ecosystem. The arrows in a food web represent how matter moves between organisms in an ecosystem.\nQuestion: Which of these organisms contains matter that was once part of the lichen?\nOptions: (A) bilberry (B) mushroom\nAnswer with the option's letter from the given choices directly.",required=True, help='discribe the image in detail')
    parser.add_argument('--output-path', type=str, required=True, help='the path to save the output json file')
    pargs = parser.parse_args()

        # %%
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
    # %%
    args = InferenceArgs()
    # %%
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    model.eval()
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

    

    # model.config.use_fast_v = False
    # model.model.reset_fastv()

    total_layers = model.config.num_hidden_layers


    # %%
    def inference(prompts,images,append_output=""):
        outputs = []
        outputs_attention = []
        for prompt,image in tqdm(zip(prompts,images),total=len(prompts)):
            image = load_image(image)
            image_tensor = process_images([image], image_processor, args)
            conv = conv_templates[args.conv_mode].copy()
            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)

            inp = prompt

            if image is not None:
                # first message
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp # False
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                conv.append_message(conv.roles[0], inp)
                image = None
            else:
                # later messages
                conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt() + append_output

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            # with torch.inference_mode():
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
                    output_scores=True,
                    return_dict_in_generate=True,
                    output_hidden_states=True
                    )
                # output_ids = model(
                #     input_ids,
                #     images=image_tensor,
                #     attention_mask=None,
                #     output_attentions=True,
                #     output_hidden_states=True
                #     )
            # logits_processor = build_logits_processor(model, tokenizer, model_name)
            # next_token_logits = output_ids.logits[:, -1, :]
            # next_tokens_scores = logits_processor(input_ids, next_token_logits)
            # next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            # output=tokenizer.decode(next_tokens)
            

            output = tokenizer.decode(output_ids['sequences'][0, input_ids.shape[1]:],skip_spectial_tokens=True).strip().replace("</s>","")
            outputs.append(output)
            print("Output---",output)
            # print(output_ids.shape)

            outputs_attention.append(output_ids['attentions'])

            hidden_states=output_ids.hidden_states
            # print("hidden_states:",hidden_states)
            print(type(hidden_states))
            print(type(hidden_states[0]))
            # print(hidden_states[0][0].shape)#torch.Size([1, 625, 4096])
            # print(hidden_states[0][1].shape)
            # print(hidden_states[0][2].shape)
            # print(hidden_states[0][3].shape)
            print(hidden_states[0][32].shape)#最大为32
            # print(hidden_states[1][0].shape)#torch.Size([1, 1, 4096])
            # print(hidden_states[600][0].shape)
            hidden_states=hidden_states[0]

            # 打开JSON文件以追加模式写入
            with open('activation_results.json', 'a') as f:  # 修改为追加模式
                for i, hidden_state in enumerate(hidden_states[:-1]):
                    next_layer_weight = model.model.layers[i].self_attn.q_proj.weight
                    max_value, max_index = torch.max(hidden_state, dim=-1,keepdim=True)
                    for batch_idx in range(max_index.size(0)):
                        for seq_idx in range(max_index.size(1)):
                            max_channel = max_index[batch_idx, seq_idx].item()  # 使用max_index而不是重新计算
                            max_weight_value = next_layer_weight[:, max_channel].max().item()

                            line = {
                                "layer": i,
                                "batch": batch_idx,
                                "sequence": seq_idx,
                                "max_value": max_value[batch_idx, seq_idx].item(),
                                "max_value_channel": max_channel,
                                "weight": max_weight_value
                            }
                            f.write(json.dumps(line) + '\n')
                    closest_to_zero_value = hidden_state.view(-1).abs().min().item()
                    line = {
                        "layer": i,
                        "closest_to_zero_value": closest_to_zero_value
                    }
                    f.write(json.dumps(line) + '\n')

                    # 找到最接近零的激活值
                    closest_to_zero_value = hidden_state.view(-1).abs().min().item()
                    line = {
                        "layer": i,
                        "closest_to_zero_value": closest_to_zero_value
                    }
                    f.write(json.dumps(line) + '\n')

                # 使用生成的输出继续前向传播，观察激活值的变化
                # for step in range(3):  # 生成3步
                #     with torch.no_grad():
                #     #     outputs = model.generate(
                #     # input_ids,
                #     # images=image_tensor,
                #     # attention_mask=None,
                #     # do_sample=False,
                #     # max_new_tokens=256,
                #     # use_cache=True,
                #     # stopping_criteria=[stopping_criteria],
                #     # output_attentions=True,
                #     # output_scores=True,
                #     # return_dict_in_generate=True,
                #     # output_hidden_states=True
                #     # )
                #         output_ids = model(input_ids,
                #     images=image_tensor,
                #     output_hidden_states=True,
                #     output_attentions=True,)
                #         hidden_states = output_ids.hidden_states
                #         # hidden_states=hidden_states[1]
                #         for i, hidden_state in enumerate(hidden_states[:-1]):
                #             max_value, max_index = torch.max(hidden_state, dim=-1)
                            
                #             for batch_idx in range(max_index.size(0)):
                #                 for seq_idx in range(max_index.size(1)):
                #                     # 获取最大值所在的通道
                #                     max_channel = torch.argmax(hidden_state[batch_idx, seq_idx, :]).item()
                                    
                #                     # 找到对应的权重值
                #                     max_weight_value = model.model.layers[i].self_attn.q_proj.weight[:, max_channel].max().item()
                                    
                #                     # 构建一行记录并写入文件
                #                     line = {
                #                         # "generation_step": step,
                #                         "layer": i,
                #                         # "batch": batch_idx,
                #                         "sequence": seq_idx,
                #                         "max_value": max_value[batch_idx, seq_idx].item(),
                #                         "max_value_channel": max_channel,
                #                         # "weight": max_weight_value
                #                     }
                #                     f.write(json.dumps(line) +','+'\n')
                            
                #             closest_to_zero_value = hidden_state.view(-1).abs().min().item()
                #             line = {
                #                 "generation_step": step,
                #                 "layer": i,
                #                 "closest_to_zero_value": closest_to_zero_value
                #             }
                #             f.write(json.dumps(line) + '\n')
                    
                #     # 假设生成新的token并更新input_ids
                #     logits_processor = build_logits_processor(model, tokenizer, model_name)
                #     # 选择最后一个token的logits
                #     next_token_logits = output_ids.logits[:, -1, :]
                #     next_tokens_scores = logits_processor(input_ids, next_token_logits)
                #     new_token_id = torch.argmax(next_tokens_scores, dim=-1)
                #     # zhumo
                #     # logits=outputs["scores"]
                #     # print(logits)
                #     # new_token_id = torch.argmax(logits[:, -1, :], dim=-1)
                #     input_ids = torch.cat((input_ids, new_token_id.unsqueeze(-1)), dim=-1)
                    
        return outputs,outputs_attention
    

        # %%

    # %%

    prompts = [pargs.prompt]
    images = [pargs.image_path]

    model_output_ori,outputs_attention = inference(prompts,images)
    model_output,outputs_attention = inference(prompts,images,append_output=model_output_ori[0])

   

    output_path = pargs.output_path

    try:
        os.mkdir(output_path)
    except:
        pass

    try:
        os.mkdir(output_path+"/attn_maps")
    except:
        pass


    with open(output_path+"/output.json","w") as f:
        # json dumps
        json.dump({"prompt":pargs.prompt,"image":pargs.image_path,"output": model_output_ori},f,indent=4)

    # draw attention maps
    for i in outputs_attention:
        for j in range(0,total_layers):
            top5_attention,average_attentions = visualize_attention(i[0][j].cpu(),output_path=output_path+"/attn_maps/atten_map_"+str(j)+".png",title="Layer "+str(j+1))





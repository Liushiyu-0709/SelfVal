import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch import nn
import numpy as np
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import SampleOutput
from transformers.generation import SampleEncoderDecoderOutput, SampleDecoderOnlyOutput
from llava.constants import IMAGE_TOKEN_INDEX
import argparse
import torch
import os
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
import re
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from PIL import Image

import requests, random
from io import BytesIO


import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
import numpy as np

import json

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out
def sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id


    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )

    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
    
    
    position_scores, js_scores, js_scores_vcd = [], [], []
    
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  
    model_kwargs_cd, model_kwargs_dd = None, None
    
    while True:
        if synced_gpus:
            
            
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            
            if this_peer_finished_flag.item() == 0.0:
                break

        
        
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            continue  

        next_token_logits = outputs.logits[:, -1, :]
        
        
        use_cd = model_kwargs.get("images_cd") != None
        use_dd = model_kwargs.get("use_dd")
        use_dd_unk = model_kwargs.get('use_dd_unk')
        output_attentions_wo_img = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states_wo_img = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        if use_cd or use_dd or use_dd_unk:
            if use_cd:
                model_kwargs_cd = model_kwargs.copy()
                model_inputs_cd = self.prepare_inputs_for_generation_cd(input_ids, **model_kwargs_cd)
            else: 
                model_kwargs_cd = model_kwargs.copy() if model_kwargs_cd is None else model_kwargs_cd
                if use_dd_unk:
                    input_ids_dd = copy.deepcopy(input_ids)
                    input_ids_dd[input_ids_dd == IMAGE_TOKEN_INDEX] = 0 
                elif use_dd:
                    indices_to_keep = torch.where(input_ids != IMAGE_TOKEN_INDEX)[0]
                    
                    attention_mask = model_kwargs['attention_mask']
                    model_kwargs_cd['attention_mask'] = attention_mask[:,indices_to_keep]
                    input_ids_dd = input_ids[input_ids != IMAGE_TOKEN_INDEX].unsqueeze(0)
                model_inputs_cd = self.prepare_inputs_for_generation_cd(input_ids_dd, **model_kwargs_cd)
            
            outputs_cd = self(
                **model_inputs_cd,
                return_dict=True,
                output_attentions=output_attentions_wo_img,
                output_hidden_states=output_hidden_states_wo_img,
            )
            next_token_logits_cd = outputs_cd.logits[:, -1, :]
            
            cd_alpha = model_kwargs.get("cd_alpha") if model_kwargs.get("cd_alpha") is not None else 0.2
            cd_beta = model_kwargs.get("cd_beta") if model_kwargs.get("cd_beta") is not None else 0.1
            
            cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
            
            diffs = (1+cd_alpha)*next_token_logits - cd_alpha*next_token_logits_cd
            next_token_logits_vcd = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

            js_scores_vcd.append(safe_jsd(nn.functional.softmax(next_token_logits_vcd[0]).cpu(), nn.functional.softmax(next_token_logits_cd[0]).cpu()))

            js_scores.append(safe_jsd(nn.functional.softmax(next_token_logits[0]).cpu(), nn.functional.softmax(next_token_logits_cd[0]).cpu()))

            
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)
        probs = nn.functional.softmax(next_token_scores, dim=-1)

        next_tokens = torch.argmax(probs, dim=-1)  
        position_scores.append(next_token_scores[0, next_tokens].item())

        
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )


        
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        
        
        
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        
        
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids, js_scores, js_scores_vcd

def eval_model(args):
    

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

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
        conv_mode = "llava_v1"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    
    
    with torch.inference_mode():
        output_ids, js_scores, js_scores_vcd = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            use_dd=True,
        )
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    return js_scores, js_scores_vcd, outputs


def wasserstein(dist1, dist2):
    from scipy.stats import wasserstein_distance
    wd = wasserstein_distance(dist1, dist2)
    return wd
def safe_jsd(p, q, epsilon=1e-5):
    
    
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return 0.5 * (kl_pm + kl_qm)
def js(dist1, dist2):
    from scipy.spatial.distance import jensenshannon
    jsd_naive = jensenshannon(dist1, dist2) ** 2
    return jsd_naive

import sys
from chair import CHAIR
import pickle
evaluator = pickle.load(open("./experiments/scripts/chair500/chair.pkl", 'rb'))



def get_object_positions(outputs_ls, js_scores_ls):
    assert len(outputs_ls) == len(js_scores_ls)
    object_js_scores = [] 
    js_scores_bins = [[] for _ in range(5)]
    
    for i in range(len(outputs_ls)):
        outputs, js_scores = outputs_ls[i], js_scores_ls[i]
        token_ids = tokenizer(outputs)['input_ids']
        if len(token_ids) != len(js_scores):
            print("=========what?=======", len(token_ids), len(js_scores))
        
        start_index = 0 
        mapping = [] 
        for word in outputs.split(" "):
            i1 = start_index
            if start_index == 0:
                word_token = tokenizer(word)['input_ids']
            else:
                word_token = tokenizer(word)['input_ids'][1:] 

            i2 = i1 + len(word_token) - 1
            mapping.append((i1, i2))
            start_index = i2 + 1
            
    
    
        
        
    
        object_indexs = []
        has_obj = set()
        words =  outputs.split(" ")
        for i in range(len(words)):
            word = evaluator.caption_to_words(words[i])[0]
            if len(word):
                
                if word[0] not in has_obj:
                    has_obj.add(word[0])
                    object_indexs.append(i)
        print("object_indexs:", object_indexs)
        
        for index in object_indexs:
            token_index = mapping[index][0] - 1
            
            relative_index = int((index / len(outputs.split(" ")) - 1) // 0.2)
            
            
            js_scores_bins[relative_index].append(np.average(js_scores[mapping[i][0]-1:mapping[i][1]]))
    js_scores_bins_avg = [np.average(js_scores_bins[i]) for i in range(5)]
    return js_scores_bins_avg

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

def plot_hist(data1, data2, data3):
    
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    plt.rcParams['font.weight'] = 'bold'

    
    bin_values = ['[0.0, 0.2)', '[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.8)', '[0.8, 1.0]']
    x = np.arange(len(bin_values))

    
    base_hr = [3.7, 16.3, 31.1, 50.7, 55.39]
    vcd_hr  = [3.1, 15.0, 33.2, 46.5, 55.4]
    deco_hr = [3.6, 13.6, 26.4, 45.2, 54.6]

    colors_bar  = ['
    colors_line = ['

    
    plt.rcParams['xtick.labelsize'] = 22
    plt.rcParams['ytick.labelsize'] = 22
    fig, ax_left = plt.subplots(figsize=(10, 6), constrained_layout=True)

    bar_width = 0.22
    
    ax_left.bar(x - bar_width, data1, color=colors_bar[0], width=bar_width, label='Origin')
    ax_left.bar(x,             data2, color=colors_bar[1], width=bar_width, label='VCD')
    ax_left.bar(x + bar_width, data3, color=colors_bar[2], width=bar_width, label='DeCo')
    ax_left.set_ylabel('JSD', fontsize=28, weight='bold')

    
    ax_right = ax_left.twinx()
    ax_right.plot(x, base_hr, color=colors_line[0], marker='o', markersize=10,
                  linewidth=3, alpha=0.8, label='Origin')
    ax_right.plot(x, vcd_hr,  color=colors_line[1], marker='s', markersize=10,
                  linewidth=3, alpha=0.8, label='VCD')
    ax_right.plot(x, deco_hr, color=colors_line[2], marker='^', markersize=10,
                  linewidth=3, alpha=0.8, label='DeCo')
    ax_right.set_ylabel('Hallucination Rate (%)', fontsize=28, weight='bold')

    
    ax_left.set_xlabel('Object Relative Position', fontsize=28, weight='bold', labelpad=12)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(bin_values)

    
    handles_left,  labels_left  = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    
    fig.legend(handles_left, labels_left,
               loc='upper center', ncol=3,
               
               fontsize=18, frameon=True)
    
    
    plt.savefig("bin.pdf", dpi=400)
    plt.savefig("bin.png", dpi=400)
    plt.close()

def plot_hist_old(data1, data2, data3):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    plt.rcParams['font.sans-serif'] = "Times New Roman"
    plt.rcParams['font.weight'] = 'bold'
    bin_values = ['[0.0, 0.2)', '[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.8)', '[0.8, 1.0]']
    bar_width = 0.25
    x = np.arange(len(bin_values))
    colors = ['
    plt.rcParams['xtick.labelsize'] = 22  
    plt.rcParams['ytick.labelsize'] = 22  
    plt.figure(figsize=(12, 6))
    
    plt.bar(x - bar_width, data1, color=colors[0], width=bar_width, label='Origin')
    

    plt.bar(x, data2, color=colors[1], width=bar_width, label='VCD')
    

    plt.bar(x + bar_width, data3, color=colors[2], width=bar_width, label='DeCo')
    


    
    
    
    
    
    plt.xticks(x, bin_values)
    
    plt.legend(fontsize=20, loc='upper right')
    
    
    plt.xlabel('Object Relative Position', fontsize=28, weight='bold', labelpad=12)
    plt.ylabel('JSD', fontsize=28, weight='bold', )
    plt.tight_layout()
    plt.savefig("bin.pdf", dpi=400)
    plt.savefig("bin.png", dpi=400)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", action='store_true', default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--use_dd", action='store_true', default=False)
    parser.add_argument("--use_dd_unk", action='store_true', default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only-plot", type=bool, default=False)
    
    args = parser.parse_args()
    args.use_dd = True
    args.use_dd_unk = True
    
    if not args.only_plot:
        transformers.generation.utils.GenerationMixin.sample = sample    

        
        coco_val_path = "val2017"  
        
        all_images = sorted([
            os.path.join(coco_val_path, f) 
            for f in os.listdir(coco_val_path) 
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ]) 
        random.seed(2)
        sample_num = 100
        selected_images = random.sample(all_images, min(sample_num, len(all_images)))
        print("selected_images:", selected_images) 

        
        disable_torch_init()
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            args.model_path, args.model_base, model_name
        )
        js_scores_ls = [0 for i in range(150)]
        js_scores_len_ls = [0 for i in range(150)]

        js_scores_vcd_ls = [0 for i in range(150)]
        js_scores_len_vcd_ls = [0 for i in range(150)]
        
        otuputs_ls,js_scores_forbin, js_scores_vcd_forbin = [], [], []
        
        for i in tqdm(range(sample_num)):
            args.image_file = selected_images[i]
            js_scores, js_scores_vcd, outputs = eval_model(args)
            otuputs_ls.append(outputs)
            js_scores_forbin.append(js_scores)
            js_scores_vcd_forbin.append(js_scores_vcd)
            for j, s in enumerate(js_scores):
                if j >= 150:
                    break
                js_scores_ls[j] += s
                js_scores_len_ls[j] += 1
            for j, s in enumerate(js_scores_vcd):
                if j >= 150:
                    break
                js_scores_vcd_ls[j] += s
                js_scores_len_vcd_ls[j] += 1
        
        js_scores_bins_avg = get_object_positions(otuputs_ls, js_scores_forbin)
        js_scores_vcd_bins_avg = get_object_positions(otuputs_ls, js_scores_vcd_forbin)
        js_scores_deco_bins_avg = get_object_positions(json.load(open("outputs.json", "r")), json.load(open("jsd_save.json", "r")))

        print("js_scores_bin:", js_scores_bins_avg)
        print("js score vcd bin:", js_scores_vcd_bins_avg)
        print("js_scores_deco_bins_avg:", js_scores_deco_bins_avg)

        json.dump([js_scores_bins_avg, js_scores_vcd_bins_avg, js_scores_deco_bins_avg], open("jsd_bins.json", "w"))
        print("已写入: jsd_bins.json")
        plot_hist(js_scores_bins_avg, js_scores_vcd_bins_avg, js_scores_deco_bins_avg)
    
        
        
        
        
        
        
    else:
        with open("jsd_bins.json", 'r') as f:  
            js_scores_bins_avg, js_scores_vcd_bins_avg, js_scores_deco_bins_avg = json.load(f)
        plot_hist(js_scores_bins_avg, js_scores_vcd_bins_avg, js_scores_deco_bins_avg)


    
    
    
    
    
    
    


    
    
    
    

    
    

    
    

    
    

    
    
    
    
    
    
    
    
    

    
import json

import base64, openai
import requests
import time
import random, os, sys
import numpy as np

sys.path.append("../chair500")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chair import CHAIR
from llava.eval.selection import *
from myutils import sample
import pickle
from tqdm import tqdm
from myutils import pos_caption_to_words, remove_adj_noun, predeined_set_caption_to_words

import transformers
import time


original_sample = transformers.generation.utils.GenerationMixin.sample


def cal_selfconsistency(args, model, tokenizer, images_tensor, caption): 
    
    objects = evaluator.caption_to_words(caption)[0] 
    score = 0
    qs = "Describe any element of the image with only one word or phrase."
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    measure_tokens_dict = {}
    for obj in objects:
        object_name = obj.capitalize() 
        object_name_token = tokenizer(object_name, add_special_tokens=False).input_ids
        measure_tokens_dict[obj] = object_name_token.copy()

    with torch.inference_mode():
        _, objects_probs_dict = model.generate(
            input_ids,
            images=images_tensor.unsqueeze(0).half().cuda(),
            do_sample=True,
            max_new_tokens=1024,
            use_cache=True,
            measure_tokens_dict=measure_tokens_dict
        )
    
    for obj in objects:
        score += np.prod(objects_probs_dict[obj])
    score = score/len(objects) if len(objects) > 0 else 1.0
    return score

def extract_objects(args, model, tokenizer, caption):
        
    
        
    qs = f'''
Task: Extract concrete physical objects from the text. Exclude abstract concepts. Return a JSON list with key "objects".
Examples:
Input: He bought a bicycle and sunglasses.
Output: {{ "objects": ["bicycle", "sunglasses"] }}
Input: "The idea of freedom filled the room."
Output: {{ "objects": [] }}
Input: The scene also includes a dirt road and a metal gate.
Output: {{ "objects": ["road", "gate"] }}

Now process this new input:
Input: "{caption}"
Output: 
'''
    model_name = get_model_name_from_path(args.model_path)
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    
    
    
    
    
    
    
    
    
    

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

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            do_sample=False, 
            temperature=0.0,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    input_token_len = input_ids.shape[1]

    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()

    print("caption:", caption)
    print("***outputs***:", outputs)
    try:
        objects = json.loads(outputs)['objects']
    except:
        return []
    new_objects = [remove_adj_noun(obj) for obj in objects if remove_adj_noun(obj) != None]
    
    
    print("new objects:", new_objects)

    return new_objects
    # return json.loads(outputs)['objects']


def filter_caption(args, model, tokenizer, images_tensor, caption, object_set=None):
    
    
    
    
    
    
    
    if args.extract_method == 'statis' and object_set is not None:
        objects = predeined_set_caption_to_words(caption, object_set)
        
    elif args.extract_method == "self":
        objects = extract_objects(args, model, tokenizer, caption)
    elif args.extract_method == "mscoco":
        objects = evaluator.caption_to_words(caption)[0]
    score = 0
    qs = "Describe any element of the image with only one word or phrase."
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    measure_tokens_dict = {}
    for obj in objects:
        object_name = obj.capitalize() 
        object_name_token = tokenizer(object_name, add_special_tokens=False).input_ids
        measure_tokens_dict[obj] = object_name_token.copy()

    with torch.inference_mode():
        _, objects_probs_dict = model.generate(
            input_ids,
            images=images_tensor.unsqueeze(0).half().cuda(),
            do_sample=True,
            max_new_tokens=1024,
            use_cache=True,
            measure_tokens_dict=measure_tokens_dict
        )

    remove_words = [word for word, prob in zip(objects_probs_dict.keys(), objects_probs_dict.values()) if np.prod(prob) <= args.threshold]
    for word in remove_words:
        if word in caption:
            caption = caption.replace(word, '[IDX]')
    sentences_ls = re.split(r'(?<=[。！？；：，.!?;:,])\s*', caption)
    sentences_ls = [sentence for sentence in sentences_ls if '[IDX]' not in sentence]

    return sentences_ls



def regenerate(args, model, tokenizer, caption, images_tensor=None):
    
    
    if isinstance(caption, list):
        
        qs = f'''
                Description from different sources:
                {'||'.join(str(c) for c in caption)}
                Given the materials above, give the final description:
                '''
    model_name = get_model_name_from_path(args.model_path)
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
        conv_mode = "llava_v0"

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

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor.unsqueeze(0).half().cuda(),
            
            do_sample=False, 
            temperature=0.0,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    input_token_len = input_ids.shape[1]

    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()

    print("outputs:", outputs)
    return outputs

def statis_objects(args, model, tokenizer): 
    
    
    if os.path.exists("most_common_mscoco_objects.json"):
        with open("most_common_mscoco_objects.json", 'r') as f:
            most_common_objects = json.load(f)
        return most_common_objects

    with open("MSCOCO/annotation/annotations/captions_train2014.json", 'r') as f:
        data = json.load(f)
        captions = [item['caption'] for item in random.sample(data['annotations'], 800)]
    
    all_objects = []
    for caption in tqdm(captions):
        objects_list = extract_objects(args, model, tokenizer, caption)
        all_objects.extend(objects_list)
    count = Counter(all_objects)
    
    
    
    most_common_objects = count.most_common(300)
    most_common_objects = [obj[0] for obj in most_common_objects if obj[0] not in ['image', 'scene']]
    

    
    with open("most_common_mscoco_objects.json", 'w') as f:
        json.dump(most_common_objects, f, indent=2)
    return most_common_objects

def sample_best(model ,tokenizer, image_tensor, captions):
    
    score_list = []
    for caption in captions:
        score = cal_selfconsistency(args, model, tokenizer, image_tensor, caption)
        score_list.append(score)
    max_index = np.argmax(np.array(score_list))
    return captions[max_index] 


def aggregate(model ,tokenizer, image_tensor, captions, most_common_objects=None):
    global verification_time, aggregation_time

    
    start_time = time.time() 
    caption_list = []
    for caption in captions:
        filtered_caption_ls = filter_caption(args, model, tokenizer, image_tensor, caption, object_set=most_common_objects)
        caption_list.extend(filtered_caption_ls)
    verification_time += time.time() - start_time 
    transformers.generation.utils.GenerationMixin.sample = original_sample
    
    start_time = time.time() 
    final_caption = regenerate(args, model, tokenizer, caption_list, image_tensor)
    aggregation_time += time.time() - start_time 
    return final_caption 

def sample_caption_model(args):
    global sampling_time
    from llava.eval.selection import sample_batch

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(args.model_path)
    compute_dtype = torch.float16
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    most_common_objects = None
    if args.extract_method == "statis":
        most_common_objects = statis_objects(args, model, tokenizer)
    
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        
        if 'POPE' in args.question_file:
            conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
            conv.append_message(conv.roles[1], None)
        else:
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image = Image.open(os.path.join(args.image_dir, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
         

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        start_time = time.time() 
        outputs_list = sample_batch(model, tokenizer, input_ids, image_tensor, args.temperature, args.sample_num, stop_str)
        sampling_time += time.time() - start_time

        
        transformers.generation.utils.GenerationMixin.sample = sample
        

        best_caption = aggregate(model, tokenizer, image_tensor, outputs_list, most_common_objects=most_common_objects) 
        ans_file.write(json.dumps({"question_id": idx,
                                    "prompt": cur_prompt,
                                    "text": best_caption,
                                    "model_id": model_name,
                                    "image": image_file,
                                    "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default="chair-500.jsonl")
    parser.add_argument("--model-path", type=str, default="checkpoints/llava1.5_7b")

    parser.add_argument("--cache", type=str, default="./experiments/scripts/chair500/chair.pkl")
    parser.add_argument("--answers-file", type=str, default="./experiments/out/time/fta3.jsonl")
    parser.add_argument("--sample-num", type=int, default=3)
    parser.add_argument("--image-dir", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--extract-method", type=str, default="mscoco") 

    args = parser.parse_args()

    evaluator = pickle.load(open(args.cache, 'rb'))

    sampling_time = 0 
    verification_time = 0
    aggregation_time = 0

    sample_caption_model(args)
    print("sampling_time:", sampling_time)
    print("verification_time:", verification_time)
    print("aggregation_time:", aggregation_time)

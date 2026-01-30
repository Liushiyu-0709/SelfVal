import json,os
from llava.eval.run_llava_times import *
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def load_json(f):
    ext = os.path.splitext(f)[-1]
    if ext == '.json':
        caps = json.load(open(f))
    elif ext == '.jsonl':
        caps = [json.loads(s) for s in open(f)]
    return caps

def chair_eval(generated_words_list, gt_words_list): 
    assert len(generated_words_list) == len(gt_words_list)
    cap_len = len(generated_words_list)
    hallucinated_word_count, coco_word_count, num_hallucinated_caps, num_recall_gt_objects, num_gt_objects = 0, 0, 0, 0, 0
    hallucinated_words = []
    
    for i in range(cap_len):
        generated_words = generated_words_list[i]
        gt_words = gt_words_list[i]
        hallucinated_words = [node_word for node_word in generated_words if node_word not in gt_words]
        true_words = [node_word for node_word in generated_words if node_word in gt_words]
        num_recall_gt_objects += len(set(true_words))
        num_gt_objects += len(gt_words)
        hallucinated_word_count += len(hallucinated_words)
        coco_word_count += len(generated_words)
        if len(hallucinated_words) > 0:
            num_hallucinated_caps += 1
    chair_i = (hallucinated_word_count/coco_word_count)
    chair_s = num_hallucinated_caps / cap_len
    recall = num_recall_gt_objects / num_gt_objects
    return {'chair_i': chair_i, 'chair_s': chair_s, 'recall': recall}



def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
wnl = WordNetLemmatizer()

def remove_adj_noun(caption):
    words = nltk.word_tokenize(caption.lower())
    tagged_sent = nltk.pos_tag(words)
    lemmas_sent = []
    print("caption:", caption)
    print("tagged_sent:", tagged_sent)
    
    
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        if tag[1] in ['NN', 'NNS']: 
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
    
    if len(lemmas_sent):
        return " ".join(lemmas_sent)
    return None
def pos_caption_to_words(caption): 

    '''
    Input: caption
    Output: MSCOCO words in the caption
    '''
    
    
    coco_double_words = ['motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light', 'traffic signal', 'stop light', 'fire hydrant', 'stop sign', 'parking meter', 'suit case', 'sports ball', 'baseball bat', 'baseball glove', 'tennis racket', 'wine glass', 'hot dog', 'cell phone', 'mobile phone', 'teddy bear', 'hair drier', 'potted plant', 'bow tie', 'laptop computer', 'stove top oven', 'hot dog', 'teddy bear', 'home plate', 'train track']
    
    
    
    animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal', 'cub']
    
    vehicle_words = ['jet', 'train']
    
    
    
    double_word_dict = {}
    for double_word in coco_double_words:
        double_word_dict[double_word] = double_word
    for animal_word in animal_words:
        double_word_dict['baby %s' %animal_word] = animal_word
        double_word_dict['adult %s' %animal_word] = animal_word
    for vehicle_word in vehicle_words:
        double_word_dict['passenger %s' %vehicle_word] = vehicle_word
    double_word_dict['bow tie'] = 'tie'
    double_word_dict['toilet seat'] = 'toilet'
    double_word_dict['wine glas'] = 'wine glass'
    

    
    words = nltk.word_tokenize(caption.lower())
    tagged_sent = nltk.pos_tag(words)
    lemmas_sent = []
    wnl = WordNetLemmatizer()

    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        if tag[1] in ['NN']: 
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
    
    words = lemmas_sent

    
    i = 0
    double_words = []
    idxs = []
    while i < len(words):
        idxs.append(i) 
        double_word = ' '.join(words[i:i+2])
        if double_word in double_word_dict: 
            double_words.append(double_word_dict[double_word])
            i += 2
        else:
            double_words.append(words[i])
            i += 1
    words = double_words
    return words


def predeined_set_caption_to_words(caption, object_set):

    '''
    Input: caption
    Output: MSCOCO words in the caption
    '''

    
    words = nltk.word_tokenize(caption.lower())
    tagged_sent = nltk.pos_tag(words)
    lemmas_sent = []
    wnl = WordNetLemmatizer()

    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        
        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
    
    words = lemmas_sent
    
    i = 0
    double_words = []
    idxs = []
    while i < len(words):
        idxs.append(i) 
        double_word = ' '.join(words[i:i+2])
        if double_word in object_set: 
            double_words.append(double_word)
            i += 2
        else:
            double_words.append(words[i])
            i += 1
    words = double_words
    words = [word for word in words if word in object_set]

    return words




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
    measure_tokens_dict = None, 
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

    
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    this_peer_finished = False  

    initial_model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
    objects_probs_dict = {} 
    initial_input_ids = input_ids.clone()
    
    for object_name, measure_tokens in measure_tokens_dict.items():
        measure_scores, measure_probs = [], []
        model_inputs = None 
        for m_token in measure_tokens:
            if synced_gpus:
                
                
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                
                if this_peer_finished_flag.item() == 0.0:
                    break
            if model_inputs == None:
                input_ids = initial_input_ids.clone()
                model_inputs = copy.deepcopy(initial_model_inputs) 
            else:
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

            
            
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            
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

            
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            
            next_tokens = m_token
            measure_scores.append(next_token_scores[0, next_tokens].item())
            measure_probs.append(probs[0, next_tokens].item())
            
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
        objects_probs_dict[object_name] = measure_probs.copy()
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
        return input_ids, objects_probs_dict 



def sample_batch(model, tokenizer, input_ids, image_tensor, temperature, sample_num, stop_str):
    
    
    expanded_input_ids = input_ids.repeat(sample_num, 1)
    expanded_image_tensor = image_tensor.unsqueeze(0).half().cuda().repeat(sample_num, 1, 1, 1)
    
    
    with torch.inference_mode():
        output_ids = model.generate(
            expanded_input_ids,
            images=expanded_image_tensor,
            do_sample=True if temperature > 0 else False,
            temperature=temperature if temperature > 0 else 1.0,
            max_new_tokens=512,
            use_cache=True
        )
    
    
    input_token_len = input_ids.shape[1]
    outputs_list = []
    for i in range(sample_num):
        
        n_diff = (expanded_input_ids[i] != output_ids[i, :input_token_len]).sum().item()
        if n_diff > 0:
            print(f'[Warning] Sample {i}: {n_diff} tokens differ')
        
        
        output = tokenizer.decode(output_ids[i, input_token_len:], skip_special_tokens=True).strip()
        if output.endswith(stop_str):
            output = output[:-len(stop_str)].strip()
        outputs_list.append(output)
    
    return outputs_list


import os
import glob
import argparse
import json
import time
import re
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import torch
import transformers

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.eval.utils import BertEvalCap
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.train.train import LazySupervisedDataset
from llava import conversation as conversation_lib

from PIL import Image

from llava.eval.pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

ds_collections = {
    'coco': {
        'test': './data_root/coco/coco_karpathy_test_gt.json',
    },
    'sharegpt': {
        'test': './data_root/sharegpt/sharegpt_dci_test_gt.json',
    }
}

SIMPLE_PREFIX = "ASSISTANT: "

@dataclass
class ModelArguments:
    model_path: str = field(default=None) # "./checkpoints/caption-opt-125m-2/checkpoint-16000"
    model_base: Optional[str] = field(default=None)
    force_pred: bool = field(default=False)
    # mm_projector_type: Optional[str] = field(default="linear")

@dataclass
class DataArguments:
    query: str = field(default=None)
    data_path: str = field(default="./data_root/coco/coco_karpathy_test_datalist.json")
    image_folder: str = field(default="./data_root/coco")
    is_multimodal: bool = True
    mm_use_im_start_end: bool = field(default=False)
    image_aspect_ratio: str = field(default="pad") # 'square' 'pad'
    lazy_preprocess: bool = True
    version: str = 'opt'
    batch_size: int = 32

@dataclass
class GenArguments:
    do_sample: bool = False
    num_beams: int = 3
    temperature: float = 0.2
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    max_new_tokens: int = 77 # CLIP's 77-token limit
    

def load_data_for_inference(annot_path, split='test'):
    annotations = json.load(open(annot_path))['images']
    data = {'test': [], 'val': []}

    for item in annotations:
        file_name = os.path.join(item['filepath'],item['filename'])
        image = {'file_name': file_name, 'image_id': str(item['cocoid'])}
        if item['split'] == 'test':
            data['test'].append(image)
        elif item['split'] == 'val':
            data['val'].append(image)

    return data[split]


# Custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_args, data_json, tokenizer, image_processor, model_config):
        self.data_args = data_args
        self.data_json = data_json
        self.image_folder = data_args.image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.data_json[index]
        file_name = line["file_name"]
        
        if self.data_args.query is not None:
            qs = self.data_args.query
        else:   
            if 'opt' in self.data_args.version:
                qs = "What is in the photo?"
            if 'llama_3' in self.data_args.version:
                qs = "What is in the photo?"
            elif 'v1' in self.data_args.version:
                # qs = "Provide a single-sentence description of the image, using 25~30 words" + "\n"  # Generalist
                qs = "What is in the photo?"  # Specialist
         
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + qs

        if self.data_args.version != 'plain':
            conv = conv_templates[self.data_args.version].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None) 
            prompt = conv.get_prompt()
        else:
            prompt = qs

        image = Image.open(os.path.join(self.image_folder, file_name)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, line['image_id'], file_name

    def __len__(self):
        return len(self.data_json)


# DataLoader
def create_data_loader(data_args, data_json, tokenizer, image_processor, model_config, num_workers=4):
    # assert data_args.batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(data_args, data_json, tokenizer, image_processor, model_config)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=data_args.batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def collate_fn(batch):
    input_ids, image_tensors, image_id, file_name= zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_id, file_name


def preprocess_captions(captions, tokenizer, data_path='coco'):
    preprocessed_captions = []
    for i in range(len(captions)): 
        # default preprocess
        caption = captions[i]
        caption = caption.strip()
        caption = caption.split(SIMPLE_PREFIX)[-1]
        if caption.startswith(tokenizer.bos_token):
            caption = caption[len(tokenizer.bos_token):]
        if caption.endswith(tokenizer.eos_token):
            caption = caption[:-len(tokenizer.eos_token)]
        
        # For models finetuned on DCI or ShareGPT data,
        # use only the first 3 sentences of the caption.
        # Detailed captions are compared using only the first 3 sentences.
        # Note that without this preprocessing, captioning metrics for generalist models like LLaVA may drop.
        # For example, longer predicted captions tend to lower CIDEr scores
        # and increase the frequency of hallucinations in later sentences, which negatively impacts CAPTURE performance.
        if 'dci' in data_path or 'sharegpt' in data_path:
            sentences = caption.split('.')
            sentences = [s.strip() for s in caption.split('.') if s.strip()]
            if len(sentences) > 3:
                caption = '.'.join(sentences[:3]) + '.'
            else:
                caption = '.'.join(sentences).strip() + '.'
        preprocessed_captions.append(caption)
    return preprocessed_captions

def pred_captions_exist(model_args):
    """
    Check if the prediction file already exists.
    """
    if not os.path.exists(model_args.model_path):
        result_path = os.path.join("checkpoints",model_args.model_path)
        os.makedirs(result_path, exist_ok=True)
    else:
        result_path = model_args.model_path

    matching_files = glob.glob(os.path.join(result_path, 'pred_captions_*.json')) 
    if ('liuhaotian' in result_path) or (model_args.force_pred == True): 
        matching_files = None
    
    time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
    results_file_name = os.path.join(result_path, f'pred_captions_{time_prefix}.json')

    # FIXME assume that all 'pred_captions' are very similar.
    return bool(matching_files), matching_files[0] if matching_files else results_file_name

def eval_model():
    # Model
    disable_torch_init()
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, GenArguments))
    model_args, data_args, gen_args = parser.parse_args_into_dataclasses() 

    exist_bool, results_file_name = pred_captions_exist(model_args)

    if not exist_bool:
        model_name = get_model_name_from_path(model_args.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_args.model_path, model_args.model_base, model_name, device_map='cuda' if torch.cuda.is_available() else 'cpu'
        )
        model.eval()
        for p in model.parameters(): p.requires_grad = False

        generation_kwargs = {'max_new_tokens': gen_args.max_new_tokens, 
                            'no_repeat_ngram_size': 0, 
                            'length_penalty': 0.,
                            'num_beams': gen_args.num_beams, 
                            'early_stopping': True, 
                            'eos_token_id': tokenizer.eos_token_id,
                            }

        if 'coco' in data_args.data_path:
            data_json = load_data_for_inference(data_args.data_path, split='test')
        else:
            data_json= json.load(open(data_args.data_path))
        data_loader = create_data_loader(data_args, data_json, tokenizer, image_processor, model.config)
        results = []
        
        for input_ids, image_tensors, image_ids, file_names in tqdm(data_loader, total=len(data_loader.dataset)//data_args.batch_size):
            input_ids = input_ids.to(device='cuda', non_blocking=True)
            image_tensors = image_tensors.to(dtype=torch.float16, device='cuda', non_blocking=True)
            images_embeds = None if 'crossatt' not in model.config.mm_projector_type \
                                else model.encode_images(image_tensors)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,           
                    images=image_tensors,
                    images_embeds=images_embeds,
                    **generation_kwargs
                    )

            filtered_output_ids = []
            for output_id in output_ids:
                filtered = output_id[output_id != IMAGE_TOKEN_INDEX]
                filtered_output_ids.append(filtered)
            filtered_output_ids = torch.stack(filtered_output_ids).to(output_ids.device)
            outputs = tokenizer.batch_decode(filtered_output_ids, skip_special_tokens=True)
            outputs = preprocess_captions(outputs, tokenizer, data_args.data_path)
            for output, image_id, file_name in zip(outputs, image_ids, file_names):
                results.append({'image_id': int(image_id), 'caption': output, 'file_name': file_name})
                print(f"file_name: {file_name}, Caption: {output}")
                # os.path.join(data_args.image_folder,file_name)

        with open(results_file_name, 'w') as f:
            json.dump(results, f)
    else:
        print('Prediction file already exists. Load the file: {}.'.format(results_file_name))


    # evaluation setup
    if 'coco' in data_args.data_path.lower():
        annotation_file = ds_collections['coco']['test']
    elif 'sharegpt' in data_args.data_path.lower():
        annotation_file = ds_collections['sharegpt']['test']
    elif 'dci' in data_args.data_path.lower():
        annotation_file = ds_collections['dci']['test']
    else:
        raise ValueError("Invalid dataset path: {}".format(data_args.data_path))
    result_path = os.path.dirname(results_file_name)
    time_prefix = re.search(r'pred_captions_(\d+)\.json', os.path.basename(results_file_name)).group(1)

    # coco evaluation
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file_name)
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.evaluate()

    # bert-score evaluation
    bert_eval = BertEvalCap(coco, coco_result)
    bert_eval.evaluate()

    # save results
    merged_score_dict = {**coco_eval.eval, **bert_eval.eval}
    lines = []
    for metric, score in merged_score_dict.items():
        if str(metric) == 'ROUGE_L': continue
        sc = f"{metric}: {score*100:.1f}"
        lines.append(sc)
    score_file_name = os.path.join(result_path, f'scores_{time_prefix}.txt')
    with open(score_file_name, 'w') as f:
        f.write('\n'.join(lines))
    print('Result path: ', result_path + f'/{time_prefix}.txt')

if __name__ == "__main__":
    eval_model()


"""

[Origianl generate configs]
'image_sizes' : image_sizes,
'do_sample' : True if gen_args.temperature > 0 else False,
'temperature' : gen_args.temperature,
'top_p' : gen_args.top_p,
'num_beams' : gen_args.num_beams,
'max_new_tokens' : gen_args.max_new_tokens,
'use_cache' : True

"""

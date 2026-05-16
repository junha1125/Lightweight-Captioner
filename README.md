# MM-SeR: Multimodal Self-Refinement for Lightweight Image Captioning

This repository contains the lightweight captioner training and evaluation code used in our experiments. The finetuned weights and conversation-formatted JSON files are shared via this [Google Drive link](https://drive.google.com/drive/folders/1s6U1bldf3YCkrmCbbbJ5Sx4JHINqECxu?usp=sharing).

## Scope

This code release focuses on the lightweight captioner. The self-refinement module described in the paper is outside the scope of this public code release. For readers interested in building a comparable refinement pipeline, the procedure can be implemented from the paper description, and [Visual-CoT](https://github.com/deepcs233/Visual-CoT) provides a related reference for self-refinement-style implementations.

## 📦 Installation

Requires **Python 3.8+**. Python 3.10 is recommended.

Install dependencies with:

```bash
pip install -e ".[train]"
```

> Make sure you have [PyTorch](https://pytorch.org/get-started/locally/) and [Deepspeed](https://www.deepspeed.ai/) installed properly.

## 📂 Datasets

### Pretraining

1. Download the LLaVA-Pretrain dataset from HuggingFace: [liuhaotian/LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/tree/main)

### Finetuning

1. **COCO Captioning**    
   Download [train2014](http://images.cocodataset.org/zips/train2014.zip) and [val2014](http://images.cocodataset.org/zips/val2014.zip) from [COCO official site](https://cocodataset.org/#home)

2. **ShareGPT-4V**    
   Follow instructions from [ShareGPT4V Data Guide](https://github.com/ShareGPT4Omni/ShareGPT4V/blob/master/docs/Data.md)

3. **DCI**    
   Follow instructions from [DCI Data Guide](https://github.com/facebookresearch/DCI?tab=readme-ov-file#setup).

```bash
data_root/
├── LLaVA-Pretrain/
│   ├── images/
│   └── blip_laion_cc_sbu_558k.json
├── coco/
│   ├── train2014/
│   ├── val2014/
│   ├── coco_karpathy_test_datalist.json
│   ├── coco_karpathy_test_gt.json
│   └── coco_karpathy_train_conv_1Q.json
├── sharegpt/
│   ├── coco/
│   ├── llava/
│   ├── sam/
│   ├── share_textvqa/
│   ├── web-celebrity/
│   ├── web-landmark/
│   ├── wikiart/
│   ├── sharegpt_dci_test_datalist.json
│   └── sharegpt_dci_test_gt.json
└── DCI/
    ├── images/
    └── sharegpt_dci_train_conv.json
```

## 🚀 Training

### Pretraining

Run the MLP-based pretraining script:

```bash
bash scripts/v1_5/mlp/pretrain_opt125m.sh
```

### Finetuning

Run the finetuning script for MS-COCO:

```bash
bash scripts/v1_5/mlp/finetune_opt125m.sh
```

Run the finetuning script for ShareGPT:

```bash
bash scripts/v1_5/mlp/sharegpt/finetune_opt125m.sh
```

The training scripts report to Weights & Biases by default. If needed, set `WANDB_MODE=offline` or change `--report_to wandb` in the scripts.

## 📊 Evaluation

Evaluation computes captioning metrics and BERTScore. On CPU-only machines, this step can be slow, and the first run may download the BERTScore backbone model.

### Single-sentence captioning

This evaluates LLaVA-1.5 7B and our specialist model. The finetuned weights of our model are available [here](https://drive.google.com/drive/folders/1s6U1bldf3YCkrmCbbbJ5Sx4JHINqECxu?usp=sharing).
```bash
python llava/eval/model_caption.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --query "Provide a single-sentence description of the image." \
    --data_path "./data_root/coco/coco_karpathy_test_datalist.json" \
    --image_folder "./data_root/coco" \
    --batch_size 6 \
    --version llava_v1 

python llava/eval/model_caption.py \
    --model_path "<finetuned_model_path>" \
    --data_path "./data_root/coco/coco_karpathy_test_datalist.json" \
    --image_folder "./data_root/coco" \
    --version opt 
```

### Detailed captioning
```bash
python llava/eval/model_caption.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --query "Provide a brief description of the given image.\n" \
    --data_path "./data_root/sharegpt/sharegpt_dci_test_datalist.json" \
    --image_folder "./data_root" \
    --batch_size 2 \
    --version llava_v1

python llava/eval/model_caption.py \
    --model_path "<finetuned_model_path>" \
    --data_path "./data_root/sharegpt/sharegpt_dci_test_datalist.json" \
    --image_folder "./data_root" \
    --batch_size 16 \
    --version opt
```

## 📄 License

This project is licensed under the terms of the license in the [LICENSE](./LICENSE) file.

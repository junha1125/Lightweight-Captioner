import torch


class BertEvalCap:
    def __init__(self, coco, cocoRes):
        self.eval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {"image_id": coco.getImgIds()}

    def evaluate(self):
        from llava.eval.bert_score import score

        img_ids = self.params["image_id"]
        candidates = []
        references = []

        for img_id in img_ids:
            if img_id not in self.cocoRes.imgToAnns:
                continue

            candidates.append(self.cocoRes.imgToAnns[img_id][0]["caption"])
            references.append([ann["caption"] for ann in self.coco.imgToAnns[img_id]])

        if not candidates:
            raise ValueError("No prediction captions were found for BERTScore evaluation.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _, _, f1 = score(candidates, references, lang="en", device=device, verbose=False)
        self.eval["BERTScore"] = f1.mean().item()

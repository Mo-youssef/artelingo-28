import os
import pandas as pd
from torchvision.io import read_image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset
from PIL import Image
import random
import torch
from torch.utils.data import DataLoader
import argparse
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from transformers import StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm
import pickle

import pdb

class Artelingo(Dataset):
    def __init__(self, annotations_file, img_dir, language=None, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        if language is not None:
            self.img_labels = self.img_labels[self.img_labels['language']==language]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,
                                f"{self.img_labels.iloc[idx]['art_style']}/{self.img_labels.iloc[idx]['painting']}.jpg")
        image = Image.open(img_path)
#         image = read_image(img_path)
        emotion = self.img_labels.iloc[idx]['emotion']
        language = self.img_labels.iloc[idx]['language']
        art_style = self.img_labels.iloc[idx]['art_style']
        caption = self.img_labels.iloc[idx]['caption']
        painting = self.img_labels.iloc[idx]['painting']
        if self.transform:
            image = self.transform(image)
        return dict(image=image,
                    emotion=emotion,
                    language=language,
                    art_style=art_style,
                    caption=caption,
                    painting=painting)


def get_parser():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--output-file", required=True, help="path to save outputs.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    return parser

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False
    
def get_transforms():
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    normalize = transforms.Normalize(mean, std)
    transform_pt = transforms.Compose(
        [
            transforms.Resize(
                (224, 224), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            normalize
        ]
    )
    return transform_pt
    
def main():
    img_dir = '/ibex/user/mohameys/artelingo/data/'
    split = 'test'
    annotations_file = f'/ibex/user/mohameys/artelingo/multilingual_{split}_metadata.csv'
    df = pd.read_csv(annotations_file)
    languages = df.language.unique()
    # load bad words
    bad_words_path = '/home/mohameys/minigpt4/bad_words'
    with open(os.path.join(bad_words_path, 'bad_words.pkl'), 'rb') as f:
        bad_words = pickle.load(f)

    transform_pt = get_transforms()
    print('Loading dataset...')
    # language = 'arabic'
    # language = None


    parser = get_parser()
    # args = parser.parse_args(['--cfg-path', 'eval_configs/minigpt4_eval.yaml'])
    args = parser.parse_args()
    cfg = Config(args)
    print('Done\nLoading model...')
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)

    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

#     prompt = '###Human: <Img><ImageHere></Img> Please provide a short one \
# sentence brief description of the picture. \
# Write the caption in English characters only. \
# ###Assistant: '


    stop_words_ids = [torch.tensor(model.llama_tokenizer.encode(i)).to(model.llama_model.device)
                    for i in ['###', ' ###', ' ##', '####']
                    ] 
    model.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    # dtype = model.visual_encoder.patch_embed.proj.weight.dtype
    device = model.visual_encoder.patch_embed.proj.weight.device
    print('Done\nGenerating captions...')
    captions, art_styles, paintings, emotions, batch_languages = [], [], [], [], []

    for language in languages:
        train_artelingo = Artelingo(annotations_file, img_dir, language=language, transform=transform_pt)
        print(f'Number of samples in {language} dataset: {len(train_artelingo)}')
        train_dataloader = DataLoader(train_artelingo, batch_size=128, shuffle=False)
        bad_word_ids = bad_words[language]
        for i, sample in enumerate(tqdm(train_dataloader)):
            img_embeds, atts_img = model.encode_img(sample['image'].to(torch.float).to(device))
            emotion = sample['emotion'] if 'emotion' in sample.keys() else None
            batch_language = sample['language'] if 'language' in sample.keys() else None
            prompt = random.choice(model.prompt_list)
            img_embeds, atts_img = model.prompt_wrap(img_embeds, atts_img, prompt, batch_language, emotion)
            # img_embeds, atts_img = model.prompt_wrap(img_embeds, atts_img, prompt)
            batch_size = img_embeds.shape[0]
            bos = torch.ones([batch_size, 1],
                            dtype=torch.long,
                            device=img_embeds.device) * model.llama_tokenizer.bos_token_id
            bos_embeds = model.llama_model.transformer.word_embeddings(bos)
            atts_bos = atts_img[:, :1]
            inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
            # attention_mask = torch.cat([atts_bos, atts_img], dim=1)
            outputs = model.llama_model.generate(
                inputs_embeds=inputs_embeds.to(model.llama_model.dtype),
                max_new_tokens=300,
                # min_new_tokens=32,
                # min_length=150,
                stopping_criteria=model.stopping_criteria,
                num_beams=1,
                do_sample=True,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.0,
                length_penalty=1,
                temperature=1.0,
                bad_words_ids=bad_word_ids,
            )
            output_text = model.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            captions.extend([o.split("###")[0] for o in output_text])
            art_styles.extend(sample['art_style'])
            paintings.extend(sample['painting'])
            emotions.extend(emotion)
            batch_languages.extend(batch_language)
            print([o.split("###")[0] for o in output_text])
            # pdb.set_trace()

    print('Done\nSaving captions...')
    df = pd.DataFrame({'caption': captions,
                       'art_style': art_styles, 
                       'painting': paintings,
                       'emotion': emotions,
                       'language': languages})
    # df.to_csv('artelingo23.csv', index=False)
    df.to_csv(args.output_file, index=False)

if __name__ == '__main__':
    main()
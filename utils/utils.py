import pandas as pd
import numpy as np
import emoji
import re
import datasets
import torch

from tqdm import tqdm
from opencc import OpenCC
from pandarallel import pandarallel
from transformers import default_data_collator, BertTokenizer, BertForTokenClassification


def define_labels(param_args):
    entity_type_list = param_args.entity_type
    label_names = [f"{prefix}-{entity_type}" for entity_type in entity_type_list for prefix in ["B", "I"]]
    label_names = ['O'] + label_names

    labels_to_ids = {k: v for v, k in enumerate(label_names)}
    ids_to_labels = {v: k for v, k in enumerate(label_names)}
    
    return labels_to_ids, ids_to_labels


def tag_char(df, args):
    def _tag_char(example, param_args):
        content = example['input_text'] if pd.notna(example['input_text']) else ""
        tag = ['O'] * len(content)

        for entity_type in param_args.entity_type:
            pos_list = []
            entity_list = example[f"{entity_type}_matched".lower()]
            if entity_list == []:
                continue
            else:
                for entity in entity_list:
                    try:
                        pos_list.extend([(match.start(), match.end()) for match in re.finditer(entity, content)])
                    except Exception as e:
                        print(entity, content)
                        continue
                    # try:
                    #     pos_list.extend([(match.start(), match.end()) for match in re.finditer(entity, content)])
                    # except Exception as e:
                    #     print(e)
                    #     continue
                for (start, end) in pos_list:
                    tag[start] = f"B-{entity_type}"
                    tag[start+1:end] = [f"I-{entity_type}"] * (end - start - 1)
        
        assert len(content) == len(tag)
        return tag
    
    tqdm.pandas(desc='tagging char-level label')
    df['tag_char'] = df.apply(_tag_char, param_args=args, axis=1)
    return df


def tokenize_and_align_labels(df, args):
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)

    def _tokenize_and_align_labels(example):
        tokenized_context, token_labels = [], []
        
        if pd.isna(example["input_text"]):
            example["input_text"] = ""

        for char, char_label in zip(example["input_text"], example["tag_char"]):
            tokenized_words = tokenizer.tokenize(char)
            n_subwords = len(tokenized_words)

            tokenized_context.extend(tokenized_words)
            token_labels.extend([char_label] * n_subwords)
        
        assert len(tokenized_context) == len(token_labels)

        return tokenized_context, token_labels
    
    # tqdm.pandas(desc='tokenizing and aligning labels')
    pandarallel.initialize(nb_workers=32, progress_bar=True, use_memory_fs=False)
    df[['tokenized_content', 'token_labels']] = df.parallel_apply(_tokenize_and_align_labels, axis=1).to_list()

    return df


def data_collator(features):
    import torch
    from collections.abc import Mapping
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}
    for k, v in first.items():
        if v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        elif v is not None and isinstance(v, str):
            batch[k] = [f[k] for f in features]
    
    return batch

def test_data_collator(features):
    import torch
    from collections.abc import Mapping
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}
    for k, v in first.items():
        if k != 'tokenized_content' and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        elif k != 'tokenized_content' and v is not None and isinstance(v, str):
            batch[k] = [f[k] for f in features]
        elif k == 'tokenized_content':
            batch[k] = [f[k] for f in features]
    
    return batch


def generate_train_and_dev_dataloader(arg):
    # load the data
    train_set = datasets.load_from_disk(f"{arg.prefix_path}{arg.train_data}")
    valid_set = datasets.load_from_disk(f"{arg.prefix_path}{arg.valid_data}")
    
    # define the dataloader
    # data_collator = default_data_collator
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=arg.batch_size, collate_fn=data_collator, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=arg.batch_size, collate_fn=data_collator, shuffle=False)
    
    return train_loader, valid_loader


def generate_test_dataloader(arg):
    # load the data
    test_set = datasets.load_from_disk(f"{arg.prefix_path}/{arg.test_data}")

    # define the dataloader
    # data_collator = default_data_collator
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=arg.batch_size, collate_fn=test_data_collator, shuffle=False)

    return test_loader


def tokenize_test_text(example, param_args):
    tokenizer = BertTokenizer.from_pretrained(param_args.pretrained_model)
    tokenized_content, tokenized_content_ = [], []
    headline_content = example["input_text"]
    if pd.notna(headline_content):
        for char in headline_content:
            if tokenizer.tokenize(char):
                tokenized_content.append(char)
                tokenized_content_.extend(tokenizer.tokenize(char))
    else:
        return [], [], []
    
    tokenized_content_ = [tokenizer.cls_token] + tokenized_content_ + [tokenizer.sep_token]
    max_len = param_args.max_len
    if len(tokenized_content_) > max_len: 
        tokenized_content_ = tokenized_content_[:max_len]
    else:
        tokenized_content_ = tokenized_content_ + [tokenizer.pad_token] * (max_len - len(tokenized_content_))
    
    attn_mask = [1 if tok != tokenizer.pad_token else 0 for tok in tokenized_content_]

    ids = tokenizer.convert_tokens_to_ids(tokenized_content_)

    return tokenized_content, ids, attn_mask


def convert_label_to_entity(df, args):
    def _convert_label_to_entity(example, param_args):
        output = {f"{label}_pred": [] for label in param_args.entity_type}
        tokenized_words = example['tokenized_content']
        label_pred = example["label_pred"]
        start_idx, end_idx = -1, -1
        for idx, label in enumerate(label_pred):
            if label == 'O':
                if start_idx != -1:
                    end_idx = idx
                    # print(idx, entity_type, tokenized_words[start_idx:end_idx])
                    output[f"{entity_type}_pred"].append("".join(tokenized_words[start_idx:end_idx]))
                    start_idx, end_idx = -1, -1
                continue
            elif "B" in label:
                if start_idx != -1:
                    end_idx = idx
                    # print(idx, entity_type, tokenized_words[start_idx:end_idx])
                    output[f"{entity_type}_pred"].append("".join(tokenized_words[start_idx:end_idx]))
                    start_idx = idx
                    entity_type = label.strip("B-")
                else:
                    start_idx = idx
                    entity_type = label.strip("B-")
            elif "I" in label:   
                if start_idx != -1:
                    if entity_type != label.strip("I-"):
                        end_idx = idx
                        # print(idx, entity_type, tokenized_words[start_idx:end_idx])
                        output[f"{entity_type}_pred"].append("".join(tokenized_words[start_idx:end_idx]))
                        start_idx, end_idx = -1, -1
                    else:
                        continue
                else:
                    continue
        if start_idx != -1:
            end_idx = idx + 1
            # print(example["docid"], '\n', entity_type, '\n', tokenized_words, '\n', label_pred)
            output[f"{entity_type}_pred"].append("".join(tokenized_words[start_idx:end_idx]))
        output = {k: list(set([i for i in v if len(i)>1])) for (k, v) in output.items()}
        return output
    
    tqdm.pandas(desc="converting label to entity")
    df_output = pd.DataFrame(df.apply(_convert_label_to_entity, param_args=args, axis=1).to_list())
    df = pd.concat([df, df_output], axis=1)
    return df
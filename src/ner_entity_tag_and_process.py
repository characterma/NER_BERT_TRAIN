import multiprocessing
import re, os
from pandarallel import pandarallel
import datasets
from transformers import BertTokenizer
from loguru import logger
import torch
import pandas as pd


class EntityTagAndProcess:
    """
    输入的数据： 文本内容、文本内容中存在的通过验证的实体（正确）
    输出的数据： 模型需要的格式的数据
    {
      'docid': str,
      'ids': torch.tensor(ids, dtype=torch.long),
      'masks': torch.tensor(attn_mask, dtype=torch.long),
      'labels': torch.tensor(label_ids, dtype=torch.long)
    } 
    1、对文本内容进行字符切分并对应标签列表
    2、过滤超出最大长度限制的数据
    3、判断tokennizer对应的实体与原始的实体是否一致, 过滤不一致的数据
    4、过滤英文匹配一半的情况（暂未使用）
    """
    def __init__(self, 
                 pretrained_model_name, 
                 max_len,
                 entity_type_list,
                 labels_to_ids,
                 tokenized_content="tokenized_content",
                 token_labels="token_labels",
                 docid="docid"
        ):
        self.pretrained_model_name = pretrained_model_name
        self.max_len = max_len
        self.entity_type_list = entity_type_list
        self.tokenized_content = tokenized_content
        self.token_labels=token_labels
        self.docid = docid
        self.labels_to_ids = labels_to_ids
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_name)
        pandarallel.initialize(nb_workers=8 if multiprocessing.cpu_count() > 8 else multiprocessing.cpu_count()-1, progress_bar=False, use_memory_fs=False)

    def tokenize_and_align_labels(self, df:pd.DataFrame):
        """
        对文本内容进行字符切分并对应标签列表
        """

        def _tokenize_and_align_labels(example):
            tokenized_context, token_labels = [], []

            if pd.isna(example["input_text"]):
                example["input_text"] = ""

            for char, char_label in zip(str(example["input_text"]), example["tag_char"]):
                tokenized_words = self.tokenizer.tokenize(char)
                n_subwords = len(tokenized_words)

                tokenized_context.extend(tokenized_words)
                token_labels.extend([char_label] * n_subwords)

            assert len(tokenized_context) == len(token_labels)

            return tokenized_context, token_labels

        # tqdm.pandas(desc='tokenizing and aligning labels')
        df[[self.tokenized_content, self.token_labels]] = df.parallel_apply(_tokenize_and_align_labels, axis=1).to_list()
        return df

    def filter_exceed_length_limit_data(self, df:pd.DataFrame):
        """
        过滤超出最大长度限制的数据
        """
        df['len_tokens'] = df['tokenized_content'].apply(lambda x: len(x))
        df = df[df['len_tokens'] < self.max_len-2]
        df = df.reset_index(drop=True)
        return df
    
    @staticmethod
    def convert_labels_to_entity(entity_type_list, tokenized_text, label_pred):
        """
        # 根据标签列表，将对应的字符转化成实体
        """
        output = {label: [] for label in entity_type_list}
        tokenized_words = tokenized_text
        label_pred = label_pred
        start_idx, end_idx = -1, -1
        for idx, label in enumerate(label_pred):
            if label == 'O':
                if start_idx != -1:
                    end_idx = idx
                    # print(idx, entity_type, tokenized_words[start_idx:end_idx])
                    output[entity_type].append("".join(tokenized_words[start_idx:end_idx]))
                    start_idx, end_idx = -1, -1
                continue
            elif "B" in label:
                if start_idx != -1:
                    end_idx = idx
                    # print(idx, entity_type, tokenized_words[start_idx:end_idx])
                    output[entity_type].append("".join(tokenized_words[start_idx:end_idx]))
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
                        output[entity_type].append("".join(tokenized_words[start_idx:end_idx]))
                        start_idx, end_idx = -1, -1
                    else:
                        continue
                else:
                    continue
        if start_idx != -1:
            end_idx = idx + 1
            output[entity_type].append("".join(tokenized_words[start_idx:end_idx]))
        output = {k: list(set([i for i in v if len(i)>1])) for (k, v) in output.items()}
        return output
    
    def entity_judgement(self, row, entity_name_list):
        """
        # 判断tokennizer对应的实体与原始的实体是否一致, 过滤不一致的数据
        """
        flag = True
        entity_name_list = [str(entity).lower() for entity in entity_name_list]
        entity_dict = EntityTagAndProcess.convert_labels_to_entity(entity_name_list, row[self.tokenized_content], row[self.token_labels])
        for entity_type in self.entity_type_list: 
            # print(entity_type)
            entity_list = eval(str(row.get(f"eval_{entity_type}")))
            gt_pred_list = list(set(entity_dict.get(entity_type)).difference(set(entity_list)))
            re_gt_pred_list = list(set(entity_list).difference(set(entity_dict.get(entity_type))))
            # print(f'{entity_type}_gt: {entity_list}')
            # print(f'{entity_type}_tokenzer: {entity_dict.get(entity_type)}')
            # print(f'gt_pred_list_tokenzer: {gt_pred_list}')
            # print(f're_gt_pred_list_tokenzer: {re_gt_pred_list}')
            if len(gt_pred_list) == 0 and len(re_gt_pred_list) == 0 :
                continue
            else:
                flag = False
                return flag
        return flag
    

    def eng_label_qc(self):
        """
        # 过滤英文匹配一半的情况
        """
        def is_alphabet(char):
            pattern = r'^[a-zA-Z]$'
            return bool(re.match(pattern, char))
        flag = 0
        for idx, label in enumerate(token_labels):
            if 'B' in label:
                if index > 0 and is_alphabet(tokenized_content[idx-1]) and is_alphabet(tokenized_content[idx]):
                    # print(f'{tokenized_content[idx-1]}{tokenized_content[idx]}{tokenized_content[idx+1]}')
                    flag = 1
                    return flag
            if "I" in label and idx<len(token_labels)-1 and tokenized_content[idx+1] == "O" and is_alphabet(tokenized_content[idx]):
                if is_alphabet(token_labels[idx+1]):
                    flag = 1
                    print(f'{tokenized_content[idx-1]}{tokenized_content[idx]}{tokenized_content[idx+1]}')
                    return flag
        return flag
    
    def add_padding_and_mask(self, example):
        tokenized_context = [self.tokenizer.cls_token] + example[self.tokenized_content] + [self.tokenizer.sep_token]
        labels = example[self.token_labels]
        labels.insert(0, 'O')
        labels.insert(len(labels), "O")

        if len(tokenized_context) > self.max_len: 
            tokenized_context = tokenized_context[:self.max_len]
            labels = labels[:self.max_len]
        else:
            tokenized_context = tokenized_context + [self.tokenizer.pad_token] * (self.max_len - len(tokenized_context))
            labels = labels + ['O'] * (self.max_len - len(labels))

        attn_mask = [1 if tok != self.tokenizer.pad_token else 0 for tok in tokenized_context]

        ids = self.tokenizer.convert_tokens_to_ids(tokenized_context)

        label_ids = [self.labels_to_ids[label] for label in labels]

        return {
              'ids': torch.tensor(ids, dtype=torch.long),
              'masks': torch.tensor(attn_mask, dtype=torch.long),
              'labels': torch.tensor(label_ids, dtype=torch.long),
              # self.tokenized_content: example[self.tokenized_content],
              # self.token_labels: example[self.token_labels]
            } 
    
    def tokenize_test_text(self, example):
        tokenized_content, tokenized_content_ = [], []
        headline_content = str(example["input_text"])
        if pd.notna(headline_content):
            for char in headline_content:
                if self.tokenizer.tokenize(char):
                    tokenized_content.append(char)
                    tokenized_content_.extend(self.tokenizer.tokenize(char))
        else:
            return [], [], []

        tokenized_content_ = [self.tokenizer.cls_token] + tokenized_content_ + [self.tokenizer.sep_token]
        max_len = self.max_len
        if len(tokenized_content_) > max_len: 
            tokenized_content_ = tokenized_content_[:max_len]
        else:
            tokenized_content_ = tokenized_content_ + [self.tokenizer.pad_token] * (max_len - len(tokenized_content_))

        attn_mask = [1 if tok != self.tokenizer.pad_token else 0 for tok in tokenized_content_]

        ids = self.tokenizer.convert_tokens_to_ids(tokenized_content_)

        return tokenized_content, ids, attn_mask

    
    def process(self, df, save_dir_path, file_name):
        logger.info(f"开始{file_name}的处理:{df.shape}")
        os.makedirs(save_dir_path, exist_ok=True)
        df = self.tokenize_and_align_labels(df)
        logger.info(f"字符分割、重名名完成:{df.shape}")
        df = self.filter_exceed_length_limit_data(df)
        logger.info(f"超过最大长度的数据过滤完成:{df.shape}")
        df['entity_is_same'] = df.apply(lambda row: self.entity_judgement(
            row=row,
            entity_name_list=self.entity_type_list
        ), axis=1)
        logger.info(f"标签对应实体是否和真实实体相同判断完成, 保存{file_name}: {df.shape}, 实体相同数据量:{df['entity_is_same'].value_counts()}")
        print(df.head().to_dict("records"))
        # single_df = df[df['docid']=='23dd347cc1aa99ce0bfaddfc3859560f_1']
        # example = single_df[[self.tokenized_content, self.token_labels]].to_dict('records')
        # logger.info(f"验证结果{example}")
        
        # 测试集处理方法和训练验证的处理方法略有差异
        if "test" in file_name:
            df.to_pickle(os.path.join(save_dir_path, f"{file_name}.pkl"))
            df[["tokenized_content", "ids", "masks"]] = df.parallel_apply(self.tokenize_test_text, axis=1).to_list()
            data = datasets.Dataset.from_pandas(df[[self.docid, self.tokenized_content, "ids", "masks"]])
        else:
            df.to_pickle(os.path.join(save_dir_path, f"{file_name}.pkl"))
            data = datasets.Dataset.from_pandas(df[[self.docid, self.tokenized_content, self.token_labels]])
            data = data.map(self.add_padding_and_mask, remove_columns=[self.tokenized_content, self.token_labels])
        data.save_to_disk(os.path.join(save_dir_path, file_name))
        logger.info("生成模型输入格式的数据")
        return data
        
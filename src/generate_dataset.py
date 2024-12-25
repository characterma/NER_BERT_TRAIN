from sklearn.model_selection import train_test_split
from collections import defaultdict
import warnings
import numpy as np
import pandas as pd
from loguru import logger 
import os
import re
from tqdm import tqdm
import json
from src.ner_entity_tag_and_process import EntityTagAndProcess
warnings.filterwarnings("ignore")
from src.utils import log_step
import copy


def greedy_match_ner(text, entities):
    """
    使用贪婪匹配法进行命名实体识别标注
    
    Args:
        text (str): 输入文本
        entities (list): 实体列表,每个元素为(entity, start, end, label)
        
    Returns:
        list: 标注结果,每个元素为(start, end, label)
    """
    # 根据实体长度进行降序排序
    entities = sorted(entities, key=lambda x: len(x[0]), reverse=True)
    
    # 存储最终标注结果
    annotations = []
    
    # 遍历每个实体
    for entity, start, end, label in entities:
        # 检查当前实体是否与已标注的实体存在交叉
        overlap = False
        for anno_start, anno_end, _ , name in annotations:
            if max(start, anno_start) <= min(end, anno_end):
                overlap = True
                break
        
        if not overlap:
            # 如果没有交叉,直接标注当前实体
            annotations.append((start, end, label, entity))
            
    # 对标注结果按起始位置进行排序
    annotations.sort(key=lambda x: x[0])
    
    return annotations

def tag_char(df, entity_types):
    def _tag_char(example, entity_types):
        content = example['input_text'] if pd.notna(example['input_text']) else ""
        tag = ['O'] * len(str(content))
        pos_list = []
        for entity_type in entity_types:
            if pd.isna(example[f"eval_{entity_type}"]):
                # print(example)
                return ""
            entity_list = eval(str(example[f"eval_{entity_type}"]))
            if entity_list == []:
                continue
            else:
                for entity in entity_list:
                    # if not entity:
                    #     continue
                    try:
                        pos_list.extend([(entity, match.start(), match.end(), entity_type) for match in re.finditer(entity, content)])
                    except Exception as e:
                        logger.error(e)
                        continue
                # extractor = get_keyword_extractor(dictionary=entity_list, use_fixed=True)
                # extractor_result = extractor(content)
                # for info in extractor_result:
                #     pos_list.append((info[0], info[1], info[2], entity_type))

        annotations = greedy_match_ner(content, pos_list)
        for (start, end, label, entity_name) in annotations:
            # print((start, end, label))
            # if start == end:
            #     # print(pos_list)
            #     print(example['docid'])
            #     print((start, end, label, entity_name))
            tag[start] = f"B-{label}"
            tag[start+1:end] = [f"I-{label}"] * (end - start - 1)
            
        if len(str(content)) == len(tag):
            return tag
        else:
            # print(f"content:{content}")
            # print(f"tag:{tag}")
            return ""
    
    tqdm.pandas(desc='tagging char-level label')
    df['tag_char'] = df.progress_apply(_tag_char, entity_types=entity_types, axis=1)
    return df

def define_labels(entity_types, position_labels=["B", "I"]):
    label_names = [f"{prefix}-{entity_type}" for entity_type in entity_types for prefix in position_labels]
    label_names = ['O'] + label_names

    labels_to_ids = {k: v for v, k in enumerate(label_names)}
    ids_to_labels = {v: k for v, k in enumerate(label_names)}
    
    return labels_to_ids, ids_to_labels

@log_step(15)
def generate_model_used_dataset(config_reader):
    """
    对两次验证后的实体以及文本内容进行处理生成NER model需要的数据格式
    """
    entity_types = config_reader.get_value("entity_type", [])
    output_dir_path = f"{config_reader.get_value('output_dir_path')}/{config_reader.get_value('industry')}_{config_reader.get_value('date')}/"
    file_name = config_reader.get_value('sentence_include_entity_eval')['second_entity_eval_result_file']
    entities_is_null_number = config_reader.get_value("generate_dataset")['entities_is_null_number']
    os.makedirs(output_dir_path, exist_ok=True)
    second_eval_result_file_path = os.path.join(output_dir_path, file_name)

    if entity_types == []:
        logger.error(f"entity_types is null in config file, please check it.{entity_types}")
        raise Exception("entity_types is null in config file, please check it")
        
    labels_to_ids, ids_to_labels = define_labels(entity_types, position_labels=["B", "I"])
    logger.info(f"generate label to index mapping:{labels_to_ids}, {ids_to_labels}")
    json.dump(labels_to_ids, open(os.path.join(output_dir_path, "labels_to_ids.json"), 'w', encoding='utf-8'), ensure_ascii=False)
    json.dump(ids_to_labels, open(os.path.join(output_dir_path, "ids_to_labels.json"), 'w', encoding='utf-8'), ensure_ascii=False)
    
    prepare_df = pd.read_excel(second_eval_result_file_path)
    prepare_df['sign'] = prepare_df.apply(lambda row: row['processed_entities'] == '[]', axis=1)
    prepare_df = pd.concat([prepare_df[prepare_df['sign']==False], prepare_df[prepare_df['sign']==True][:entities_is_null_number]])
    logger.info(f"read second entity eval result file: {second_eval_result_file_path}, data shape: {prepare_df.shape}, whether processed entities is nan count: {prepare_df['sign'].value_counts()}")

    df_train_and_eval, df_test = train_test_split(prepare_df, test_size=0.2, train_size=0.8, random_state=1)
    df_train, df_valid = train_test_split(df_train_and_eval, test_size=0.2, train_size=0.8, random_state=1)
    logger.info(f"split dataset complete: trainset: {df_train.shape}, validset:{df_valid.shape}, testset: {df_test.shape}")
    
    # 通过随机数阈值进行添加提示语和不添加提示语的数据选择
    tqdm.pandas()
    threshold = config_reader.get_value("generate_dataset")["threshold"]
    df_train["input_text"] = df_train.progress_apply(lambda x: x["context_keywords"] if np.random.random()>threshold else x["context"], axis = 1)
    df_valid["input_text"] = df_valid.progress_apply(lambda x: x["context_keywords"] if np.random.random()>threshold else x["context"], axis = 1)
    df_test_add_keyword = copy.copy(df_test)
    df_test["input_text"] = df_test["context"]
    df_test_add_keyword['input_text'] = df_test_add_keyword['context_keywords']
    
    df_train = tag_char(df_train, entity_types)
    df_train.drop(df_train[df_train['tag_char'] == ""].index, inplace=True)
    
    df_valid = tag_char(df_valid, entity_types)
    df_valid.drop(df_valid[df_valid['tag_char'] == ""].index, inplace=True)
    
    df_test = tag_char(df_test, entity_types)
    df_test.drop(df_test[df_test['tag_char'] == ""].index, inplace=True)
    
    df_test_add_keyword = tag_char(df_test_add_keyword, entity_types)
    df_test_add_keyword.drop(df_test_add_keyword[df_test_add_keyword['tag_char'] == ""].index, inplace=True)
    
    logger.info(f"df_train: {df_train.shape}, df_valid: {df_valid.shape}, df_test: {df_test.shape}, df_test_add_keywords: {df_test_add_keyword.shape}")
    
    # logger.info("验证第一步打标是否成功")
    # single_df = df_train[df_train['docid']=='23dd347cc1aa99ce0bfaddfc3859560f_1']
    # tokenize_content = [i for i in single_df['input_text'].tolist()[0]]
    # label_char = single_df['tag_char'].tolist()[0]
    # assert tokenize_content.__len__() == label_char.__len__()
    # logger.info(str(list(zip(tokenize_content, label_char))))

    pretrained_model_name = config_reader.get_value("model")["pretrained_model_name"]
    max_len = config_reader.get_value("model")['max_len']
    entity_type_list = entity_types
    tokenized_content="tokenized_content"
    token_labels="token_labels"
    docid="docid"
    tag_and_process_tool = EntityTagAndProcess(
        pretrained_model_name=pretrained_model_name,
        max_len=max_len,
        entity_type_list=entity_type_list,
        labels_to_ids=labels_to_ids,
        tokenized_content=tokenized_content,
        token_labels=token_labels,
        docid=docid,
    )
    data_train_new = tag_and_process_tool.process(df_train, output_dir_path, config_reader.get_value("generate_dataset")['train_data'])
    data_valid_new = tag_and_process_tool.process(df_valid, output_dir_path, config_reader.get_value("generate_dataset")['valid_data'])
    data_test_new = tag_and_process_tool.process(df_test, output_dir_path, config_reader.get_value("generate_dataset")['test_data'])
    data_test_add_keyword_new = tag_and_process_tool.process(df_test_add_keyword, output_dir_path, config_reader.get_value("generate_dataset")['test_add_keyword_data'])

    logger.info("model dataset save complete")
    
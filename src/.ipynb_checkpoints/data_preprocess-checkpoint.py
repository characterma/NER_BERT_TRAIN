# -*- coding: utf-8 -*-
__author__ = "Ma Shunda"

import emoji
import re
import random
import pandas as pd 
import numpy as np
from pandarallel import pandarallel
from zhconv import convert
from tqdm.auto import tqdm
from loguru import logger
import sys
import os
import json
import traceback
import multiprocessing
from src.utils.keyword_matching import KeyWordExtractor
from src.utils import log_step


def read_data_from_pickle(filepath):
    with open(filepath, "rb") as infile:
        return pickle.load(infile)


def read_data_from_excel(filepath):
    df_data = pd.read_excel(filepath)
    docid_name = next(
        (
            col
            for col in ["DOC_ID", "DOCID", "doc_id", "docid"]
            if col in df_data.columns
        ),
        None,
    )
    if docid_name is not None:
        df_data[docid_name] = df_data[docid_name].apply(str)
    return df_data


# def read_data_from_mysql(config_path):
#     config = load_yaml(config_path)
#     data_loader = MySqlDataLoader(**config)
#     df_output = data_loader.get_data()
#     data_loader.close()
#     return df_output


# def read_data_from_mongodb(config_path):
#     config = load_yaml(config_path)
#     data_loader = MongoDataLoader(**config)
#     df_output = data_loader.get_data()
#     data_loader.close()
#     return df_output


FORMAT_TO_LOADER = {
    # "mongodb": read_data_from_mongodb,
    # "mysql": read_data_from_mysql,
    "pickle": read_data_from_pickle,
    "excel": read_data_from_excel,
    "json": pd.read_json,
    "csv": pd.read_csv,
}


def get_dataset(format, file_path, **kwargs):
    if format not in FORMAT_TO_LOADER:
        raise NotImplementedError
    loader = FORMAT_TO_LOADER[format]
    logger.info(f"***** Loading data from {file_path}*****")
    data = loader(file_path)
    logger.info(f"len of data = {len(data)}")
    return data

def get_dataset_by_filename(filename):
    format_type = str(filename).split('.')[-1]
    if format_type == 'pkl':
        format_type = 'pickle'
    elif format_type == 'xlsx':
        format_type = 'excel'
    else:
        pass
    return get_dataset(format_type, filename)


## 清理文章内容
def clean_data(text):
    text = emoji.replace_emoji(text, replace="")
    text = convert(text, "zh-cn")
    text = ''.join([x for x in text if x.isprintable()])
    text = text.lower()
    return text

def get_entity2entity_type_from_vkg_entity_data(file_path, file_format):
    """
    从这批数据构建的VKG数据中获取到所有的实体信息，保存为{实体: 实体类型}
    """
    entity_and_type_df = get_dataset(format=file_format, file_path=file_path)
    cols = entity_and_type_df.columns
    entity2type = {}
    if "entity" in cols and "entity_type" in cols:
        entity2type = {row['entity']: row['entity_type'] for index, row in entity_and_type_df.iterrows()}
        return entity2type
    else:
        raise Exception(f"entity and entity type must be in columns: {entity_and_type_df.columns}")

def convert_entities_to_json_and_merge_entities_info(entities, entity2type):
    """
    将entities列转化成json格式并将实体信息合并成list类型
    """
    entities_json = {}
    if isinstance(entities, str):
        entities_str = convert(entities, "zh-cn").lower()
        try:
            entities_json = eval(entities_str)
        except Exception as e:
            logger.error(traceback.print_exception())
    elif isinstance(entities, dict):
        entities_json = entities
    else:
        logger.error(f"entities type is not str or dict, entities: {entities}")
        
    all_entities = []
    
    
    if "input_entity_info_confirm" in entities_json:
        for dic in entities_json['input_entity_info_confirm']:
            if dic['entity'] in entity2type:
                all_entities.append({"entity": dic['entity'], "entity_type": entity2type[dic['entity']]})
                
    if "newly_extracted_entity_info" in entities_json:
        for dic in entities_json['newly_extracted_entity_info']:
            if dic['entity'] in entity2type:
                all_entities.append({"entity": dic['entity'], "entity_type": entity2type[dic['entity']]})
    
    # all_entities.extend(for dic in entities_json['input_entity_info_confirm'] if "input_entity_info_confirm" in entities_json else [])
    # all_entities.extend(entities_json['newly_extracted_entity_info'] if "newly_extracted_entity_info" in entities_json else [])
    return all_entities

@log_step(15)
def data_preprocess(config_reader):
    file_format = config_reader.get_value("data", "excel")['format'] 
    raw_data_path = config_reader.get_value("data")['raw_data_path']
    vkg_entity_data_path = config_reader.get_value("data")['vkg_entity_data_path']
    
    if not os.path.exists(raw_data_path):
        raise Exception(f"{raw_data_path} is not exists")
    
    if not os.path.exists(vkg_entity_data_path):
        raise Exception(f"{vkg_entity_data_path} is not exists")
    
    raw_data = get_dataset(format=file_format, file_path=raw_data_path)
    entity2type = get_entity2entity_type_from_vkg_entity_data(file_path=vkg_entity_data_path, file_format=file_format)

    raw_cols = raw_data.columns.tolist()
    if "headline_content" in raw_data.columns.tolist():
        pass
    elif "headline" in raw_cols and "content" in raw_cols:
        raw_data['headline_content'] = raw_data.apply(lambda row:row['content'] if isinstance(row['headline'], str) and isinstance(row['content'], str) and row['headline'] in row['content'] else f"{str(row['headline'])}。{str(row['content'])}", axis=1)
    else:
        raise Exception(f"miss headline or content columns, raw data columns: {raw_cols}")
        
    
    pandarallel.initialize(nb_workers=8 if multiprocessing.cpu_count() > 8 else multiprocessing.cpu_count()-1, progress_bar=False, use_memory_fs=False)
    raw_data["cleaned_headline_content"] = raw_data.parallel_apply(lambda x: clean_data(x["headline_content"]) if isinstance(x["headline_content"], str) else None, axis=1)
    filter_na_df = raw_data[(~raw_data['cleaned_headline_content'].isna()) & (~raw_data['entities'].isna())]
    logger.info(f"filter before: {raw_data.shape}, filter after: {filter_na_df.shape}")
    filter_na_df["processed_entities"] = filter_na_df.parallel_apply(lambda x: convert_entities_to_json_and_merge_entities_info(x['entities'], entity2type), axis=1)
    output_dir_path = f"{config_reader.get_value('output_dir_path')}/{config_reader.get_value('industry')}_{config_reader.get_value('date')}/"
    os.makedirs(output_dir_path, exist_ok=True)
    file_name = config_reader.get_value('data_preprocess')['processed_file_name']
    processed_file_path = os.path.join(output_dir_path, file_name)
    filter_na_df.to_excel(processed_file_path, index=False)
    return filter_na_df
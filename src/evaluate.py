import json
import re, os
import pandas as pd
from tqdm import tqdm
from src.train_model import inference
from loguru import logger
import torch
from transformers import default_data_collator, BertTokenizer, BertForTokenClassification
from src.utils.muc_sc import evaluate_all
from src.ner_entity_tag_and_process import EntityTagAndProcess


# def run_eval4ner(df, entity_type):
#     logger.info(f"Number of data for evaluation:{len(df)}")
#     pred = [[(f"{entity_type}", j.lower()) for j in i] for i in df[f'{entity_type}_pred'].to_list()]
#     label_gt = [[(f"{entity_type}", j.lower()) for j in i] for i in df[f'eval_{entity_type}'.lower()].to_list()]
#     logger.info(f"======= {entity_type} =======")
#     # logger.info(f"Example:\ncontent:{df['input_text'].tolist()[0]},\npred: {pred[0]}\nlabel: {label_gt[0]}")
#     evaluate_all(pred, label_gt, texts=df['input_text'].to_list())

def run_eval4ner(df, entity_type):
    preds, label_gts, texts = [], [], []
    for text, pred, label_gt in df[['input_text', f'{entity_type}_pred', f'eval_{entity_type}']].values:
        if isinstance(text, str) and isinstance(pred, str) and isinstance(label_gt, str):
            pred = eval(pred)
            label_gt = eval(label_gt)
        elif isinstance(text, str) and isinstance(pred, list) and isinstance(label_gt, list):
            pass
        else:
            logger.error("pred and label_gt must be list")
            continue
        preds.append([(entity_type, wd.lower()) for wd in pred])
        label_gts.append([(entity_type, wd.lower()) for wd in label_gt])
        texts.append(text)
    logger.info(f"Example:\ncontent:{texts[0]},\npred: {preds[0]}\nlabel: {label_gts[0]}")
    logger.info(f"texts: {len(texts)}, preds: {preds.__len__()}, label_gts: {label_gts.__len__()}")
    evaluate_all(preds, label_gts, texts=texts)


def load_model(pretrained_model_name, num_labels, save_model_path):
    model = BertForTokenClassification.from_pretrained(pretrained_model_name, num_labels=num_labels)
    model.load_state_dict(torch.load(save_model_path))
    return model


def entity_col_type_transform(data, entity_types):
    def transform(entities):
        process_entities = []
        if isinstance(entities, str):
            process_entities = eval(entities)
        elif isinstance(entities, list):
            process_entities = entities
        else:
            pass
        return process_entities
        
    for entity_type in entity_types:
        data[f'eval_{entity_type}'] = data[f'eval_{entity_type}'].apply(transform)
    logger.info(f"test dataset: {data.shape}, {data.columns}"
                f"{data.head(1).to_dict('records')}"
               )
    return data

def evaluate_by_type(data, data_predict_result, entity_types, max_len, save_file_path):
    df_test_all = pd.merge(data, data_predict_result, on='doc_id', how='left')
    df_test_all['input_len'] = df_test_all["input_text"].str.len()
    df_test_all["sign"] = df_test_all['input_len'].map(lambda input_len: input_len < max_len)
    cols_name = df_test_all.columns
    for entity_type in entity_types:
        if f"eval_{entity_type}" in cols_name and f"{entity_type}_pred" in cols_name:            
            df_test_all[f'{entity_type}_real_diff'] = df_test_all.apply(lambda row: list(set(row[f"eval_{entity_type}"]) - set(row[f"{entity_type}_pred"])), axis=1)
            df_test_all[f'{entity_type}_real_diff_length'] = df_test_all[f'{entity_type}_real_diff'].map(len)
            df_test_all[f'{entity_type}_pred_diff'] = df_test_all.apply(lambda row: list(set(row[f"{entity_type}_pred"]) - set(row[f"eval_{entity_type}"])), axis=1)
            df_test_all[f'{entity_type}_pred_diff_length'] = df_test_all[f'{entity_type}_pred_diff'].map(len)
    df_test_all.to_excel(save_file_path, index=False)
    logger.info(f"test dataset columns: {df_test_all.columns}")
    logger.info(f"input text length count: {df_test_all['sign'].value_counts()}")
    for entity_type in entity_types:
        run_eval4ner(df_test_all[df_test_all['sign']==True], entity_type)
    

def model_evalate(config_reader):
    output_dir_path = f"{config_reader.get_value('output_dir_path')}/{config_reader.get_value('industry')}_{config_reader.get_value('date')}"
    test_dataset_file_path = f"{output_dir_path}/{config_reader.get_value('generate_dataset')['test_data']}.pkl"
    testset_dir_path = f"{output_dir_path}/{config_reader.get_value('generate_dataset')['test_data']}"

    test_add_keyword_dataset_file_path = f"{output_dir_path}/{config_reader.get_value('generate_dataset')['test_add_keyword_data']}.pkl"
    testset_add_keyword_dir_path = f"{output_dir_path}/{config_reader.get_value('generate_dataset')['test_add_keyword_data']}"
    
    test_add_keyword_data_predict_file_path = f"{output_dir_path}/{config_reader.get_value('model_evaluate')['test_add_keyword_data_predict_file']}"
    test_no_add_keyword_data_predict_file_path = f"{output_dir_path}/{config_reader.get_value('model_evaluate')['test_no_add_keyword_data_predict_file']}"
    logger.info(f"test_add_keyword_data_predict_file_path: {test_add_keyword_data_predict_file_path}\ntest_no_add_keyword_data_predict_file_path:{test_no_add_keyword_data_predict_file_path}")

    device_name = config_reader.get_value("model")['device']
    entity_types = config_reader.get_value("entity_type")
    
    
    pretrained_model_name = config_reader.get_value("model")['pretrained_model_name']
    batch_size = config_reader.get_value("model")['batch_size']  
    max_len = config_reader.get_value("model")['max_len'] 
    device = config_reader.get_value("model")['device'] 
    ids_to_labels = json.load(open(os.path.join(output_dir_path, "ids_to_labels.json"), 'r'))
    labels_to_ids = json.load(open(os.path.join(output_dir_path, "labels_to_ids.json"), 'r'))
    ids_to_labels = {int(k): v for k, v in ids_to_labels.items()}
    num_labels = len(labels_to_ids)
    logger.info(f"num_labels: {num_labels}, labels_to_ids: {labels_to_ids}, ids_to_labels:{ids_to_labels}")
    
    
    test_dataset = pd.read_pickle(test_dataset_file_path)
    test_dataset = entity_col_type_transform(test_dataset, entity_types)
    
    save_model_dir = f"{output_dir_path}/{config_reader.get_value('train_model')['save_model_dir']}" 
    save_model_path = f"{save_model_dir}/{config_reader.get_value('train_model')['model_name']}"
    
        
    if os.path.exists(test_no_add_keyword_data_predict_file_path):
        test_data_predict_result = pd.read_excel(test_no_add_keyword_data_predict_file_path)
    else:
        if os.path.exists(save_model_path):
            model = load_model(
                pretrained_model_name=pretrained_model_name, 
                num_labels=num_labels, 
                save_model_path=save_model_path
            )
            model.to(device)
            tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

        else:
            raise Exception(f"model is not exists: {save_model_path}")
        logger.info("begin model predict")
        test_data_predict_result = inference(
                model=model, 
                device=device,
                tokenizer=tokenizer, 
                ids_to_labels=ids_to_labels, 
                test_dir_path=testset_dir_path, 
                batch_size=batch_size,
                entity_types=entity_types
            )
    
    logger.info(f"{'=' * 15} no add keyword evaluate {'=' * 15}")
    evaluate_by_type(
        data=test_dataset, 
        data_predict_result=test_data_predict_result, 
        entity_types=entity_types,
        max_len=max_len,
        save_file_path=test_no_add_keyword_data_predict_file_path
    )
    


def model_evalate_add_keyword_and_no_add_keyword(config_reader):
    output_dir_path = f"{config_reader.get_value('output_dir_path')}/{config_reader.get_value('industry')}_{config_reader.get_value('date')}"
    test_dataset_file_path = f"{output_dir_path}/{config_reader.get_value('generate_dataset')['test_data']}.pkl"
    testset_dir_path = f"{output_dir_path}/{config_reader.get_value('generate_dataset')['test_data']}"

    test_add_keyword_dataset_file_path = f"{output_dir_path}/{config_reader.get_value('generate_dataset')['test_add_keyword_data']}.pkl"
    testset_add_keyword_dir_path = f"{output_dir_path}/{config_reader.get_value('generate_dataset')['test_add_keyword_data']}"
    
    test_add_keyword_data_predict_file_path = f"{output_dir_path}/{config_reader.get_value('model_evaluate')['test_add_keyword_data_predict_file']}"
    test_no_add_keyword_data_predict_file_path = f"{output_dir_path}/{config_reader.get_value('model_evaluate')['test_no_add_keyword_data_predict_file']}"
    logger.info(f"test_add_keyword_data_predict_file_path: {test_add_keyword_data_predict_file_path}\ntest_no_add_keyword_data_predict_file_path:{test_no_add_keyword_data_predict_file_path}")

    device_name = config_reader.get_value("model")['device']
    entity_types = config_reader.get_value("entity_type")
    
    
    pretrained_model_name = config_reader.get_value("model")['pretrained_model_name']
    batch_size = config_reader.get_value("model")['batch_size']  
    max_len = config_reader.get_value("model")['max_len'] 
    device = config_reader.get_value("model")['device'] 
    ids_to_labels = json.load(open(os.path.join(output_dir_path, "ids_to_labels.json"), 'r'))
    labels_to_ids = json.load(open(os.path.join(output_dir_path, "labels_to_ids.json"), 'r'))
    ids_to_labels = {int(k): v for k, v in ids_to_labels.items()}
    num_labels = len(labels_to_ids)
    logger.info(f"num_labels: {num_labels}, labels_to_ids: {labels_to_ids}, ids_to_labels:{ids_to_labels}")
    
    # logger.info(f"test_dataset: {test_dataset_file_path},test_dataset_add_keyword: {test_add_keyword_dataset_file_path}")
    test_dataset = pd.read_pickle(test_dataset_file_path)
    test_dataset_add_keyword = pd.read_pickle(test_add_keyword_dataset_file_path)
    logger.info(f"test dataset: {test_dataset.shape}, test_dataset_add_keyword: {test_dataset_add_keyword.shape}")
    # raise Exception("stop model eval")
    test_dataset = entity_col_type_transform(test_dataset, entity_types)
    test_dataset_add_keyword = entity_col_type_transform(test_dataset_add_keyword, entity_types)
    
    save_model_dir = f"{output_dir_path}/{config_reader.get_value('train_model')['save_model_dir']}" 
    save_model_path = f"{save_model_dir}/{config_reader.get_value('train_model')['model_name']}"
    if os.path.exists(save_model_path):
        model = load_model(
            pretrained_model_name=pretrained_model_name, 
            num_labels=num_labels, 
            save_model_path=save_model_path
        )
        model.to(device)
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

    else:
        raise Exception(f"model is not exists: {save_model_path}")
        
    logger.info("begin model predict")
    test_data_predict_result = inference(
            model=model, 
            device=device,
            tokenizer=tokenizer, 
            ids_to_labels=ids_to_labels, 
            test_dir_path=testset_dir_path, 
            batch_size=batch_size,
            entity_types=entity_types
        )
    
    logger.info(f"{'=' * 15} no add keyword evaluate {'=' * 15}")
    evaluate_by_type(
        data=test_dataset, 
        data_predict_result=test_data_predict_result, 
        entity_types=entity_types,
        max_len=max_len,
        save_file_path=test_no_add_keyword_data_predict_file_path
    )
    
    test_data_add_keyword_predict_result = inference(
            model=model, 
            device=device,
            tokenizer=tokenizer, 
            ids_to_labels=ids_to_labels, 
            test_dir_path=testset_add_keyword_dir_path, 
            batch_size=batch_size,
            entity_types=entity_types
    )
    
    logger.info(f"{'=' * 15} add keyword evaluate {'=' * 15}")
    evaluate_by_type(
        data=test_dataset_add_keyword, 
        data_predict_result=test_data_add_keyword_predict_result, 
        entity_types=entity_types,
        max_len=max_len,
        save_file_path=test_add_keyword_data_predict_file_path
    )
    
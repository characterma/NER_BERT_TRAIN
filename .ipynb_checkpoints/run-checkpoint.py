__author__ = "Sally He"
import json
import argparse
import logging
import pickle, os
import pandas as pd
from loguru import logger
from src.config import ConfigReader
from src.entity_word_eval import first_entity_word_eval, first_entity_word_eval_update
from src.data_preprocess import data_preprocess
from src.product_type_word_evaluation import product_type_word_evaluation
from src.sentence_include_entity_eval import sentence_include_entity_eval, sentence_include_entity_eval_update
from src.generate_dataset import generate_model_used_dataset
from src.train_model import train_model
from src.evaluate import model_evalate, model_evalate_add_keyword_and_no_add_keyword


def run(config_reader):
    """
    - entity_word_eval
    - data_preprocess
    - sentence_include_entity_eval
    - generate_dataset
    - train_model
    - test_model
    """
    # 获取本次的处理环节
    
    stages = config_reader.get_value("stages") 
    
#     if "data_preprocess" in stages:
#         df_data = data_preprocess(config_reader)
#         logger.info("step 1: data_preprocess is completed")
#     else:
#         output_dir_path = f"{config_reader.get_value('output_dir_path')}/{config_reader.get_value('industry')}_{config_reader.get_value('date')}/"
#         file_name = config_reader.get_value('data_preprocess')['processed_file_name']
#         processed_file_path = os.path.join(output_dir_path, file_name)
#         if os.path.exists(processed_file_path):
#             logger.info("step 1: data_preprocess result file always exists")
#         else:
#             raise Exception(f"step 1: data_preprocess result file is not exists, Please follow the step1. {processed_file_path}")
            
#     if "product_type_word_evaluation" in stages:
#         product_type_word_evaluation(config_reader)
#         logger.info("step 2: product_type_word_evaluation is completeed")
#     else:
#         product_type_word_evaluation_result_file_name = config_reader.get_value("product_type_word_evaluation")['product_type_word_evaluation_result']
#         output_dir_path = f"{config_reader.get_value('output_dir_path')}/{config_reader.get_value('industry')}_{config_reader.get_value('date')}/"
#         product_type_word_evaluation_result_file_path = os.path.join(output_dir_path, product_type_word_evaluation_result_file_name)
#         if os.path.exists(processed_file_path):
#             logger.info(f"step 2: product_type_word_evaluation_result_file_path always exists:{product_type_word_evaluation_result_file_path}")
#         else:
#             raise Exception(f"step 2: product_type_word_evaluation result file is not exists, Please follow the step2. {product_type_word_evaluation_result_file_path}")
        
    
#     if "entity_word_eval" in stages:
#         first_entity_word_eval_update(config_reader)
#         # first_entity_word_eval(config_reader)
#         logger.info("step 3: entity_word_eval is completed")
#     else:
#         output_dir_path = f"{config_reader.get_value('output_dir_path')}/{config_reader.get_value('industry')}_{config_reader.get_value('date')}/"
#         file_name = config_reader.get_value('entity_word_eval')['first_entity_eval_result_file']
#         first_eval_result_file_path = os.path.join(output_dir_path, file_name)
#         if os.path.exists(first_eval_result_file_path):
#             logger.info("step 3: entity_word_eval result file always exists")
#         else:
#             raise Exception(f"step 3: entity_word_eval result file is not exists, Please follow the step3. {first_eval_result_file_path}")
 
    
    if "sentence_include_entity_eval" in stages:
        # sentence_include_entity_eval(config_reader)
        sentence_include_entity_eval_update(config_reader)
        logger.info("step 4: sentence_include_entity_eval is completed")

    else:
        output_dir_path = f"{config_reader.get_value('output_dir_path')}/{config_reader.get_value('industry')}_{config_reader.get_value('date')}/"
        file_name = config_reader.get_value('sentence_include_entity_eval')['second_entity_eval_result_file']
        second_eval_result_file_path = os.path.join(output_dir_path, file_name)
        if os.path.exists(second_eval_result_file_path):
            logger.info("step 4: sentence_include_entity_eval result file always exists")
        else:
            raise Exception(f"step 4: sentence_include_entity_eval result file is not exists, Please follow the step4. {second_eval_result_file_path}")
    
    if "generate_dataset" in stages:
        generate_model_used_dataset(config_reader)
        logger.info("step 5: generate_dataset is completed")

    else:
        save_dir_path = f"{config_reader.get_value('output_dir_path')}/{config_reader.get_value('industry')}_{config_reader.get_value('date')}"
        train_dir = os.path.join(save_dir_path, config_reader.get_value("generate_dataset")['train_data'])
        test_dir = os.path.join(save_dir_path, config_reader.get_value("generate_dataset")['test_data'])
        valid_dir = os.path.join(save_dir_path, config_reader.get_value("generate_dataset")['valid_data'])

        
        if os.path.exists(train_dir) \
        and os.listdir(train_dir).__len__() > 0 \
        and os.path.exists(test_dir) \
        and os.listdir(test_dir).__len__() > 0 \
        and os.path.exists(valid_dir) \
        and os.listdir(valid_dir).__len__() > 0:
            logger.info(f"step 5: generate_dataset result file always exists. {train_dir}")
        else:
            raise Exception(f"step 5: generate_dataset result file is not exists, Please follow the step5, {train_dir}")
    
    if "train_model" in stages:
        train_model(config_reader)
        logger.info("step 6: model trianing is completed")
    else:
        output_dir_path = f"{config_reader.get_value('output_dir_path')}/{config_reader.get_value('industry')}_{config_reader.get_value('date')}/"
        save_model_dir = f"{output_dir_path}/{config_reader.get_value('train_model')['save_model_dir']}" 
        save_model_path = f"{save_model_dir}/{config_reader.get_value('train_model')['model_name']}"
        if os.path.exists(save_model_path):
            logger.info(f"step 6: model always exists: {save_model_path}")
        else:
            raise Exception(f"step 6: model is not exists, please follow the step6, {save_model_path}")
        
    if "model_evaluate" in stages:
        model_evalate_add_keyword_and_no_add_keyword(config_reader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    logger.info(f"args: {args}")
    config_reader = ConfigReader(args.config)
    config_reader.load_config()
    run(config_reader)
    


    

    

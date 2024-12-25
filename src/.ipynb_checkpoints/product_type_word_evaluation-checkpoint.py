from tqdm import tqdm
from src.utils.content_split import generate_entity2sentence_list, get_mid_length_sentence
from src.utils.gpt_caller_asyncio import get_gpt_result
from loguru import logger
import pandas as pd
import os
from src.utils import log_step

# 结果解析
def parse_llm_response(example):
    def _find_backticks_index(example):
        pos = []
        index = 0

        while True:
            index = example.find("```", index)
            if index == -1:
                break
            else:
                pos.append(index)
                index += 1
        
        if len(pos) >= 2:
            return example[pos[-2]+3: pos[-1]]
        else:
            return example

    if isinstance(example, dict):
        return example
    elif isinstance(example, str):
        example = _find_backticks_index(example)
        example = example.replace("json", "")
        example = example.replace("`", "")
        example = example.strip("\n")
    try:
        example = eval(example)
        if isinstance(example, dict):
            return example
        else:
            return {}
    except Exception as e:
        print(e)
        print(example)
        return {}
    
def get_llm_response_result_and_merge_result(need_eval_entity_type, gpt_output):
    """
    解析deepseek对品类实体的验证结果，并添加到原数据中
    """
    new_data_list = []
    for index, row in tqdm(need_eval_entity_type.iterrows()):
        response_json = gpt_output.get(index, {})
        response_dict = parse_llm_response(response_json)
        row_dict = row.to_dict()
        row_dict["entity_llm_response"] = response_dict
        product_type_root = response_dict.get("品类词根", "")
        if isinstance(product_type_root, str) and product_type_root != "" and product_type_root.__len__() > 1:
            row_dict['entity_root'] = product_type_root
        else:
            row_dict['entity_root'] = row_dict['entity']
        new_data_list.append(row_dict)
    return pd.DataFrame(new_data_list)


@log_step(15)
def product_type_word_evaluation(config_reader):
    entity_key_list = config_reader.get_value("entity_type", None)
    vkg_entity_data_path = config_reader.get_value("data")["vkg_entity_data_path"]
    product_type_name = config_reader.get_value("product_type_word_evaluation")['product_type_name']
    template_code = config_reader.get_value("product_type_word_evaluation")['template_code']
    product_type_word_evaluation_result_file_name = config_reader.get_value("product_type_word_evaluation")['product_type_word_evaluation_result']
    output_dir_path = f"{config_reader.get_value('output_dir_path')}/{config_reader.get_value('industry')}_{config_reader.get_value('date')}/"
    product_type_word_evaluation_result_file_path = os.path.join(output_dir_path, product_type_word_evaluation_result_file_name)
    industry_name = config_reader.get_value("industry")
    
    entity_df = pd.read_excel(vkg_entity_data_path)
    need_eval_entity_df, no_need_eval_entity_df = entity_df[entity_df['entity_type']==product_type_name], entity_df[entity_df['entity_type']!=product_type_name]
    if need_eval_entity_df.shape[0] == 0:
        logger.info("need_eval_entity_df is null")
        entity_df['entity_root'] = entity_df['entity']
        sign_list = ["entity" in entity_df.columns.tolist(), "entity_type" in entity_df.columns.tolist(), "entity_root" in entity_df.columns.tolist()]
        assert all(sign_list), Exception(f"entity and entity_type must be in entity_df columns, entity: {sign_list[0]}, entity_type: {sign_list[1]}, entity_root: {sign_list[2]}")
        logger.info(f"{product_type_name}实体优化前的实体量: {entity_df['entity_type'].value_counts().to_dict()}, 优化后的实体量： {evaled_product_type_entity_df.drop_duplicates(subset=['entity_root'])['entity_type'].value_counts().to_dict()}")
        entity_df.to_excel(evaled_product_type_entity_df, index=False)
    else:
        doc_list = [{
            "doc_id": index,
            "headline": "", 
            "content": row['entity'],
            "行业": industry_name, 
            "keyword": row['entity']
        } for index, row in need_eval_entity_df.iterrows()]
        logger.info(f"need_eval_entity_df shape: {need_eval_entity_df.shape}, no_need_eval_entity_df: {no_need_eval_entity_df.shape}, need eval example: {doc_list[0]}")
        gpt_output = get_gpt_result(
            input_data = doc_list, 
            template_code = template_code,
            api_url = "http://aiapi.wisers.com/openai-result-service-api/common/invoke", 
            semaphore_num = 60, 
            tags=config_reader.get_value("tags"),
            system_message_variable = []
        )

        product_type_entity_eval_df = get_llm_response_result_and_merge_result(need_eval_entity_df, gpt_output)
        logger.info(f"数据列名: {product_type_entity_eval_df.columns}, {product_type_name}实体优化结果示例：{product_type_entity_eval_df.head(5).to_dict('records')}")
        no_need_eval_entity_df['entity_root'] = no_need_eval_entity_df['entity']
        evaled_product_type_entity_df = pd.concat([no_need_eval_entity_df, product_type_entity_eval_df])
        logger.info(f"evaled_product_type_entity_df列名: {evaled_product_type_entity_df.columns}")
        os.makedirs(output_dir_path, exist_ok=True)
        logger.info(f"{product_type_name}实体优化前的实体量: {entity_df['entity_type'].value_counts().to_dict()}, 优化后的实体量： {evaled_product_type_entity_df.drop_duplicates(subset=['entity_root'])['entity_type'].value_counts().to_dict()}")
        sign_list = ["entity" in evaled_product_type_entity_df.columns.tolist(), "entity_type" in evaled_product_type_entity_df.columns.tolist(), "entity_root" in evaled_product_type_entity_df.columns.tolist()]
        assert all(sign_list), Exception(f"entity and entity_type must be in entity_df columns, entity: {sign_list[0]}, entity_type: {sign_list[1]}, entity_root: {sign_list[2]}")
        logger.info(f"验证完成并将数据保存在{product_type_word_evaluation_result_file_path}")
        evaled_product_type_entity_df.to_excel(product_type_word_evaluation_result_file_path, index=False)
    

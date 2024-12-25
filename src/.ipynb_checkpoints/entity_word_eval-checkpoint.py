from tqdm import tqdm
from src.utils.content_split import generate_entity2sentence_list, get_mid_length_sentence
from src.utils.gpt_caller_asyncio import get_gpt_result
from loguru import logger
import pandas as pd
import os
from src.utils import log_step


def generate_eval_prompt1(eval_prompt1_path, industry_name):
    with open(eval_prompt1_path) as f:
        eval_prompt1 = f.read().replace("{{industry_name}}", industry_name)
    return eval_prompt1

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

def get_llm_response_result_and_merge_result(entity_sentence_list_df, gpt_output):
    """
    解析deepseek对实体的验证结果，并添加到原数据中
    """
    new_data_list = []
    correct_entity_list = []
    for index, row in tqdm(entity_sentence_list_df.iterrows()):
        doc_id = str(row.get("doc_id", ""))
        response_json = gpt_output.get(doc_id, {})
        response_dict = parse_llm_response(response_json)
        row_dict = row.to_dict()
        row_dict["entity_llm_response"] = response_dict
        row_dict.update(response_dict)
        new_data_list.append(row_dict)
        entity_type_correct = response_dict.get("entity_type_correct", "").lower()
        keyword_redundant = response_dict.get("keyword_redundant", "").lower()
        if keyword_redundant == "":
            if entity_type_correct == "yes":
                correct_entity_list.append({"entity_type": row.get("entity_type"), "entity": row.get("entity"), "count": row.get("count")})
        else:
            if entity_type_correct == "yes" and keyword_redundant == "no":
                correct_entity_list.append({"entity_type": row.get("entity_type"), "entity": row.get("entity"), "count": row.get("count")})
    return pd.DataFrame(new_data_list), correct_entity_list

@log_step(15)
def first_entity_word_eval(config_reader):
    entity_key_list = config_reader.get_value("entity_type", None)
    if entity_key_list is None:
        raise Exception("entity type must be not null")
    output_dir_path = f"{config_reader.get_value('output_dir_path')}/{config_reader.get_value('industry')}_{config_reader.get_value('date')}/"
    file_name = config_reader.get_value('data_preprocess')['processed_file_name']
    processed_file_path = os.path.join(output_dir_path, file_name)
    df_data = pd.read_excel(processed_file_path)
    df_data['processed_entities'] = df_data['processed_entities'].map(eval)
    logger.info(f"processed data is parpare completed: {df_data.shape}, {df_data.columns}")
    entity_type_name_con_dict, entity_type_name_count_dict = generate_entity2sentence_list(df_data, "headline_content", "processed_entities", entity_key_list=entity_key_list)
    logger.info("生成验证需要的user message作为输入")
    entity_sentence_list = get_mid_length_sentence(entity_type_name_con_dict, entity_type_name_count_dict)
    entity_sentence_list_df = pd.DataFrame(entity_sentence_list)
    entity_sentence_list_df['content'] = entity_sentence_list_df.apply(lambda row: {
            "keyword": row["entity"],
            "sentence": row["sentence"],
            "question": f"上述句子中出现的'{row['entity']}'是否属于{row['entity_type']}实体类型？"
        }, axis=1)
    
    if "doc_id" not in entity_sentence_list_df.columns:
        entity_sentence_list_df["doc_id"] = range(0, entity_sentence_list_df.shape[0])
    template_code = config_reader.get_value("entity_word_eval")["template_code"]
    industry_name = config_reader.get_value("industry")

    logger.info(f"请求prompt template: {template_code} 进行{industry_name}行业实体验证")
    doc_list = [{"doc_id": str(row.get("doc_id", "")),"headline": "", "content": str(row.get("content", ""))}  for _, row in entity_sentence_list_df.iterrows()]
    industry_list = [{"industry_name": f"{industry_name}"} for i in doc_list] 
    gpt_output = get_gpt_result(
            input_data = doc_list, 
            template_code = config_reader.get_value("entity_word_eval")["template_code"], 
            api_url = "http://aiapi.wisers.com/openai-result-service-api/invoke", 
            semaphore_num = 60, 
            # tags = "vkg_luxury", 
            tags=config_reader.get_value("tags"),
            system_message_variable = industry_list
    )
    first_eval_result_data, correct_entity_list = get_llm_response_result_and_merge_result(entity_sentence_list_df, gpt_output)
    
    output_dir_path = f"{config_reader.get_value('output_dir_path')}/{config_reader.get_value('industry')}_{config_reader.get_value('date')}/"
    file_name = config_reader.get_value('entity_word_eval')['first_entity_eval_result_file']
    first_eval_result_file_path = os.path.join(output_dir_path, file_name)
    os.makedirs(output_dir_path, exist_ok=True)
    logger.info(f"验证完成并将数据保存在{first_eval_result_file_path}")
    first_eval_result_data.to_excel(first_eval_result_file_path)
    

@log_step(15)
def first_entity_word_eval_update(config_reader):
    product_type_word_evaluation_result_file_name = config_reader.get_value("product_type_word_evaluation")['product_type_word_evaluation_result']
    output_dir_path = f"{config_reader.get_value('output_dir_path')}/{config_reader.get_value('industry')}_{config_reader.get_value('date')}/"
    product_type_word_evaluation_result_file_path = os.path.join(output_dir_path, product_type_word_evaluation_result_file_name)
    vkg_entity_df = pd.read_excel(product_type_word_evaluation_result_file_path)
    origin_entity2root_entity = {row['entity']: row['entity_root'] for index, row in vkg_entity_df.iterrows()}
    logger.info(f"VKG 实体词表的数量：{origin_entity2root_entity.__len__()}, entiy root的类型分布: {vkg_entity_df.drop_duplicates(subset=['entity_root'])['entity_type'].value_counts()}")
    entity_key_list = config_reader.get_value("entity_type", None)
    if entity_key_list is None:
        raise Exception("entity type must be not null")
    logger.info("使用更新后的实体次验证方式")
    output_dir_path = f"{config_reader.get_value('output_dir_path')}/{config_reader.get_value('industry')}_{config_reader.get_value('date')}/"
    entity_word_eval_example_file = config_reader.get_value("entity_word_eval")["entity_word_eval_example"]
    entity_word_eval_example = open(entity_word_eval_example_file).read().strip()
    file_name = config_reader.get_value('data_preprocess')['processed_file_name']
    processed_file_path = os.path.join(output_dir_path, file_name)
    df_data = pd.read_excel(processed_file_path)
    df_data['processed_entities'] = df_data['processed_entities'].map(eval)
    logger.info(f"processed data is parpare completed: {df_data.shape}, {df_data.columns}")
    entity_type_name_con_dict, entity_type_name_count_dict = generate_entity2sentence_list(df_data, "headline_content", "processed_entities", entity_key_list=entity_key_list)
    logger.info("生成验证需要的user message作为输入")
    entity_sentence_list = get_mid_length_sentence(entity_type_name_con_dict, entity_type_name_count_dict)
    entity_sentence_list_df = pd.DataFrame(entity_sentence_list)
    
    # 因为我们抽取到的品类实体很多都是冗余的，所以上一步优化了品类实体，这里需要将原品类实体替换成优化后的品类实体进行验证
    entity_sentence_list_df['entity'] = entity_sentence_list_df['entity'].map(origin_entity2root_entity)
    entity_sentence_list_df.drop_duplicates(subset=['entity'], inplace=True)
    
    if "doc_id" not in entity_sentence_list_df.columns:
        entity_sentence_list_df["doc_id"] = range(0, entity_sentence_list_df.shape[0])
    template_code = config_reader.get_value("entity_word_eval")["template_code"]
    industry_name = config_reader.get_value("industry")

    logger.info(f"请求prompt template: {template_code} 进行{industry_name}行业实体验证")
    doc_list = [{
        "doc_id": str(row.get("doc_id", "")),
        "headline": "", 
        "content": f"上述句子中出现的'{row['entity']}'是否属于{row['entity_type']}实体类型？'{row['entity']}'作为{row['entity_type']}实体类型是否冗余？", 
        "entity": row["entity"],
        "sentence": row["sentence"],
        "entity_type": row['entity_type'],
        "industry_name": f"{industry_name}",  
        "examples": entity_word_eval_example
    } for _, row in entity_sentence_list_df.iterrows()]
    logger.info(f"需要验证数据条数： {doc_list.__len__()}, \n示例: {doc_list[0]}")
    # raise Exception("stop first_entity_word_eval_update")
    # industry_list = [{"industry_name": f"{industry_name}",  "examples": entity_word_eval_example} for i in doc_list] 
    gpt_output = get_gpt_result(
            input_data = doc_list, 
            template_code = template_code, 
            api_url = "http://aiapi.wisers.com/openai-result-service-api/common/invoke", 
            semaphore_num = 60, 
            tags=config_reader.get_value("tags"),
            system_message_variable = []
    )
    first_eval_result_data, correct_entity_list = get_llm_response_result_and_merge_result(entity_sentence_list_df, gpt_output)
    
    output_dir_path = f"{config_reader.get_value('output_dir_path')}/{config_reader.get_value('industry')}_{config_reader.get_value('date')}/"
    file_name = config_reader.get_value('entity_word_eval')['first_entity_eval_result_file']
    first_eval_result_file_path = os.path.join(output_dir_path, file_name)
    os.makedirs(output_dir_path, exist_ok=True)
    logger.info(f"验证完成并将数据保存在{first_eval_result_file_path}")
    first_eval_result_data.to_excel(first_eval_result_file_path)
    

    
    
    
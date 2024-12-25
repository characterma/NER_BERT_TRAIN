from src.utils.content_split import get_sub_content
from src.utils.keyword_matching import KeyWordExtractor
from loguru import logger
import os
import pandas as pd
from tqdm import tqdm
from src.utils.gpt_caller_asyncio import get_gpt_result
from src.utils import log_step


def split_content_and_generate_sub_content_and_expand_entity_col(data, entities_col_name, content_col_name='cleaned_headline_content', max_length=300):
    """
    对长句进行切句并找到分句后的句子中存在实体
    """
    new_data = []
    for index, row in tqdm(data.iterrows()):
        row_dict = row.to_dict()
        doc_entity_dict = {}
        for dic in row_dict.get(entities_col_name):
            entity = dic['entity']
            entity_type = dic['entity_type']
            if entity_type in doc_entity_dict:
                doc_entity_dict[entity_type].add(entity)
            else:
                doc_entity_dict[entity_type] = set()
                doc_entity_dict[entity_type].add(entity)
        
        processed_text = str(row.get(content_col_name)).lower()
        sub_content_list = get_sub_content(processed_text, max_length)
        for idx, sub_content in enumerate(sub_content_list):
            row_dict_copy = row_dict.copy()
            row_dict_copy["context"] = sub_content
            sub_entity_dict = {}
            for key, entity_list in doc_entity_dict.items():
                sub_entity_list = []
                for entity in entity_list:
                    if entity in sub_content:
                        sub_entity_list.append(entity)
                sub_entity_dict[key] = sub_entity_list
            row_dict_copy.update(sub_entity_dict)
            row_dict_copy["docid"] = f'{row_dict_copy.get("doc_id")}_{idx}'
            new_data.append(row_dict_copy)
    return pd.DataFrame(new_data)

def filter_entities(row, entity_types, entity_type2correct_entities):
    all_correct_entities = []
    for entity_type in entity_types:
        all_correct_entities.append([entity for entity in row[entity_type] if entity in entity_type2correct_entities[entity_type]])
    return all_correct_entities

def get_eval_correct_entity(first_eval_result_file_path, filter_correct_entity_condition):
    if os.path.exists(first_eval_result_file_path):
        entity_eval_result = pd.read_excel(first_eval_result_file_path)
        filter_correct_df = entity_eval_result.query(filter_correct_entity_condition)
        logger.info(f"验证的实体数量： {entity_eval_result.shape}, 验证结果的数据列名：{entity_eval_result.columns}, 验证正确的实体数量:{filter_correct_df.shape}")
        entity_type2correct_entities = {}
        # 通过配置文件中配置的过滤条件进行正确实体的验证
        for index, row in filter_correct_df.iterrows():
            # 实体类型是否正确：entity_type_correct  行业是否相关：industry_related
            # if row['entity_type_correct'] == "yes":
            if row['entity_type'] in entity_type2correct_entities:
                entity_type2correct_entities[row['entity_type']].add(str(row['entity']).lower())
            else:
                entity_type2correct_entities[row['entity_type']] = set()
                entity_type2correct_entities[row['entity_type']].add(str(row['entity']).lower())
        return entity_type2correct_entities
    else:
        raise Exception(f"eval_result_file_path is not exists: {eval_result_file_path}")


def split_content_and_filter_error_entity(df_data, entity_type2correct_entities, entity_types, max_length):
    """
    切句并将找出切分后的句子中包含的实体,
    """        
    split_content_df = split_content_and_generate_sub_content_and_expand_entity_col(data=df_data, 
                                                                                    entities_col_name="processed_entities", 
                                                                                    content_col_name="cleaned_headline_content", 
                                                                                    max_length=max_length
                                                                                   )
    logger.info(f"split_content_df: {split_content_df.shape}")
    split_content_df['first_eval_correct_entities'] = [[] for i in range(split_content_df.shape[0])]
    for entity_type in entity_types:
        """
        过滤掉验证结果为错误的实体并在文本内容中添加提示语，生成添加提示语之后的文本列
        """
        split_content_df[f'filtered_{entity_type}'] = split_content_df[entity_type].apply(lambda entities: filter_substring([str(entity).lower() for entity in entities if str(entity).lower() in entity_type2correct_entities[entity_type]]) if isinstance(entities, list) else [])
        split_content_df['first_eval_correct_entities'] = split_content_df.apply(lambda row: row['first_eval_correct_entities'] + row[f'filtered_{entity_type}'], axis=1)
    
    split_content_df['context_keywords'] = split_content_df.apply(lambda row: add_prompt_in_content(row['context'], row[f'first_eval_correct_entities']), axis=1)
    return split_content_df 

def add_prompt_in_content(text, labels_):
    """
    添加提示语是在第二次实体验证之前，验证后的实体准确率很高，基本都是正确的实体，再加上提示语模型会过拟合，验证之前的实体包含一定的错误实体，需要模型能学习到错误实体的特征
    在text的内容中添加 '||可能的实体：' + ",".join(entities) + '||' 的提示语
    :param text: str, 文本内容
    :param labels_: 匹配到的所有实体
    """
    text_list = []
    separator = "。！？#?!"
    start, end = 0, 0
    while end < len(text):
        if text[end] in separator:
            text_list.append(text[start:end+1])
            start, end = end + 1, end + 1
        else:
            end += 1
        
        if end == len(text) - 1:
            text_list.append(text[start:end+1])

    for idx, t in enumerate(text_list):
        temp = []
        for l in labels_:
            if l in t:
                temp.append(l)
        if temp != []:
            text_list[idx] = text_list[idx] + '||可能的实体：' + ",".join(temp) + '||' 
    return "".join(text_list)

def filter_substring(labels):
    labels_ = []
    for i in labels:
        i = str(i)
        temp = labels.copy()
        temp.remove(i)
        flag = 1
        for j in temp:
            j = str(j)
            if i not in j:
                continue
            else:
                flag = 0
                break
        if flag:
            labels_.append(i)
    return labels_

    
def generate_eval_entity_input(row, entity_types):
    need_eval_entity = []
    for entity_type in entity_types:
        need_eval_entity.extend([{"entity_name": entity, "entity_type": entity_type} for entity in row[f'filtered_{entity_type}']])
    
    input_info = {"doc_id": str(row.get("docid", "")),"need_eval_entity": need_eval_entity, "content": str(row.get("context", ""))}
    return need_eval_entity, input_info


def parse_llm_response(example):
    """
    一条一条解析二次验证的结果
    """
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

def parse_and_generate_df(gpt_output):
    """
    解析llm响应结果并生成dataframe
    """
    all_content_eval_result = []
    for doc_id, llm_response in gpt_output.items():
        response_js = parse_llm_response(llm_response)
        if "judgement_result" in response_js:
            all_content_eval_result.append({
                "doc_id": doc_id,
                "llm_response": response_js['judgement_result']
            })
        else:
            all_content_eval_result.append({
                "doc_id": doc_id,
                "llm_response": []
            })
    return pd.DataFrame(all_content_eval_result)


@log_step(15)
def sentence_include_entity_eval(config_reader):
    output_dir_path = f"{config_reader.get_value('output_dir_path')}/{config_reader.get_value('industry')}_{config_reader.get_value('date')}/"
    file_name = config_reader.get_value('entity_word_eval')['first_entity_eval_result_file']
    entity_type2correct_entities = {}
    business_entity_dict_path = config_reader.get_value("data")['business_entity_dict_path']
    first_eval_result_file_path = os.path.join(output_dir_path, file_name)
    logger.info(f"开始二次实体验证的数据准备,读取验证结果数据：{first_eval_result_file_path}")
    
    if os.path.exists(first_eval_result_file_path):
        filter_correct_entity_condition = config_reader.get_value("sentence_include_entity_eval")["filter_correct_entity_condition"]
        entity_type2correct_entities = get_eval_correct_entity(first_eval_result_file_path, filter_correct_entity_condition)
        entity_type2count = {entity_type: len(correct_entities) for entity_type, correct_entities in entity_type2correct_entities.items()}
        logger.info(f"vkg entity dict: {entity_type2count}")
    else:
        raise Exception(f"first entity eval result is not exists:{first_eval_result_file_path}")
        
    
    if os.path.exists(business_entity_dict_path):
        business_entity_dict = pd.read_excel(business_entity_dict_path)
        for index, row in business_entity_dict.iterrows():
            entity, entity_type = row['entity'], row['entity_type']
            entity_type2correct_entities[entity_type].add(str(entity).lower())
        entity_type2count = {entity_type: len(correct_entities) for entity_type, correct_entities in entity_type2correct_entities.items()}
        logger.info(f"vkg and business entity: {entity_type2count}")
    else:
        raise Exception(f"business_entity_dict_path is not exists:{business_entity_dict_path}")
        
    output_dir_path = f"{config_reader.get_value('output_dir_path')}/{config_reader.get_value('industry')}_{config_reader.get_value('date')}/"
    file_name = config_reader.get_value('data_preprocess')['processed_file_name']
    processed_file_path = os.path.join(output_dir_path, file_name)
    if os.path.exists(processed_file_path):
        logger.info(f"读取原实体打标数据：{processed_file_path}")
        df_data = pd.read_excel(processed_file_path)
        df_data['processed_entities'] = df_data['processed_entities'].map(eval)
        logger.info(f"原实体打标数据：{df_data.shape}")
        df_data.dropna(subset=['processed_entities'], inplace=True)  
        logger.info(f"原实体打标数据通过processed_entities去除空值：{df_data.shape}")
    else:
        raise Exception(f"processed file is not exists: {processed_file_path}")
    
    entity_types = config_reader.get_value("entity_type")
    max_length = config_reader.get_value("sentence_include_entity_eval")['split_content_max_length']
    industry_name = config_reader.get_value("industry")
    
    split_content_df = split_content_and_filter_error_entity(df_data, entity_type2correct_entities, entity_types, max_length)
    split_content_df.to_excel(f"{output_dir_path}/split_content_df.xlsx")
    logger.info(f"split_content_df: {split_content_df.shape}")            
    
    split_content_df[['need_eval_entity', 'input_info']] = split_content_df.apply(generate_eval_entity_input, args=(entity_types, ), axis=1, result_type="expand")
    
    template_code = config_reader.get_value("sentence_include_entity_eval")['template_code']
    entity_type_definition_and_example_file_path = config_reader.get_value("sentence_include_entity_eval")['entity_type_definition_and_example_file_path']
    logger.info(f"second eval template_code: {template_code}, entity_type_definition_and_example_file_path: {entity_type_definition_and_example_file_path}")
    
    entity_type_definition_and_example_df = pd.read_csv(entity_type_definition_and_example_file_path)
    entity_type_definition_and_example = str(entity_type_definition_and_example_df.to_dict("records"))
    logger.info(f"目标类型的定义： {entity_type_definition_and_example}")
    
    # 实体为空的数据不进行验证
    logger.info(split_content_df.columns)
    doc_list = []
    entity_list_is_null_num = 0
    for input_info in split_content_df['input_info'].tolist():
        if isinstance(input_info['need_eval_entity'], list) and input_info['need_eval_entity'].__len__() == 0:
            entity_list_is_null_num += 1
            continue
        input_info.update({
            "headline": "",
            "industry": industry_name,
            "entity_type_definition_and_example": entity_type_definition_and_example,
            "all_entity_types": entity_types
        })
        doc_list.append(input_info)
    logger.info(f"文本内容中没有实体的数据有{entity_list_is_null_num}条，需要验证的数据条数doc_list: {doc_list.__len__()}, 开始二次验证,示例如下：\n{doc_list[0]}")
    
    # raise Exception("second entity eval stop")
    gpt_output = get_gpt_result(
        input_data = doc_list, 
        template_code = template_code, 
        api_url = "http://aiapi.wisers.com/openai-result-service-api/common/invoke", 
        semaphore_num = 60, 
        tags = config_reader.get_value("tags") 
    )
    all_content_eval_result = parse_and_generate_df(gpt_output)
    for entity_type in entity_types:
        all_content_eval_result[f'eval_{entity_type}'] = all_content_eval_result['llm_response'].map(lambda llm_response: [str(dic['entity_name'].lower()) for dic in llm_response if dic['correct_entity_type'] == entity_type])
    
    complete_split_content_df = split_content_df[["docid", "processed_entities", "need_eval_entity", "context", "context_keywords", "input_info"]+[f"filtered_{entity_type}" for entity_type in entity_types]].merge(all_content_eval_result, left_on='docid', right_on='doc_id', how='left')
    
    output_dir_path = f"{config_reader.get_value('output_dir_path')}/{config_reader.get_value('industry')}_{config_reader.get_value('date')}/"
    file_name = config_reader.get_value('sentence_include_entity_eval')['second_entity_eval_result_file']
    os.makedirs(output_dir_path, exist_ok=True)
    second_eval_result_file_path = os.path.join(output_dir_path, file_name)
    logger.info(f"二次验证完成并保存结果: {second_eval_result_file_path}")
    complete_split_content_df.to_excel(second_eval_result_file_path)

    
@log_step(15)
def sentence_include_entity_eval_update(config_reader):
    output_dir_path = f"{config_reader.get_value('output_dir_path')}/{config_reader.get_value('industry')}_{config_reader.get_value('date')}/"
    file_name = config_reader.get_value('entity_word_eval')['first_entity_eval_result_file']
    few_shot = open(config_reader.get_value("sentence_include_entity_eval")['few_shot_file_path']).read().strip()
    entity_type2correct_entities = {}
    business_entity_dict_path = config_reader.get_value("data")['business_entity_dict_path']
    first_eval_result_file_path = os.path.join(output_dir_path, file_name)
    split_content_df_file_path = f"{output_dir_path}/split_content_df.xlsx"
    logger.info(f"开始二次实体验证的数据准备,读取验证结果数据：{first_eval_result_file_path}")
    
    if os.path.exists(first_eval_result_file_path):
        filter_correct_entity_condition = config_reader.get_value("sentence_include_entity_eval")["filter_correct_entity_condition"]
        entity_type2correct_entities = get_eval_correct_entity(first_eval_result_file_path, filter_correct_entity_condition)
        entity_type2count = {entity_type: len(correct_entities) for entity_type, correct_entities in entity_type2correct_entities.items()}
        logger.info(f"vkg entity dict: {entity_type2count}")
    else:
        raise Exception(f"first entity eval result is not exists:{first_eval_result_file_path}")


    if os.path.exists(business_entity_dict_path):
        business_entity_dict = pd.read_excel(business_entity_dict_path)
        for index, row in business_entity_dict.iterrows():
            entity, entity_type = row['entity'], row['entity_type']
            entity_type2correct_entities[entity_type].add(str(entity).lower())
        entity_type2count = {entity_type: len(correct_entities) for entity_type, correct_entities in entity_type2correct_entities.items()}
        logger.info(f"vkg and business entity: {entity_type2count}")
    else:
        raise Exception(f"business_entity_dict_path is not exists:{business_entity_dict_path}")

    output_dir_path = f"{config_reader.get_value('output_dir_path')}/{config_reader.get_value('industry')}_{config_reader.get_value('date')}/"
    file_name = config_reader.get_value('data_preprocess')['processed_file_name']
    processed_file_path = os.path.join(output_dir_path, file_name)
    if os.path.exists(processed_file_path):
        logger.info(f"读取原实体打标数据：{processed_file_path}")
        df_data = pd.read_excel(processed_file_path)
        df_data['processed_entities'] = df_data['processed_entities'].map(eval)
        logger.info(f"原实体打标数据：{df_data.shape}")
        df_data.dropna(subset=['processed_entities'], inplace=True)  
        logger.info(f"原实体打标数据通过processed_entities去除空值：{df_data.shape}")
    else:
        raise Exception(f"processed file is not exists: {processed_file_path}")

    entity_types = config_reader.get_value("entity_type")
    max_length = config_reader.get_value("sentence_include_entity_eval")['split_content_max_length']
    industry_name = config_reader.get_value("industry")

    
    if os.path.exists(split_content_df_file_path):
        split_content_df = pd.read_excel(split_content_df_file_path)
        for entity_type in entity_types:
            split_content_df[f'filtered_{entity_type}'] = split_content_df[f'filtered_{entity_type}'].map(eval)
    else:
        split_content_df = split_content_and_filter_error_entity(df_data, entity_type2correct_entities, entity_types, max_length)
        split_content_df.to_excel(split_content_df_file_path)
        
    logger.info(f"split_content_df: {split_content_df.shape}")            
    
    split_content_df[['need_eval_entity', 'input_info']] = split_content_df.apply(generate_eval_entity_input, args=(entity_types, ), axis=1, result_type="expand")
    
    template_code = config_reader.get_value("sentence_include_entity_eval")['template_code']
    entity_type_definition_and_example_file_path = config_reader.get_value("sentence_include_entity_eval")['entity_type_definition_and_example_file_path']
    logger.info(f"second eval template_code: {template_code}, entity_type_definition_and_example_file_path: {entity_type_definition_and_example_file_path}")
    
    entity_type_definition_and_example_df = pd.read_csv(entity_type_definition_and_example_file_path)
    entity_type_definition_and_example ="\n".join([":".join(list(dict(row).values())) for index, row in entity_type_definition_and_example_df.iterrows()])
    logger.info(f"目标类型的定义： {entity_type_definition_and_example}")
    
    # 实体为空的数据不进行验证
    logger.info(split_content_df.columns)
    doc_list = []
    entity_list_is_null_num = 0
    for input_info in split_content_df['input_info'].tolist():
        if isinstance(input_info['need_eval_entity'], list) and input_info['need_eval_entity'].__len__() == 0:
            entity_list_is_null_num += 1
            continue
        elif isinstance(input_info['need_eval_entity'], list) and input_info['need_eval_entity'].__len__() > 0:
            input_info.update({
                "headline": "",
                "industry": industry_name,
                "entity_type_definition_and_example": entity_type_definition_and_example,
                "all_entity_types": entity_types,
                "few_shot": few_shot
            })
            doc_list.append(input_info)
        else:
            entity_list_is_null_num += 1
    logger.info(f"文本内容中没有实体的数据有{entity_list_is_null_num}条，需要验证的数据条数doc_list: {doc_list.__len__()}, 开始二次验证,示例如下：\n{doc_list[0]}")
    
    gpt_output = get_gpt_result(
        input_data = doc_list, 
        template_code = template_code, 
        api_url = "http://aiapi.wisers.com/openai-result-service-api/common/invoke", 
        semaphore_num = 60, 
        tags = config_reader.get_value("tags") 
    )
    # raise Exception(f"second entity eval stop: {gpt_output}")

    all_content_eval_result = parse_and_generate_df(gpt_output)
    for entity_type in entity_types:
        all_content_eval_result[f'eval_{entity_type}'] = all_content_eval_result['llm_response'].map(lambda llm_response: [str(dic['entity_name'].lower()) for dic in llm_response if dic['correct_entity_type'] == entity_type])
    
    complete_split_content_df = split_content_df[["docid", "processed_entities", "need_eval_entity", "context", "context_keywords", "input_info"]+[f"filtered_{entity_type}" for entity_type in entity_types]].merge(all_content_eval_result, left_on='docid', right_on='doc_id', how='left')
    
    output_dir_path = f"{config_reader.get_value('output_dir_path')}/{config_reader.get_value('industry')}_{config_reader.get_value('date')}/"
    file_name = config_reader.get_value('sentence_include_entity_eval')['second_entity_eval_result_file']
    os.makedirs(output_dir_path, exist_ok=True)
    second_eval_result_file_path = os.path.join(output_dir_path, file_name)
    logger.info(f"二次验证完成并保存结果: {second_eval_result_file_path}")
    complete_split_content_df.to_excel(second_eval_result_file_path)
    
    
    




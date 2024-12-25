from tqdm import tqdm
# 分句
def split_sentence(content, regex=""):
    """
    对句子按照标点符号进行分句
    """
    start = 0
    i = 0
    sents = []
    sents_index = []
    if regex:
        punt_list = regex
    else:
        punt_list = '?!;~。？！；～>\n\r'
    for word in content:
        try:
            if word in punt_list and token not in punt_list:  # 检查标点符号下一个字符是否还是标点
                sents.append(content[start:i + 1])
                start = i + 1
                i += 1
            else:
                i += 1
                token = list(content[start:i + 2]).pop()  # 取下一个字符
        except:
            i += 1
            token = list(content[start:i + 2]).pop()  # 取下一个字符
    if start < len(content):
        sents.append(content[start:])
    start_index = 0
    for sent in sents:
        sents_index.append([start_index,  start_index + len(sent)])
        start_index = start_index + len(sent)
    return sents, sents_index

def get_sub_content(content, max_length):
    sub_content_list = []
    sub_content = ""
    sents, sents_index = split_sentence(content)
    for sent in sents:
        if len(sub_content) >= max_length:
            sub_content_list.append(sub_content)
            sub_content = ""
        sub_content += str(sent)
    if sub_content:
        sub_content_list.append(sub_content)
    return sub_content_list

def generate_entity2sentence_list(data, content_cols, entities_cols, entity_key_list:list=["品牌", "产品"]):
    """
    该方法是按照每个实体进行句子整理，从包含实体词的句子中选择一个长度中等的句子进行实体的验证
    """
    # 每个类型下每个实体的句子列表
    entity_type_name_con_dict = {}
    # 每个类型下每个实体的文档频率
    entity_type_name_count_dict = {}
    miss_entity_dict = {}

    select_count = 0
    for index, doc_info in tqdm(data.iterrows()):
        select_count += 1
        content = str(doc_info.get(content_cols, ""))
        sents, sents_index = split_sentence(content)
        entities_list = doc_info.get(entities_cols, [])
        # print(sents.__len__(), sents_index.__len__(), entities_list.__len__(), entities_list)
        for entity_info in entities_list:
            if "entity" in entity_info and "entity_type" in entity_info:
                entity = entity_info['entity']
                entity_type = entity_info['entity_type']
                # print(entity, entity_type, entity_type_name_count_dict)
                if entity_type in entity_type_name_count_dict:
                    if entity in entity_type_name_count_dict[entity_type]:
                        entity_type_name_count_dict[entity_type][entity] += 1
                    else:
                        entity_type_name_count_dict[entity_type][entity] = 1
                else:
                    entity_type_name_count_dict[entity_type] = {entity: 1}
                    
                    
                if entity_type in entity_type_name_con_dict:
                    if entity in entity_type_name_con_dict[entity_type]:
                        entity_type_name_con_dict[entity_type][entity].extend(list(set(sent for sent in sents if entity in sent)))
                    else:
                        entity_type_name_con_dict[entity_type][entity] = list(set(sent for sent in sents if entity in sent))
                else:
                    entity_type_name_con_dict[entity_type] = {entity: list(set(sent for sent in sents if entity in sent))}
                
            else:
                continue
    return entity_type_name_con_dict, entity_type_name_count_dict


def get_mid_length_sentence(entity_type_name_con_dict, entity_type_name_count_dict):
    """
    entity_type_name_con_dict: {
        "实体类型": {"实体": ["所有包含实体的句子"]}
    }
    entity_type_name_count_dict: {
        "实体类型": {"实体": "实体出现的文章量"}
    }
    """
    entity_sentence_list = []
    for entity_type, entity_sentence_dict in entity_type_name_con_dict.items():
        for entity, sentence_list in entity_sentence_dict.items():
            count = entity_type_name_count_dict.get(entity_type).get(entity, 0)
            sentence_dict = {sentence: len(sentence) for sentence in sentence_list}
            sentence_rank_list = sorted(sentence_dict.items(), key=lambda item: item[1], reverse=True)
            if sentence_rank_list:
                info = {"entity_type": entity_type, "entity": entity, "count": count, "sentence": sentence_rank_list[int(len(sentence_rank_list)/2)][0]}
                entity_sentence_list.append(info)
    return entity_sentence_list
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author:Shelly
# @time:2024/3/11:12:24
import json
import re

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report as sklearn_classification_report


# 数据统计
def data_statistic_2(template_id, infile, outfile):
    entity_type_dict = {}
    human_entity_type_dict = {}
    entity_key_list = ["brand","product"]
    data = pd.read_excel(infile, keep_default_na=False)
    for index, row in data.iterrows():
        doc_id = str(row.get("docid"))
        content = str(row.get("context_cleaned")).lower()
        for dim_key in entity_key_list:
            dim_key_value = row.get(f'{template_id}_{dim_key}')
            if not dim_key_value:
                dim_key_value_list = []
            elif "[" in dim_key_value and "]" in dim_key_value:
                dim_key_value_list = eval(dim_key_value)
            else:
                dim_key_value_list = eval(f"{dim_key_value}]")
            for dim_key_value in dim_key_value_list:
                dim_key_value = dim_key_value.lower()
                if dim_key_value not in content:
                    continue
                entity_count_dict = entity_type_dict.get(dim_key_value, {})
                count = entity_count_dict.get(dim_key, 0)
                count += 1
                entity_count_dict[dim_key] = count
                entity_type_dict[dim_key_value] = entity_count_dict

            human_dim_key_value = row.get("human_" + dim_key)
            if not human_dim_key_value:
                human_dim_key_value_list = []
            elif "[" in human_dim_key_value and "]" in human_dim_key_value:
                human_dim_key_value_list = eval(human_dim_key_value)
            else:
                human_dim_key_value_list = eval(f"{human_dim_key_value}]")
            for dim_key_value in human_dim_key_value_list:
                if dim_key_value == "胸腔积水":
                    print(dim_key_value)
                dim_key_value = dim_key_value.lower()
                if dim_key_value not in content:
                    continue
                human_entity_count_dict = human_entity_type_dict.get(dim_key_value, {})
                count = human_entity_count_dict.get(dim_key, 0)
                count += 1
                human_entity_count_dict[dim_key] = count
                human_entity_type_dict[dim_key_value] = human_entity_count_dict
    print(human_entity_type_dict.get("胸腔积水", "no"))
    data_list = []
    for entity_name, entity_count_dict in entity_type_dict.items():
        if entity_name == "维生素c泡腾片":
            print(entity_name)
        human_entity_type = ""
        human_count = 0
        human_entity_count_dict = human_entity_type_dict.get(entity_name, {})
        for entity_type, count in human_entity_count_dict.items():
            human_entity_type = entity_type
            human_count = count
        for entity_type, count in entity_count_dict.items():
            # human_count = human_entity_count_dict.get(entity_type, 0)
            # human_entity_type = entity_type
            # if human_count == 0:
            #     human_entity_type = ""
            info = {"entity_name": entity_name,
                    "entity_type": entity_type,
                    "doc_count": count,
                    "human_entity_type": human_entity_type,
                    "human_doc_count": human_count,
                    }
            data_list.append(info)
        # for entity_type, count in human_entity_count_dict.items():
        #     if entity_type in entity_count_dict.keys():
        #         continue
        #     info = {"entity_name": entity_name,
        #             "entity_type": "",
        #             "doc_count": 0,
        #             "human_entity_type": entity_type,
        #             "human_doc_count": count,
        #             }
        #     data_list.append(info)
    for entity_name, human_entity_count_dict in human_entity_type_dict.items():
        if entity_name == "胸腔积水":
            print(entity_name)
        if entity_name in entity_type_dict.keys():
            continue
        for entity_type, count in human_entity_count_dict.items():
            # if entity_type in entity_count_dict.keys():
            #     continue
            info = {"entity_name": entity_name,
                    "entity_type": "",
                    "doc_count": 0,
                    "human_entity_type": entity_type,
                    "human_doc_count": count,
                    }
            data_list.append(info)
    df_type_map_list = pd.DataFrame(data_list)
    df_type_map_list.to_excel(outfile)


def data_evaluate_2(infile, outfile, outfile_2, outfile_3):
    entity_type_dict = {}
    entity_type_count_dict = {}
    data = pd.read_excel(infile, keep_default_na=False, sheet_name="Sheet1")
    for index, row in data.iterrows():
        entity_type = row.get("entity_type", "")
        human_entity_type = row.get("human_entity_type", "")
        human_type_dict = entity_type_dict.get(entity_type, {})
        count = human_type_dict.get(human_entity_type, 0)
        count += 1
        human_type_dict[human_entity_type] = count
        entity_type_dict[entity_type] = human_type_dict

        doc_count = row.get("doc_count", "")
        human_doc_count = row.get("human_doc_count", "")
        if entity_type == human_entity_type:
            correct_count_dict = entity_type_count_dict.get(entity_type, {})
            correct_count = correct_count_dict.get("correct_count", 0)
            if doc_count < human_doc_count:
                correct_count += doc_count
            else:
                correct_count += human_doc_count
            correct_count_dict["correct_count"] = correct_count
            entity_type_count_dict[entity_type] = correct_count_dict
        correct_count_dict = entity_type_count_dict.get(entity_type, {})
        gpt_count = correct_count_dict.get("gpt_count", 0)
        gpt_count += doc_count
        correct_count_dict["gpt_count"] = gpt_count
        entity_type_count_dict[entity_type] = correct_count_dict

        correct_count_dict = entity_type_count_dict.get(human_entity_type, {})
        human_count = correct_count_dict.get("human_count", 0)
        human_count += human_doc_count
        correct_count_dict["human_count"] = human_count
        entity_type_count_dict[human_entity_type] = correct_count_dict

    data_list = []
    for entity_type, human_type_dict in entity_type_dict.items():
        info = {"gpt": entity_type}
        info.update(human_type_dict)
        data_list.append(info)
    df_type_map_list = pd.DataFrame(data_list)
    df_type_map_list.to_excel(outfile)

    doc_count_entity_evaluate = []
    for entity_type, count_dict in entity_type_count_dict.items():
        if count_dict.get("gpt_count", 0) != 0:
            precision = round(count_dict.get("correct_count", 0) / count_dict["gpt_count"], 3)
        else:
            precision = 0
        if count_dict.get("human_count", 0) != 0:
            recall = round(count_dict.get("correct_count", 0)/ count_dict["human_count"], 3)
        else:
            recall = 0
        if (precision + recall) != 0:
            f1 = round(2 * precision * recall / (precision + recall), 3)
        else:
            f1 = 0
        info = {"entity_type": entity_type, "precision": precision, "recall": recall, "f1": f1,
                "support": count_dict.get("human_count",0)}
        info.update(count_dict)
        doc_count_entity_evaluate.append(info)
    df_type_map_list = pd.DataFrame(doc_count_entity_evaluate)
    df_type_map_list.to_excel(outfile_2)

    data = pd.read_excel(infile, keep_default_na=False)
    # entity_type = list(data["gpt_entity_true_type"])
    entity_type = list(data["entity_type"])
    answer = list(data["human_entity_type"])
    result = compute_metrics_sequence_classification(answer, entity_type)

    entity_eval_dict = {}
    for key, value in result.items():
        if "-" in key:
            key_split = key.split("-")
            entity_type = key_split[0]
            dim = key_split[1]
            dim_dict = entity_eval_dict.get(entity_type, {})
            dim_dict[dim] = value
            entity_eval_dict[entity_type] = dim_dict

    data_list = []
    for type, eval_dict in entity_eval_dict.items():
        info = {"entity_type": type}
        info.update(eval_dict)
        data_list.append(info)
    mydf = pd.DataFrame(data_list)
    mydf.to_excel(outfile_3)


def compute_metrics_sequence_classification(labels, predictions):
    report = sklearn_classification_report(
        labels, predictions, output_dict=True
    )
    labels_unique = set(labels)
    metrics = {
        "macro_f1": report["macro avg"]["f1-score"],
        "micro_f1": report["weighted avg"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "micro_precision": report["weighted avg"]["precision"],
        "micro_recall": report["weighted avg"]["recall"],
        "support": report["macro avg"]["support"]
    }
    for label, v1 in report.items():
        if label in labels_unique:
            for score_name, v2 in v1.items():
                metrics[f"{label}-{score_name}"] = v2
    return metrics


if __name__ == '__main__':
    # infile = "D:/project/topic_tagging/益禾堂/POC_2/打标结果expand.xlsx"
    # outfile = "D:/project/topic_tagging/益禾堂/POC_2/打标结果expand_add_human_statistic.xlsx"
    # data_statistic_2(infile, outfile)

    infile = "D:/project/topic_tagging/益禾堂/POC_2/打标结果expand_add_human_statistic.xlsx"
    outfile = "D:/project/topic_tagging/益禾堂/POC_2/打标结果expand_ner_mitrix.xlsx"
    outfile_2 = "D:/project/topic_tagging/益禾堂/POC_2/打标结果expand_ner_evaluate_doc_count.xlsx"
    outfile_3 = "D:/project/topic_tagging/益禾堂/POC_2/打标结果expand_ner_evaluate.xlsx"
    data_evaluate_2(infile, outfile, outfile_2, outfile_3)
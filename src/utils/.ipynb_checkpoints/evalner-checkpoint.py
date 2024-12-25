import pandas as pd


def checkIfOverlap(pred_val, true_val, text):
    rang_a = findBoundary(true_val, text)
    rang_b = findBoundary(pred_val, text)
    if len(rang_a) == 0 or len(rang_b) == 0:
        return False
    else:
        for i, j in rang_a:
            for k, m in rang_b:
                intersec = set(range(i, j)).intersection(set(range(k, m)))
                if len(intersec) > 0:
                    return True
                else:
                    return False


def findBoundary(val, text):
    res = []
    for i in range(0, len(text) - len(val) + 1):
        if text[i:i + len(val)] == val:
            res.append((i, i + len(val)))
    return res


def partial_match(pred_list, golden_list, text):
    precision_hit, precision_all = 0, 0
    recall_hit, recall_all = 0, 0
    
    for pred in pred_list:
        precision_all += 1
        for golden in golden_list:
            if checkIfOverlap(pred, golden, text):
                precision_hit += 1
                break
    
    for golden in golden_list:
        recall_all += 1
        for pred in pred_list:
            if checkIfOverlap(pred, golden, text):
                recall_hit += 1
                break

    return precision_hit, precision_all, recall_hit, recall_all


def evalNER(pred, golden, text, entity_type):
    p_hit, p_all, r_hit, r_all = 0, 0, 0, 0
    
    for pred_, golden_, text_ in zip(pred, golden, text):
    
        p_hit_, p_all_, r_hit_, r_all_ = partial_match(pred_, golden_, text_)

        p_hit += p_hit_
        p_all += p_all_
        r_hit += r_hit_
        r_all += r_all_
    
    p = p_hit / p_all
    r = r_hit / r_all
    
    result = {
        f"{entity_type}": {
            "precision": p,
            "recall": r,
            "f1": 2 * p * r / (p + r),
            "details": [p_hit, p_all, r_hit, r_all]
        }
    }
    return result


if __name__ == '__main__':

    ### dataframe
    # df_data = pd.read_excel("./...")
    # entity_type_list = ["brand", "product"]

    # result = {}
    # for entity_type in tqdm(entity_type_list):
    #     p_hit, p_all, r_hit, r_all = 0, 0, 0, 0
    #     for _, row in df_data.iterrows():
    #         golden_list = row[f'human_{entity_type}'] ## 改名称
    #         pred_list = row[f'pred_{entity_type}'] ## 改名称
    #         text = str(row['context_cleaned']) ## 改名称

    #         p_hit_, p_all_, r_hit_, r_all_ = partial_match(pred_list, golden_list, text)
    #         p_hit += p_hit_
    #         p_all += p_all_
    #         r_hit += r_hit_
    #         r_all += r_all_
        
    #     p = p_hit / p_all
    #     r = r_hit / r_all
    #     result.update(
    #         {
    #             f"{entity_type}": {
    #                 "precision": p,
    #                 "recall": r,
    #                 "f1": 2 * p * r / (p + r),
    #                 "details": [p_hit, p_all, r_hit, r_all]
    #             }
    #         }
    #     )

    # print(result)
    

    ### list of list
    pred = [["星冰乐", "摩卡咖啡"]]
    golden = [["抹茶星冰乐", "摩卡咖啡"]]
    text = ["星巴克的抹茶星冰乐真好喝！摩卡咖啡也不差。#星巴克"]

    eval_result = evalNER(pred, golden, text, "product")
    print(eval_result)
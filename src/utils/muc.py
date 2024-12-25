# !/usr/bin/env python

# -*- encoding: utf-8


import re
import pprint
from copy import deepcopy
from fuzzywuzzy import fuzz


def evaluate_one(prediction: list, ground_truth: list, text: str = None, tag_index: int = 1):
    """
    Evaluate single case
    Calculate detailed partial evaluation metric. See Evaluation of the SemEval-2013 Task 9.1
    :param prediction (list): [(slot tag, slot content), (slot tag, slot content), (slot tag, slot content), ...]
    :param ground_truth (list): [(slot tag, slot content), (slot tag, slot content), (slot tag, slot content), ...]
    :return: eval_results (dict)

    修改 recall 的计算方法：
    原始 recall = 所有匹配上的标签数量 / 所有匹配上的标签数量 + 遗漏的标签数量
    修正 recall = 已匹配 gt 标签数量 / gt 标签数量
    """
    prediction = deepcopy(prediction)
    ground_truth = deepcopy(ground_truth)
    # if no label and no prediction, reguard as all correct!
    if len(prediction) == 0 and len(ground_truth) == 0:
        eval_metics = {
            "correct": 1,
            "incorrect": 1,
            "partial": 1,
            "missed": 1,
            "spurius": 1,
            "precision": 1,
            "recall": 1,
            "f1_score": 1,
        }
        # evaluation metrics in total
        eval_results = {
                "strict": deepcopy(eval_metics),
                "exact": deepcopy(eval_metics),
                "partial": deepcopy(eval_metics),
                "type": deepcopy(eval_metics), 
                "similar": deepcopy(eval_metics)
            }

        return eval_results

    eval_metics = {
        "correct": 0,
        "incorrect": 0,
        "partial": 0,
        "missed": 0,
        "spurius": 0,
        "precision": 0,
    }
    # evaluation metrics
    eval_results = {
            "strict": deepcopy(eval_metics),
            "exact": deepcopy(eval_metics),
            "partial": deepcopy(eval_metics),
            "type": deepcopy(eval_metics), 
            "similar": deepcopy(eval_metics)
        }
    prediction_tmp = deepcopy(prediction)
    ground_truth_tmp = deepcopy(ground_truth)
    # 增加一个标志位，用于判断 gt 是否被匹配
    ground_truth_check = {
        'strict': {str(item):0 for item in ground_truth_tmp},
        'type': {str(item):0 for item in ground_truth_tmp},
        'exact': {str(item):0 for item in ground_truth_tmp},
        'partial': {str(item):0 for item in ground_truth_tmp},
        'similar': {str(item):0 for item in ground_truth_tmp}
    }
    prediction_check = {
        'strict': {str(item):0 for item in prediction_tmp},
        'type': {str(item):0 for item in prediction_tmp},
        'exact': {str(item):0 for item in prediction_tmp},
        'partial': {str(item):0 for item in prediction_tmp},
        'similar': {str(item):0 for item in prediction_tmp}
    }
    for item in prediction:
        # exact match, i.e. both entity boundary and entity type match
        # scenario 1
        flag1, matched_gts = check_Scenario1(item, ground_truth, tag_index=tag_index)
        for matched_gt in matched_gts:
            ground_truth_check['strict'][str(matched_gt)] += 1
            ground_truth_check['type'][str(matched_gt)] += 1
            ground_truth_check['exact'][str(matched_gt)] += 1
            ground_truth_check['partial'][str(matched_gt)] += 1
            ground_truth_check['similar'][str(matched_gt)] += 1

        if flag1:
            # 'strict' matching
            eval_results['strict']['correct'] += 1
            eval_results['type']['correct'] += 1
            eval_results['exact']['correct'] += 1
            eval_results['partial']['correct'] += 1
            eval_results['similar']['correct'] += 1

            prediction_check['strict'][str(item)] += 1
            prediction_check['exact'][str(item)] += 1
            prediction_check['partial'][str(item)] += 1
            prediction_check['type'][str(item)] += 1
            prediction_check['similar'][str(item)] += 1

            continue

        # partial match
        # scenario 5
        flag5, matched_gts = check_Scenario5(item, ground_truth, text, tag_index=tag_index)
        for matched_gt in matched_gts:
            # ground_truth_check['strict'][str(matched_gt)] += 1
            # ground_truth_check['exact'][str(matched_gt)] += 1
            ground_truth_check['partial'][str(matched_gt)] += 1
            ground_truth_check['type'][str(matched_gt)] += 1
            ground_truth_check['similar'][str(matched_gt)] += 1
            
        if flag5:
            # exact boundary matching
            eval_results['strict']['incorrect'] += 1
            eval_results['exact']['incorrect'] += 1
            eval_results['partial']['partial'] += 1
            eval_results['type']['correct'] += 1
            eval_results['similar']['correct'] += 1

            # prediction_check['strict'][str(item)] += 1
            # prediction_check['exact'][str(item)] += 1
            prediction_check['partial'][str(item)] += 1
            prediction_check['type'][str(item)] += 1
            prediction_check['similar'][str(item)] += 1

            continue

        # scenario 4: same pred value，entity type disagree
        flag4, matched_gts = check_Scenario4(item, ground_truth, tag_index = tag_index)
        for matched_gt in matched_gts:
            # ground_truth_check['strict'][str(matched_gt)] += 1
            ground_truth_check['exact'][str(matched_gt)] += 1
            ground_truth_check['partial'][str(matched_gt)] += 1
            # ground_truth_check['type'][str(matched_gt)] += 1
            ground_truth_check['similar'][str(matched_gt)] += 1

        if flag4:
            eval_results['strict']['incorrect'] += 1
            eval_results['exact']['correct'] += 1
            eval_results['partial']['correct'] += 1
            eval_results['type']['incorrect'] += 1
            eval_results['similar']['correct'] += 1

            # prediction_check['strict'][str(item)] += 1
            prediction_check['exact'][str(item)] += 1
            prediction_check['partial'][str(item)] += 1
            # prediction_check['type'][str(item)] += 1
            prediction_check['similar'][str(item)] += 1

            continue

        # scenario 6 : overlap exists, but tags disagree
        flag6, matched_gts = check_Scenario6(item, ground_truth, text, tag_index=tag_index)
        for matched_gt in matched_gts:
            # ground_truth_check['strict'][str(matched_gt)] += 1
            # ground_truth_check['exact'][str(matched_gt)] += 1
            ground_truth_check['partial'][str(matched_gt)] += 1
            # ground_truth_check['type'][str(matched_gt)] += 1
            ground_truth_check['similar'][str(matched_gt)] += 1

        if flag6:
            eval_results['strict']['incorrect'] += 1
            eval_results['exact']['incorrect'] += 1
            eval_results['partial']['partial'] += 1
            eval_results['type']['incorrect'] += 1
            eval_results['similar']['correct'] += 1

            # prediction_check['strict'][str(item)] += 1
            # prediction_check['exact'][str(item)] += 1
            prediction_check['partial'][str(item)] += 1
            # prediction_check['type'][str(item)] += 1
            prediction_check['similar'][str(item)] += 1

            continue
            
        # scenario 7: 类型相同，其余字段相似度大于阈值
        flag7, matched_gts = check_Similar(item, ground_truth, text, tag_index=tag_index)
        for matched_gt in matched_gts:
            # ground_truth_check['strict'][str(matched_gt)] += 1
            # ground_truth_check['exact'][str(matched_gt)] += 1
            # ground_truth_check['partial'][str(matched_gt)] += 1
            # ground_truth_check['type'][str(matched_gt)] += 1
            ground_truth_check['similar'][str(matched_gt)] += 1

        if flag7:
            eval_results['strict']['incorrect'] += 1
            eval_results['exact']['incorrect'] += 1
            eval_results['partial']['incorrect'] += 1
            eval_results['type']['incorrect'] += 1
            eval_results['similar']['partial'] += 1

            # prediction_check['strict'][str(item)] += 1
            # prediction_check['exact'][str(item)] += 1
            # prediction_check['partial'][str(item)] += 1
            # prediction_check['type'][str(item)] += 1
            prediction_check['similar'][str(item)] += 1

            continue

    # predictee not exists in golden standard
    # scenario 2: SPU, predicted entity not exists in golden, and no overlap on entity boundary
    # 由于 similar 的引入，仅靠边界是否重合不能作为判断 SPU 的条件。所以引入标志位。
    for pred_item in prediction_tmp:
        # count SPU
        if not prediction_check['strict'][str(pred_item)]:
            eval_results['strict']['spurius'] += 1
        if not prediction_check['exact'][str(pred_item)]:
            eval_results['exact']['spurius'] += 1
        if not prediction_check['partial'][str(pred_item)]:
            eval_results['partial']['spurius'] += 1
        if not prediction_check['type'][str(pred_item)]:
            eval_results['type']['spurius'] += 1
        if not prediction_check['similar'][str(pred_item)]:
            eval_results['similar']['spurius'] += 1

    for true_item in ground_truth_tmp:
        # 如果边界没有重叠就算 missed，加入 similar 之后不太合理，通过标签判断
        # flag, prediction_tmp = check_Scenario3(true_item, prediction_tmp, text, tag_index=tag_index)
            # count missing
            if not ground_truth_check['strict'][str(true_item)]:
                eval_results['strict']['missed'] += 1
            if not ground_truth_check['exact'][str(true_item)]:
                eval_results['exact']['missed'] += 1
            if not ground_truth_check['partial'][str(true_item)]:
                eval_results['partial']['missed'] += 1
            if not ground_truth_check['type'][str(true_item)]:
                eval_results['type']['missed'] += 1
            if not ground_truth_check['similar'][str(true_item)]:
                eval_results['similar']['missed'] += 1

    # calculate P, R, F1
    # POS = len(grount_truth)
    # ACT = len(prediction)

    for k, eval_ in eval_results.items():
        true_truth_item = len([count for item, count in ground_truth_check[k].items() if count > 0])
        true_pred_item = len([count for item, count in prediction_check[k].items() if count > 0])
        all_truth_items = len(ground_truth_check[k])
        all_pred_items = len(prediction_check[k])

# 忽略部分匹配、相似匹配和完全匹配，统一作为分子
        eval_["precision"] = true_pred_item / all_pred_items if all_pred_items > 0 else 0 
        eval_["recall"] = true_truth_item / all_truth_items if all_truth_items > 0 else 0
        eval_["f1_score"] = 2 * eval_["precision"] * eval_["recall"] / (eval_["precision"] + eval_["recall"]) \
            if eval_["precision"] + eval_["recall"] > 0 else 0
    return eval_results


def check_Scenario1(pred_item: str, grount_truth: list, tag_index: int = 1):
    # scenario 1: both entity type and entity boundary strictly match
    # COR_list = [1 for true_tag, true_val in grount_truth if true_tag == pred_tag and true_val == pred_val]
    flag, matched_items = False, []
    for true_item in grount_truth:
        if true_item[tag_index] == pred_item[tag_index] and true_item == pred_item:
            flag = True
            matched_items.append(true_item)
    return flag, matched_items


def check_Scenario5(pred_item: str, grount_truth: list, text: str, tag_index: int = 1):
    # scenario 5: same entity type and entity boundary overlap
    pred_str = pred_item[:tag_index] + pred_item[tag_index+1:]
    flag, matched_items = False, []
    for true_item in grount_truth:
        item_flag = True
        true_str = true_item[:tag_index] + true_item[tag_index+1:]

        if true_item[tag_index] != pred_item[tag_index]:
            item_flag = False
            continue

        for _index in range(len(pred_str)):
            if not checkIfOverlap(true_str[_index], pred_str[_index], text):
                item_flag = False
        
        if item_flag:
            flag = item_flag
            matched_items.append(true_item)
    return flag, matched_items


def check_Scenario2(pred_item: str, grount_truth: list, text: str, tag_index: int = 1):
    # scenario 2: SPU, predicted entity type not exists in golden, and no overlap on entity boundary
    flag, matched_items = True, []
    for true_item in grount_truth:
        if checkIfOverlap(true_item[0], pred_item[0], text) and checkIfOverlap(true_item[-1], pred_item[-1], text):
            # return False, grount_truth, (true_tag, true_val)
            flag = False
            matched_items.append(true_item)
    return flag, matched_items


def check_Scenario3(true_item: str, prediction: list, text: str, tag_index: int = 1):
    # Missed
    # scenario 3:entity boundary not overlap,  golden standard not exists in prediction
    true_str = true_item[:tag_index] + true_item[tag_index+1:]
    flag = False
    for pred_item in prediction:
        item_flag = True
        pred_str = pred_item[:tag_index] + pred_item[tag_index+1:]

        for _index in range(len(pred_str)):
            if not checkIfOverlap(true_str[_index], pred_str[_index], text):
                item_flag = False
        
        if item_flag:
            flag = item_flag
            prediction.remove(pred_item)
    return flag, prediction

def check_Scenario4(pred_item: str, grount_truth: list, tag_index: int = 1):
    # scenario 4: same pred value，entity type disagree
    flag, matched_items = False, []
    pred_str = pred_item[:tag_index] + pred_item[tag_index+1:]
    for true_item in grount_truth:
        true_str = true_item[:tag_index] + true_item[tag_index+1:]
        if true_item[tag_index] != pred_item[tag_index] and true_str == pred_str:
            flag = True
            matched_items.append(true_item)
    return flag, matched_items


def check_Scenario6(pred_item: str, grount_truth: list, text: str, tag_index: int = 1):
    # scenario 6: entity boundary overlap, entity type disagree
    flag, matched_items = False, []
    pred_str = pred_item[:tag_index] + pred_item[tag_index+1:]

    for true_item in grount_truth:
        item_flag = True
        true_str = true_item[:tag_index] + true_item[tag_index+1:]

        if true_item[tag_index] == pred_item[tag_index]:
            item_flag = False
            continue
        
        for _index in range(len(pred_str)):
            if not checkIfOverlap(true_str[_index], pred_str[_index], text):
                item_flag = False

        if item_flag:
            flag = item_flag
            matched_items.append(true_item)
    return flag, matched_items


def check_Similar(pred_item: str, grount_truth: list, text: str, tag_index: int = 1, similar_threshold=70):
    # scenario 7: 关系类型一致，且其余字段相似度大于阈值
    flag, matched_items = False, []
    pred_str = pred_item[:tag_index] + pred_item[tag_index+1:]

    for true_item in grount_truth:
        item_flag = True
        true_str = true_item[:tag_index] + true_item[tag_index+1:]

        if true_item[tag_index] != pred_item[tag_index]:
            item_flag = False
            continue

        for _index in range(len(pred_str)):
            if not calculatingSimilarity(true_str[_index], pred_str[_index], threshold=similar_threshold):
                item_flag = False
        
        if item_flag:
            flag = True
            matched_items.append(true_item)
    return flag, matched_items


def calculatingSimilarity(true_val, pred_val, threshold=70):
    # method 2: check if there are intersections (in surface string level)
    return True if fuzz.partial_ratio(true_val, pred_val) >= threshold else False
    # return True if fuzz.ratio(true_val, pred_val) >= threshold else False


def checkIfOverlap(true_val, pred_val, text):
    # method 1: check if index ranges have intersection (in index level)
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
    
    # method 2: check if there are intersections (in surface string level)
    # return not set(true_val).isdisjoint(pred_val)
    
    return False


def findBoundary(val, text):
    pattern = re.compile(re.escape(val))
    res = [(match.start(), match.end()) for match in pattern.finditer(text)]
    return res


def update_overall_result(total_res: dict, res_single: dict):
    for mode in res_single:
        total_res[mode]["precision"] += res_single[mode]["precision"]
        total_res[mode]["recall"] += res_single[mode]["recall"]
        total_res[mode]["f1_score"] += res_single[mode]["f1_score"]
        total_res[mode]["count"] += 1
    return total_res


def evaluate_all(predictions: list, golden_labels: list, texts: list, verbose=False, tag_index=0):
    """
    evaluate all cases
    :param predictions: list(list) [
                                [(slot tag, slot content), (slot tag, slot content), (slot tag, slot content), ...],
                                [(slot tag, slot content), (slot tag, slot content), (slot tag, slot content), ...],
                                [(slot tag, slot content), (slot tag, slot content), (slot tag, slot content), ...]
                            ]
    :param golden_labels: list(list)  [
                                [(slot tag, slot content), (slot tag, slot content), (slot tag, slot content), ...],
                                [(slot tag, slot content), (slot tag, slot content), (slot tag, slot content), ...],
                                [(slot tag, slot content), (slot tag, slot content), (slot tag, slot content), ...]
                            ]
    :param texts: list(str) [ text1, test2, text3, ...]
    :return: dict of results
    """
    assert len(predictions) == len(golden_labels) == len(
        texts), 'the counts of predictions/golden_labels/texts are not equal!'
    eval_metics = {"precision": 0, "recall": 0, "f1_score": 0, 'count': 0}
    # evaluation metrics in total
    total_results = {
            "strict": deepcopy(eval_metics),
            "exact": deepcopy(eval_metics),
            "partial": deepcopy(eval_metics),
            "type": deepcopy(eval_metics), 
            "similar": deepcopy(eval_metics), 
        }
    predictions_copy = deepcopy(predictions)
    golden_labels_copy = deepcopy(golden_labels)
    for i, (pred, gt, text) in enumerate(zip(predictions_copy, golden_labels_copy, texts)):
        one_result = evaluate_one(pred, gt, text, tag_index)
        if verbose:
            print('--'*6, 'sample_{:0>6}:'.format(i + 1))
            pprint.pprint(one_result)
        total_results = update_overall_result(total_results, one_result)

    print('\n', 'NER evaluation scores:')
    for mode, res in total_results.items():
        res['precision'] /= res['count']
        res['recall'] /= res['count']
        res['f1_score'] /= res['count']
        print(f"{mode:>8s} mode, Precision={res['precision']:<6.4f}, Recall={res['recall']:<6.4f}, F1:{res['f1_score']:<6.4f}")
    return total_results


if __name__ == '__main__':
    # # grount_truth = [('PER', 'John Jones'), ('PER', 'Peter Peters'), ('LOC', 'York')]
    # # prediction = [('PER', 'John Jones and Peter Peters came to York')]
    # # text = 'John Jones and Peter Peters came to York'
    # # # print(evaluate_one(prediction, grount_truth, text))
    # # # print(evaluate_one(prediction, grount_truth, text))

    # # evaluate_all([prediction] * 1, [grount_truth] * 1, [text] * 1, verbose=True)
    # # evaluate_all([prediction] * 1, [grount_truth] * 1, [text] * 1, verbose=True)

    # grount_truths = [
    # [('PER', 'John Jones'), ('PER', 'Peter Peters'), ('LOC', 'York')],
    # [('PER', 'John Jones'), ('PER', 'Peter Peters'), ('LOC', 'York')],
    # [('PER', 'John Jones'), ('PER', 'Peter Peters'), ('LOC', 'York')]
    # ]
    # # NER model prediction
    # predictions = [
    #     [('PER', 'John Jones and Peter Peters came to York')],
    #     [('LOC', 'John Jones'), ('PER', 'Peters'), ('LOC', 'York')],
    #     [('PER', 'John Jones'), ('PER', 'Peter Peters'), ('LOC', 'York')]
    # ]
    # # input texts
    # texts = [
    #     'John Jones and Peter Peters came to York',
    #     'John Jones and Peter Peters came to York',
    #     'John Jones and Peter Peters came to York'
    # ]
    # res = evaluate_all(predictions, grount_truths * 1, texts, verbose=True)
    # print(res)

    labels = [[
        ('PRODUCT', '北京号牌'),
        ('PERSON', '王贵彬'),
        ('ORGANIZATION', '海关'),
        ('PRODUCT', '京牌皮卡'),
        ('LOCATION', '北京A市')
    ]]
    
    preds = [[
        ('PRODUCT', '进口皮卡车型'),
        ('PRODUCT', '北京号牌'),
        ('PRODUCT', '北京号牌A'),
        ('PRODUCT', '北京号牌AB'),
        ('PRODUCT', '北京号牌ABC'),
        ('LOCATION', '北京'),
        ('LOCATION', '北京B市'),
        ('PRODUCT', '北京'),
        ('ORGANIZATION', '新京报记者'),
        ('PERSON', '王贵彬'),
        # ('ORGANIZATION', '海关'),
        ('PRODUCT', '京牌皮卡')
    ]]

    # preds = [[('LOCATION', '北京号牌'),
    #           ('PRODUCT', '北京'),
    #           ('LOCATION', '北京'),
    #         ('PRODUCT', '北京号牌A')]]
    contents = ['”\n\n这位交警表示，划归乘用车进口后，部分进口皮卡车型综合税率由原来的46%提高到95%。但这只是海关在税目中归类的改变，根据相关技术参数，这部分皮卡车在车辆类型里依旧属于多用途货车。在机动车登记中，这些车型依旧按照载货汽车进行登记，因此也必须遵守货车限行规定。只不过因为属于多用途货车，可以不用在驾驶室两侧喷涂总质量、不用粘贴反光标识等。\n\n一位刚刚上完号牌的车主为皮卡加装了后盖。摄影新京报记者王贵彬\n\n提示：\n\n京牌皮卡白天禁入五环违者扣三分\n\n根据北京A市货车限行规定，北京号牌ABC每天6时至23时，禁止进入五环路以内道路行驶']
    eval_result = evaluate_all(preds, labels, contents, verbose=False, tag_index=0)
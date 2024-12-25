# VKG-TRIAN-NER
## 配置文件：config.yaml

## 1、输入数据
### （1）VKG的实体表
    (必须存在entity、entity_type两列）
### （2）VKG的打标结果
    （输入数据的id列必须是“doc_id”,标题必须是“headline”， 内容必须是“content”， 实体打标结果（格式）： entities：{'input_entity_info_confirm': [], 'newly_extracted_entity_info': [{'entity': '香奈儿', 'entity_type': '品牌'}, {'entity': 'Gucci', 'entity_type': '品牌'}, {'entity': 'loewe', 'entity_type': '品牌'}, {'entity': 'mcm', 'entity_type': '品牌'}, {'entity': 'prada', 'entity_type': '品牌'}, {'entity': 'versace', 'entity_type': '品牌'}, {'entity': 'ysl', 'entity_type': '品牌'}, {'entity': '华伦天奴', 'entity_type': '品牌'}, {'entity': '迪奥', 'entity_type': '品牌'}, {'entity': '阿玛尼', 'entity_type': '品牌'}, {'entity': 'lv', 'entity_type': '品牌'}, {'entity': '路易威登', 'entity_type': '品牌'}]}）
### （3）业务提供的实体词表
    （必须存在entity、entity_type两列）

## 2、输出结果
    输出结果通过读取到config.yaml的output_dir_path、industry、date，组成完成的输出文件夹： {output_dir_path}/{industry}_{date}

## 3、处理过程

    需要注意的是一下的处理过程都依赖前一步的处理结果，如果前一步已经执行过可以注视掉该环节，例如stages里边的data_preprocess如果已经执行完成，则可以注释掉data_preprocess

```yaml
stages:
  - data_preprocess # 数据预处理
  - product_type_word_evaluation # 如果实体类型中存在品类实体，需要对品类实体进行单独验证
  - entity_word_eval # VKG创建的实体词验证
  - sentence_include_entity_eval # 句子中的实体验证
  - generate_dataset # 生成模型所需数据集
  - train_model # 模型训练
  - model_evaluate # 模型效果验证（添加关键词/不添加关键词分开进行验证）
# 每个过程会对应该过程中需要用的配置参数，例如该过程生成的文件名称或者位置、prompt对应的template_code等相关参数
```

## 3、启动命令
```shell
nohup python run.py --config="config.yaml">log.txt 2>&1 &
```


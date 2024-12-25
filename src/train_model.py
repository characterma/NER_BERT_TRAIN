import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pandas as pd
import numpy as np

import re,os
import torch
import datasets
import argparse
import yaml
import json
from loguru import logger
from sklearn.metrics import classification_report
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import default_data_collator, BertTokenizer, BertForTokenClassification
from src.utils.utils import generate_train_and_dev_dataloader, generate_test_dataloader, convert_label_to_entity
from src.utils import log_step


def train(model, device, train_loader, dev_loader, optimizer, n_epochs, num_labels, save_model_path):

    model.train()
    best_loss = 1e5
    train_loss_record, train_acc_record = [], []
    eval_loss_record, eval_acc_record = [], []

    for epoch in range(n_epochs):
        train_epoch_loss, train_epoch_acc = [], []
        for batch in tqdm(train_loader, desc=f"[ Train | {epoch+1}/{n_epochs} ]"):
            # print(batch)
            ids = batch['ids'].to(device, dtype=torch.long) # (batch_size, seq_len)
            masks = batch['masks'].to(device, dtype=torch.long) # (batch_size, seq_len)
            labels = batch['labels'].to(device, dtype=torch.long) # (batch_size, seq_len)
            
            # put into model
            output = model(input_ids=ids, attention_mask=masks, labels=labels, return_dict=True)
            loss, logits = output['loss'], output['logits'] # logits: (batchsize, seq_len, #labels)

            # get the predicted label
            pred_labels = torch.argmax(logits.view(-1, num_labels), dim=1)

            # update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add the loss to epoch loss
            train_epoch_loss.append(loss.item())

            # calculate the accuracy
            flatten_labels = labels.view(-1) # (batch_size * seq_len, )
            active_accuracy = masks.view(-1) == 1 # (batch_size * seq_len, )
            labels = torch.masked_select(flatten_labels, active_accuracy)
            pred_labels = torch.masked_select(pred_labels, active_accuracy)
            acc = ((pred_labels == labels).sum()).item()/len(labels)
            train_epoch_acc.append(acc)

        # compute the average acc and loss
        train_epoch_loss = sum(train_epoch_loss)/len(train_epoch_loss)
        train_epoch_acc = sum(train_epoch_acc)/len(train_epoch_acc)
        print(f'[ {"Train":<5} | {epoch+1}/{n_epochs} ] loss:{train_epoch_loss:6f}, acc:{train_epoch_acc:6f}')

        # compute validation loss
        valid_epoch_loss, valid_epoch_acc = [], []
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"[ Valid | {epoch+1}/{n_epochs} ]"):
                
                ids = batch['ids'].to(device, dtype = torch.long)
                masks = batch['masks'].to(device, dtype = torch.long)
                labels = batch['labels'].to(device, dtype = torch.long)
                
                # put into model
                output = model(input_ids=ids, attention_mask=masks, labels=labels, return_dict=True)
                loss, logits = output['loss'], output['logits'] # logits: (batchsize, seq_len, #labels)
                
                # add the loss to epoch_loss
                valid_epoch_loss.append(loss.item())

                # get the predicted label
                pred_labels = torch.argmax(logits.view(-1, num_labels), dim=1)

                # calculate the accuracy
                flatten_labels = labels.view(-1) # (batch_size * seq_len, )
                active_accuracy = masks.view(-1) == 1 # (batch_size * seq_len, )
                labels = torch.masked_select(flatten_labels, active_accuracy)
                pred_labels = torch.masked_select(pred_labels, active_accuracy)
                acc = ((pred_labels == labels).sum()).item()/len(labels)
                valid_epoch_acc.append(acc)
        
        # compute the average acc and loss
        valid_epoch_loss = sum(valid_epoch_loss)/len(valid_epoch_loss)
        valid_epoch_acc = sum(valid_epoch_acc)/len(valid_epoch_acc)
        print(f'[ {"Valid":<5} | {epoch+1}/{n_epochs} ] loss:{valid_epoch_loss:6f}, acc:{valid_epoch_acc:6f}')  

        if valid_epoch_loss < best_loss:
            best_loss = valid_epoch_loss
            # save the model
            torch.save(model.state_dict(), f"{save_model_path}")
            print(f'[ {"Info":<5} | {epoch+1}/{n_epochs} ] Model saved.') 

        train_loss_record.append(train_epoch_loss)
        train_acc_record.append(train_epoch_acc)
        eval_loss_record.append(valid_epoch_loss)
        eval_acc_record.append(valid_epoch_acc)
    
    return train_loss_record, train_acc_record, eval_loss_record, eval_acc_record



def inference(model, device, tokenizer, ids_to_labels, test_dir_path, batch_size, entity_types):
    """ make prediction
        model: a trained model
        tokenizer: a tokenizer
    """
    test_loader = generate_test_dataloader(test_dir_path, batch_size)
    
    model.eval()

    inference_results = []
    i = 0
    output_dict = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="inferencing"):
            docids = batch['docid']
            tokenized_content = batch['tokenized_content']
            ids = batch['ids'].to(device, dtype=torch.long)
            masks = batch['masks'].to(device, dtype=torch.long)
            # if i == 0:
            #     i+=1
            #     print(f"ids: {ids.shape}, masks: {masks.shape}")
            # put into model
            output = model(input_ids=ids, attention_mask=masks, return_dict=True)
            logits = output['logits'] # logits: (batchsize, seq_len, #labels)
            
            pred_labels = torch.argmax(logits, dim=2).cpu().tolist() # (batchsize, seq_len)
            pred_labels = [[ids_to_labels[j] for j in i] for i in pred_labels]
            
            tokens = [tokenizer.convert_ids_to_tokens(i) for i in ids.cpu().squeeze().tolist()]

            for d, t, t_c, p_l in zip(docids, tokens, tokenized_content, pred_labels):
                p_l_temp = []
                for pair in zip(t, p_l):
                    if pair[0] in ['[CLS]', '[SEP]', '[PAD]']:
                        continue
                    else:
                        p_l_temp.append(pair[1])

                output_dict.append({"doc_id": d, "tokenized_content": t_c, "label_pred": p_l_temp})

    print(len(output_dict))
    df_output = pd.DataFrame(output_dict)
    # print(len(df_output))
    df_output = convert_label_to_entity(df_output, entity_types)
    return df_output

@log_step(15)        
def train_model(config_reader):
    device_name = config_reader.get_value("model")['device']
    entity_types = config_reader.get_value("entity_type", [])
    output_dir_path = f"{config_reader.get_value('output_dir_path')}/{config_reader.get_value('industry')}_{config_reader.get_value('date')}"
    
    trainset_path = f"{output_dir_path}/{config_reader.get_value('generate_dataset')['train_data']}"
    testset_path = f"{output_dir_path}/{config_reader.get_value('generate_dataset')['test_data']}"
    validset_path = f"{output_dir_path}/{config_reader.get_value('generate_dataset')['valid_data']}"
    test_data_predict_file_path = f"{output_dir_path}/{config_reader.get_value('train_model')['test_data_predict_file']}"
    
    # 训练的模型保存的位置
    save_model_dir = f"{output_dir_path}/{config_reader.get_value('train_model')['save_model_dir']}" 
    save_model_path = f"{save_model_dir}/{config_reader.get_value('train_model')['model_name']}"
    os.makedirs(save_model_dir, exist_ok=True)
    # 预训练模型的缓存位置
    model_cache_dir = config_reader.get_value("model")['pretrain_model_dir']
    pretrained_model_name = config_reader.get_value("model")['pretrained_model_name']
    lr = config_reader.get_value("model")['lr']
    do_training = config_reader.get_value("model")['do_training']
    do_prediction = config_reader.get_value("model")['do_prediction']
    n_epochs = config_reader.get_value("model")['n_epochs']
    batch_size = config_reader.get_value("model")['batch_size']
    

    if entity_types == []:
        logger.error(f"entity_types is null in config file, please check it.{entity_types}")
        raise Exception("entity_types is null in config file, please check it")
        
    if os.path.exists(model_cache_dir) and os.listdir(model_cache_dir).__len__() > 1:
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name, cache_dir=model_cache_dir)
        logger.info(f"save success, model configs cache dir: {model_cache_dir}, model name: {pretrained_model_name}")

    else:
        raise Exception(f"{pretrained_model_name} no cache, Download the model to {model_cache_dir}")
                              
        
    labels_to_ids = json.load(open(os.path.join(output_dir_path, "labels_to_ids.json"), 'r'))
    ids_to_labels = json.load(open(os.path.join(output_dir_path, "ids_to_labels.json"), 'r'))
    ids_to_labels = {int(k): v for k, v in ids_to_labels.items()}
    logger.info(f"labels_to_ids: {labels_to_ids}, ids_to_labels: {ids_to_labels}")
    num_labels = len(ids_to_labels) 
    device = device_name if cuda.is_available() else 'cpu'
    
    # define the tokenizer and model
    model = BertForTokenClassification.from_pretrained(pretrained_model_name, num_labels=num_labels)
    model.to(device)

    # define the optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=float(lr))
    if do_training:
        train_loader, valid_loader = generate_train_and_dev_dataloader(trainset_path, validset_path, batch_size)
        # train                               
        train(
            model=model, 
            device=device,
            train_loader=train_loader, 
            dev_loader=valid_loader, 
            optimizer=optimizer, 
            n_epochs=n_epochs, 
            num_labels=num_labels, 
            save_model_path=save_model_path
        )

    
    if do_prediction:
        model.load_state_dict(torch.load(save_model_path))
        df_test = inference(
            model=model, 
            device=device,
            tokenizer=tokenizer, 
            ids_to_labels=ids_to_labels, 
            test_dir_path=testset_path, 
            batch_size=batch_size,
            entity_types=entity_types
        )
        df_test.to_pickle(test_data_predict_file_path)


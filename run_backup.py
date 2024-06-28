import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pandas as pd
import numpy as np

import re
import torch
import datasets
import argparse
import yaml


from sklearn.metrics import classification_report
from torch import cuda
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import default_data_collator, BertTokenizer, BertForTokenClassification
from utils.utils import *
from torchcrf import CRF






def train(args, model, train_loader, dev_loader, optimizer):

    model.train()
    best_loss = 1e5
    train_loss_record, train_acc_record = [], []
    eval_loss_record, eval_acc_record = [], []

    for epoch in range(args.n_epochs):
        train_epoch_loss, train_epoch_acc = [], []
        for batch in tqdm(train_loader, desc=f"[ Train | {epoch+1}/{args.n_epochs} ]"):
            # print(batch)
            ids = batch['ids'].to(device, dtype=torch.long) # (batch_size, seq_len)
            masks = batch['masks'].to(device, dtype=torch.long) # (batch_size, seq_len)
            labels = batch['labels'].to(device, dtype=torch.long) # (batch_size, seq_len)
            
            # put into model
            # output = model(input_ids=ids, attention_mask=masks, labels=labels, return_dict=True)
            output = model(input_ids=ids, attention_mask=masks, labels=labels)
            loss, logits = output['loss'], output['logits'] # logits: (batchsize, seq_len, #labels)

            # get the predicted label
            pred_labels = torch.argmax(logits.view(-1, args.num_labels), dim=1)

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
        print(f'[ {"Train":<5} | {epoch+1}/{args.n_epochs} ] loss:{train_epoch_loss:6f}, acc:{train_epoch_acc:6f}')

        # compute validation loss
        valid_epoch_loss, valid_epoch_acc = [], []
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"[ Valid | {epoch+1}/{args.n_epochs} ]"):
                
                ids = batch['ids'].to(device, dtype = torch.long)
                masks = batch['masks'].to(device, dtype = torch.long)
                labels = batch['labels'].to(device, dtype = torch.long)
                
                # put into model
                output = model(input_ids=ids, attention_mask=masks, labels=labels, return_dict=True)
                loss, logits = output['loss'], output['logits'] # logits: (batchsize, seq_len, #labels)
                
                # add the loss to epoch_loss
                valid_epoch_loss.append(loss.item())

                # get the predicted label
                pred_labels = torch.argmax(logits.view(-1, args.num_labels), dim=1)

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
        print(f'[ {"Valid":<5} | {epoch+1}/{args.n_epochs} ] loss:{valid_epoch_loss:6f}, acc:{valid_epoch_acc:6f}')  

        if valid_epoch_loss < best_loss:
            best_loss = valid_epoch_loss
            # save the model
            torch.save(model.state_dict(), f"{args.prefix_path}/model.pth")
            print(f'[ {"Info":<5} | {epoch+1}/{args.n_epochs} ] Model saved.') 

        train_loss_record.append(train_epoch_loss)
        train_acc_record.append(train_epoch_acc)
        eval_loss_record.append(valid_epoch_loss)
        eval_acc_record.append(valid_epoch_acc)
    
    return train_loss_record, train_acc_record, eval_loss_record, eval_acc_record



def inference(arg, model, tokenizer):
    """ make prediction
    Args:
        arg: arguments
        model: a trained model
        tokenizer: a tokenizer
    """
    _, ids_to_labels = define_labels(arg)
    test_loader = generate_test_dataloader(arg)
    
    model.eval()

    inference_results = []

    output_dict = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="inferencing"):
            docids = batch['docid']
            tokenized_content = batch['tokenized_content']
            ids = batch['ids'].to(device, dtype=torch.long)
            masks = batch['masks'].to(device, dtype=torch.long)

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

                output_dict.append({"docid": d, "tokenized_content": t_c, "label_pred": p_l_temp})

    print(len(output_dict))
    df_output = pd.DataFrame(output_dict)
    # print(len(df_output))
    df_output = convert_label_to_entity(df_output, arg)
    
    return df_output

class BERTCRF(nn.Module):
    def __init__(self, args):
        super().__init__()
        # bert token classification
        self.bert_classification = BertForTokenClassification.from_pretrained(args.pretrained_model, num_labels=args.num_labels)
        self.crf = CRF(num_tags=args.num_labels, batch_first=True)
    
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert_classification(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits
        # print(logits.shape, logits)
        if labels is not None:
            loss = self.crf(logits, labels, mask=attention_mask.byte(), reduction='mean')
            # print(loss)
            return {"loss": loss, "logits": logits}
        else:
            return self.crf.decode(logits, attention_mask.byte())



def main(args):
    labels_to_ids, ids_to_labels = define_labels(args)

    # define the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)    
    model = BERTCRF(args)
    model.to(device)

    # define the optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=float(args.lr))

    if args.do_training:
        # generate train and dev dataloader
        train_loader, valid_loader = generate_train_and_dev_dataloader(args)

        # train 
        train(args, model, train_loader, valid_loader, optimizer)
    
    if args.do_prediction:
        model.load_state_dict(torch.load(f"{args.prefix_path}/model.pth"))
        df_test = inference(args, model, tokenizer)
        df_test.to_pickle(f"{args.prefix_path}/test_data_predict_result.pkl")


if __name__ == '__main__':

    device = 'cuda' if cuda.is_available() else 'cpu'
    print(device)

    with open("config.yaml", 'r') as config_file:
        config = yaml.safe_load(config_file)

    parser = argparse.ArgumentParser(description='') 
    parser.set_defaults(**config)
    args = parser.parse_args(args = [])

    parser.add_argument("--prefix_path", type=str, default=f"./experiments/{args.date}_{args.experiment_name}_{args.mode}")
    args = parser.parse_args(args = [])

    # construct the save path
    os.makedirs(args.prefix_path, exist_ok=True)

    main(args)
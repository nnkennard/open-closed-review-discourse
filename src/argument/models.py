import torch
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import random
from sklearn.metrics import classification_report

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='micro')

def report_multiclass(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return classification_report(labels_flat, preds_flat)

def evaluate(model, device, dataloader_val):
    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals

def models(choice, df):
    if choice == 'Bert':
        model_choice = 'bert-base-uncased'
    else:
        model_choice = 'allenai/scibert_scivocab_uncased'

    tokenizer = BertTokenizer.from_pretrained(model_choice,
                                              do_lower_case=True)

    encoded_data_train = tokenizer.batch_encode_plus(
        df[df.data_type == 'train'].reviews.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    encoded_data_test = tokenizer.batch_encode_plus(
        df[df.data_type == 'test'].reviews.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(df[df.data_type == 'train'].label.values)

    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']
    labels_test = torch.tensor(df[df.data_type == 'test'].label.values)

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)
    print(len(dataset_train))
    print(len(dataset_test))

    model = BertForSequenceClassification.from_pretrained(model_choice,
                                                          num_labels=6,
                                                          output_attentions=False,
                                                          output_hidden_states=False)

    batch_size = 16
    dataloader_train = DataLoader(dataset_train,
                                  sampler=RandomSampler(dataset_train),
                                  batch_size=batch_size)

    dataloader_test = DataLoader(dataset_test,
                                 sampler=SequentialSampler(dataset_test),
                                 batch_size=batch_size)
    optimizer = AdamW(model.parameters(),
                      lr=1e-5,
                      eps=1e-8)
    epochs = 1
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(dataloader_train) * epochs)
    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in tqdm(range(1, epochs + 1)):

        model.train()

        loss_train_total = 0

        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        for batch in progress_bar:
            model.zero_grad()

            batch = tuple(b.to(device) for b in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2],
                      }

            outputs = model(**inputs)

            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})

        torch.save(model.state_dict(), f'models/finetuned_{choice}_epoch_{epoch}.model')

        tqdm.write(f'\nEpoch {epoch}')

        loss_train_avg = loss_train_total / len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')

        test_loss, predictions, true_vals = evaluate(model, device, dataloader_test)
        test_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'Test loss: {test_loss}')
        tqdm.write(f'F1 Score (Micro): {test_f1}')
        report = report_multiclass(predictions, true_vals)
        print('\n Classification Report')
        print('======================================================')
        print('\n', report)

def get_models():
    data_df = pd.read_csv('dataset.csv')

    # Bert model
    print('====================Bert======================')
    models('Bert', data_df)

    # SciBert model
    print('====================SciBert======================')
    models('SciBert', data_df)

if __name__ == '__main__':
    get_models()
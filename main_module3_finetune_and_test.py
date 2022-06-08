from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm 
tqdm.pandas()
import pandas as pd
import os, torch, copy, shutil
import numpy as np

from transformers_helper import load_tokenizer_and_model
from CustomDataset import CustomDataset, encode_for_inference
import finetuning_classification, reports

root_dir = '/data/finetune/jihye_data/fin_tweet_spam' # 43 server
model_save_dir = os.path.join(root_dir, 'temp')
train_filepaths = sorted(glob(os.path.join(root_dir, 'data', 'train_*.csv')))
test_filepath = os.path.join(root_dir, 'data', 'test_10000.csv')

model_name_or_dirs = sorted(glob(os.path.join(root_dir, 'pt_10-Ks_*', 'item*')))
model_name_or_dirs.extend(['bert-base-uncased', 'nghuyong/ernie-2.0-en', 'ProsusAI/finbert'])

save_dir_format = os.path.join(root_dir, 'result_{}_{}')

def do_prepare_data(relabel_dict, filepath):
    df = pd.read_csv(filepath)[['text', 'label']]
    print('Loaded {}'.format(filepath))
    df['label'] = df['label'].apply(lambda x: relabel_dict[x])
    return df
    
def start_finetuning(model_name_or_dir, num_classes, train_texts, train_labels, val_texts, val_labels, save_dir):
    tokenizer, model = load_tokenizer_and_model(model_name_or_dir, num_classes=num_classes, mode='classification')
    
    print('Getting data..\n')
    train_dataset = CustomDataset(tokenizer, train_texts, train_labels)
    val_dataset = CustomDataset(tokenizer, val_texts, val_labels)
    
    finetuning_classification.train(model, train_dataset, val_dataset, save_dir)
    tokenizer.save_pretrained(save_dir)

def start_test(device, model_name_or_dir, df, save_dir, postfix=''):
    tokenizer, model = load_tokenizer_and_model(model_name_or_dir, mode='classification')
    model = model.to(device)
    
    print('Inferencing..\n')
    df['predicted_label'] = df['text'].progress_apply(lambda x: finetuning_classification.inference(model, *encode_for_inference(device, tokenizer, x)))
    
    # Save results
    df['correct'] = df.apply(lambda x: x.true_label==x.predicted_label, axis=1)
    labels, preds = df.true_label, df.predicted_label
    accuracy = len(df[df['correct']==True]) / len(df)

    csv_filepath = os.path.join(save_dir, 'results_{}.csv'.format(postfix))
    df.to_csv(csv_filepath, index=False)
    
    report_filepath = os.path.join(save_dir, 'classification_report_{}.csv'.format(postfix))
    reports.create_classification_report(labels, preds, accuracy, report_filepath)
    
    confusion_filepath = os.path.join(save_dir, 'confusion_matrix_{}.csv'.format(postfix))
    reports.create_confusion_matrix(labels, preds, confusion_filepath)

if __name__ == '__main__':        

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    relabel_dict = {'human':0, 'bot':1}
    num_classes = len(relabel_dict)    
    


    for train_filepath in train_filepaths:
        ##### fine-tuning #####
        source_df = do_prepare_data(relabel_dict, train_filepath)
        train_df = source_df.iloc[:int(len(source_df)*0.8)]
        val_df = source_df.iloc[int(len(source_df)*0.8):]
        train_texts, val_texts = train_df['text'].values, val_df['text'].values
        train_labels, val_labels = train_df['label'].values, val_df['label'].values
        
        for model_name_or_dir in model_name_or_dirs:
            masking_mode = os.path.basename(os.path.dirname(model_name_or_dir)).replace('pt_10-Ks_','')
            data_mode = os.path.basename(model_name_or_dir)
            save_dir = save_dir_format.format(masking_mode, data_mode)
            if not os.path.exists(save_dir): os.makedirs(save_dir)
                
            start_finetuning(model_name_or_dir, num_classes, train_texts, train_labels, val_texts, val_labels, model_save_dir)

            ##### test #####
            test_df = do_prepare_data(relabel_dict, test_filepath)
            test_df.rename(columns = {'label' : 'true_label'}, inplace = True)
            start_test(device, model_save_dir, test_df, save_dir, \
                       postfix='train_{}'.format(os.path.basename(train_filepath.split('_')[-1].replace('.csv',''))))

            # 메모리 확보를 위해 한번 프로세싱이 끝난 폴더는 삭제
            try: shutil.rmtree(model_save_dir)
            except OSError as e: print("Error: %s - %s." % (e.filename, e.strerror))
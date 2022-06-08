import pandas as pd
import os, re
from tqdm import tqdm
tqdm.pandas()

from Preprocessor import Preprocessor

root_dir = '/data/finetune/jihye_data/fin_tweet_spam'
scraped_10k_filepath = os.path.join(root_dir, 'pt_data', 'Scraped_5616_parsed.csv')
save_10k_filepath = os.path.join(root_dir, 'pt_data', 'post_{}.txt')

def create_txt_for_post_training(preprocessor, docs, cik_list, save_filepath, num_of_duplicates=1):    
    with open(save_filepath, 'w') as output_file:
        for idx in range(num_of_duplicates): 
            print('[{}/{}]..'.format(idx+1, num_of_duplicates), end=' ')
            
            for doc, cik in tqdm(zip(docs, cik_list)):
                subnames, fullname = preprocessor.subnames_of_company_name(cik)
                # 원 텍스트에 기업명이 등장한 케이스만 post-training에 사용
                if fullname == '':
                    continue
                
                for sent in preprocessor.sentence_splitter.tokenize(doc):
                    sent = preprocessor.clean(sent)
                    sent = preprocessor.replace_subnames_to_target(sent, subnames, fullname)
                    
                    # 원 텍스트에 기업명이 등장한 케이스만 post-training에 사용
                    if fullname in sent: 
                        sent = preprocessor.trim(sent)
                        output_file.write('{}\n\n'.format(sent))
                            
        output_file.write('[EOD]')
    print(f'Created {save_filepath}')

if __name__ == '__main__': 
    preprocessor = Preprocessor()

    df = pd.read_csv(scraped_10k_filepath)
    df = df.astype({"CIK": str}, errors='raise') 
    
    df['itemAll'] = df.progress_apply(lambda x: '. '.join([x['item1_business'], x['item1a_risk'], x['item7_mda']]), axis=1)
    
    for colname in ['item1_business', 'item1a_risk', 'item7_mda', 'itemAll']:
        texts = df[(df[colname]!='Something went wrong!')][colname].values
        cik_list = df[(df[colname]!='Something went wrong!')]['CIK'].values
        create_txt_for_post_training(preprocessor, texts, cik_list, save_10k_filepath.format(colname))
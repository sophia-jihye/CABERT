from nltk.tokenize.punkt import PunktSentenceTokenizer
from transformers import AutoTokenizer
import pandas as pd
import os, re, random

root_dir = '/data/jihye_data/CABERT'

def decision(percent=50):
    return random.randrange(100) < percent

class Preprocessor():
    def __init__(self, root_dir=root_dir):
        companies_filepath = os.path.join(root_dir, 'companies_stockmapper.csv')
        company_df = pd.read_csv(companies_filepath)
        company_df = company_df.astype({"CIK": str}, errors='raise')
        self.company_names = list(company_df.Name.unique())

        self.convert_cik_to_name = dict(zip(company_df.CIK,company_df.Name)) # treats lower-cased words
        self.convert_cik_to_ticker = dict(zip(company_df.CIK,company_df.Ticker)) # treats lower-cased words
        self.sentence_splitter = PunktSentenceTokenizer()
        self.removal_list =  "‘, ’, ◇, ‘, ”,  ’, ', ·, \“, ·, △, ●,  , ■, (, ), \", >>, `, /, -,∼,=,ㆍ<,>, .,?, !,【,】, …, ◆,%"
        self.convert_ticker_to_name = dict(zip('$'+company_df.Ticker,company_df.Name)) # treats lower-cased words

    # Lowercase 처리 함
    def clean(self, sent):
        sent = sent.translate(str.maketrans(self.removal_list, ' '*len(self.removal_list)))
        sent = re.sub("\s+", " ", sent)
        sent = sent.lower()
        return sent

    def subnames_of_company_name(self, cik): 
        fullname = self.convert_cik_to_name.setdefault(cik, '') # 'amazon com inc' | 'abbott laboratories'
        ticker = self.convert_cik_to_ticker.setdefault(cik, '') # amzn
        
        strings = fullname.split(' ')
        if len(strings) <= 2:        
            subnames = [' {} '.format(fullname)] # [' abbott laboratories ']
        else:
            # [' amazon ', ' amazon com ', ' amazon com inc ']
            subnames = [' {} '.format(' '.join(strings[:i+1]).lower()) for i in range(len(strings))] 

        subnames.reverse() # [' amazon com inc ', ' amazon com ', ' amazon ']

        subnames.extend(['we ', 'the company ', ' {} '.format(ticker)])    
        if '  ' in subnames: 
            subnames.remove('  ') # [' amazon com inc ', ' amazon com ', ' amazon ', 'we ', 'the company ']
        return subnames, fullname

    def replace_subnames_to_target(self, sent, subnames, target):
        for sub in subnames:
            if sub in sent:
                sent = sent.replace(sub, ' {} '.format(target))
        return sent
    
    def trim(self, sent):
        sent = re.sub("\s+", " ", sent)
        sent = sent.strip()
        return sent
    
    def convert_ticker_with_cashtag_to_name(self, text):
        return re.sub(r'\$([a-zA-Z.-]+)', lambda m: self.convert_ticker_to_name.setdefault(m.group(0), m.group(0)), text.lower())
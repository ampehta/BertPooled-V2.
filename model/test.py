from transformers import BertTokenizer
from model import PooledBert
tokenizer = BertTokenizer.from_pretrained("kykim/bert-kor-base")

sample= ['나는 공부를 하고 있다.','노래를 들으면서 공부를 하고 있는데 너무 좋다.','나는 기계학습을 수강 철회하고 이걸 하고 있어',\
         '살 빼야되는데 배가 너무 고파','이상한 단어를 위한 예시 문장 쀌케케']

bert = PooledBert(tokenizer,args)
a, b = bert(sample)
bert.summary()
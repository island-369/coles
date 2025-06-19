import json
from transformers import BertTokenizer

from feat.categorical_processor import CategoricalProcessor
from feat.numerical_processor import NumericalProcessor
from feat.text_processor import TextlProcessor
from feat.time_processor import TimelProcessor


# BERT分词器的初始化
def prepare_bert_tokenizer():
    bert_tokenizer = None
    def lazy_init_bert_tokenizer():
        nonlocal bert_tokenizer
        if not bert_tokenizer:
            bert_tokenizer = BertTokenizer.from_pretrained('/home/xqy/YL-model 2/bert-base-chinese', cache_dir='./')
        return bert_tokenizer
    return lazy_init_bert_tokenizer

get_bert_tokenizer = prepare_bert_tokenizer()

def read_config():
    with open('config_universal_example.json', 'r', encoding='utf8') as f:
        config = json.load(f)
    return config

def init_config():
    config = read_config()
    for k, v in config['feature_config'].items():
        if v['type'] == 'text':
            if v['model'] == 'bert':
                bert_tokenizer = get_bert_tokenizer()
                v['vocab_size'] = bert_tokenizer.vocab_size
                v['pad_idx'] = bert_tokenizer.pad_token_id
                v['bos_token_id'] = bert_tokenizer.cls_token_id
                v['eos_token_id'] = bert_tokenizer.sep_token_id
                v['func'] = bert_tokenizer
    for key, v in config['feature_config'].items():
        if v['type']=='categorical':
            v['idx_map'] = {name:i for i,name in enumerate(v['choices'])}
        if v['type']=='text':
            bert_tokenizer = get_bert_tokenizer()
            v['pad_idx'] = v.get('pad_idx', bert_tokenizer.pad_token_id)
    return config

PROCESSOR_MAP = {
    "numerical": NumericalProcessor,
    "categorical": CategoricalProcessor,
    "text": TextlProcessor,
    "time": TimelProcessor
}

#按类型获取并构造对应处理器
def get_processor(cfg):
    feat_type = cfg['type']
    processor = PROCESSOR_MAP[feat_type]
    return processor(cfg)
import omegaconf
import hydra
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from pl_data_modules import BasePLDataModule
from pl_modules import BasePLModule
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from pytorch_lightning.callbacks import LearningRateMonitor
from generate_samples import GenerateTextSamplesCallback

def read_archive_tokens(token_file):
    with open(token_file, 'r') as reader:
        all_terms = reader.readlines()
    
    all_tokens = []
    for term in all_terms:
        token = term.strip()
        if ":" in token:
            token = token.split(":")[-1].strip()

        if len(token) > 0:
            all_tokens.append(token)

    return all_tokens

def train(conf: omegaconf.DictConfig) -> None:
    pl.seed_everything(conf.seed)

    config = AutoConfig.from_pretrained(
        conf.config_name if conf.config_name else conf.model_name_or_path,
        decoder_start_token_id = 0,
        early_stopping = False,
        no_repeat_ngram_size = 0,
        dropout=conf.dropout,
        forced_bos_token_id=None,
    )
    config.save_pretrained(conf.save_dir)

    tokenizer_kwargs = {
        "use_fast": conf.use_fast_tokenizer,
        "additional_special_tokens": ['<obj>', '<subj>', '<triplet>', '<head>', '</head>', '<tail>', '</tail>']
    }

    tokenizer = AutoTokenizer.from_pretrained(
        conf.tokenizer_name if conf.tokenizer_name else conf.model_name_or_path,
        **tokenizer_kwargs
    )

    if conf.dataset_name.split('/')[-1] == 'conll04_typed.py':
        tokenizer.add_tokens(['<peop>', '<org>', '<other>', '<loc>'], special_tokens = True)
    if conf.dataset_name.split('/')[-1] == 'nyt_typed.py':
        tokenizer.add_tokens(['<loc>', '<org>', '<per>'], special_tokens = True)
    if conf.dataset_name.split('/')[-1] == 'docred_typed.py':
        tokenizer.add_tokens(['<loc>', '<misc>', '<per>', '<num>', '<time>', '<org>'], special_tokens = True)
    if conf.dataset_name.split('/')[-1] == 'archive.py':
        archive_terms = read_archive_tokens(conf.token_terms)
        tokenizer.add_tokens(archive_terms, special_tokens = True)

    tokenizer.save_pretrained(conf.save_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        conf.model_name_or_path,
        config=config,
    )

    model.resize_token_embeddings(len(tokenizer))
    print("Resized model for", len(tokenizer), "tokens")
    pl_data_module = BasePLDataModule(conf, tokenizer, model)
    pl_module = BasePLModule(conf, config, tokenizer, model)

    model = pl_module.load_from_checkpoint(checkpoint_path = conf.checkpoint_path, config = config, tokenizer = tokenizer, model = model)
    model.model.save_pretrained('../model/archive_tuned')
    model.tokenizer.save_pretrained('../model/archive_tuned')

@hydra.main(config_path='../conf', config_name='root')
def main(conf: omegaconf.DictConfig):
    train(conf)

if __name__ == '__main__':
    main()
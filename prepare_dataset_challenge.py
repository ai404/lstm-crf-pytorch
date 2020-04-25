import pandas as pd
import numpy as np
from parameters import *

TAG_TO_IX = {
    PAD:PAD_IDX, 
    SOS:SOS_IDX, 
    EOS:EOS_IDX, 
    UNK:UNK_IDX,
    "B":4, 
    "E":5, 
    "I":6, 
    "O":7
}

class Dataset:
    def __init__(self, fname, tokenizer, padding_len = 140, dropna=True):
        
        self.data = pd.read_csv(fname)
        if dropna:
            self.data.dropna(inplace=True)
            self.data.reset_index(inplace=True, drop=True)
        self.tokenizer = tokenizer
        self.threshold_select = 0.
        self.padding_len = padding_len
        
        self.sentiments_idx = {
            'positive': 10,
            'negative': 20,
            'neutral': 30
        }
        self._process()
        

    def _process(self):
        self.objects_ = self.tokenizer.encode_batch(self.data["text"].tolist())
        self.targets = []
        for i, obj in enumerate(self.objects_):
            text = self.data.loc[i,"text"]
            if "selected_text" in self.data.columns:
                selected_text = self.data.loc[i,"selected_text"]
                self.targets.append(self._create_target(obj, text, selected_text))
            else:
                self.targets.append(None)
                
    def _create_target(self, obj, text, selected_text):
        offsets = obj.offsets
        
        # find selected text index
        index_from = text.find(selected_text)
        assert index_from>=0, f"Text not found! text:/*{text}*/ selected_text:/*{selected_text}*/"
        
        index_to = index_from + len(selected_text)
        selected_text_mask = np.zeros(len(text))
        selected_text_mask[index_from:index_to-1] = 1
        
        target = []
        for start, end in offsets:
            target.append(TAG_TO_IX["I"] if np.mean(selected_text_mask[start:end])>self.threshold_select else 0)
        
        counts = sum(target)
        if counts>0:
            target[target.index(TAG_TO_IX["I"])] = TAG_TO_IX["B"]
        if counts>TAG_TO_IX["I"]:
            target[len(target) - 1 - target[::-1].index(TAG_TO_IX["B"])] = TAG_TO_IX["E"]
        return target
    
    def get_ids_target(self, idx):
        
        payload = [SOS_IDX, self.sentiments_idx[self.data.loc[idx, "sentiment"]]]
        payload_tag = [SOS_IDX, TAG_TO_IX["O"]]

        ids = payload+self.objects_[idx].ids
        suffix = [PAD_IDX]* (self.padding_len - len(ids))

        ids = ids + suffix
        
        if self.targets[idx]:
            target = np.array(payload_tag+self.targets[idx]+suffix)#.reshape(-1, 1)
        else:
            target = None
            
        return np.array(ids), target
    
    def __len__(self):
        return len(self.data)
    
    def _get_selected_text(self, idx):
        v_name = "selected_text"
        return self.data.loc[idx, v_name] if v_name in self.data.columns else None
    
    def __getitem__(self, idx):
        ids, target = self.get_ids_target(idx)
        target = target if self.targets[idx] else None
        """return {
            "text": self.data.loc[idx, "text"],
            "selected_text": self._get_selected_text(idx),
            "offset": self.objects_[idx].offsets,
            "ids": torch.tensor(ids, dtype=torch.long),
            "tokens": self.objects_[idx].tokens,
            "sentiment": self.data.loc[idx, "sentiment"],
            "target": target
        }"""
        return [], LongTensor(ids), LongTensor(target)





if __name__ == "__main__":
    from tokenizers import BertWordPieceTokenizer
    import transformers
    import os

    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    DISTILBERT_PATH = 'distilbert-base-uncased'

    # Save Tokenizer locally
    if not os.path.isdir(DISTILBERT_PATH):
        tmp_tok = transformers.DistilBertTokenizer.from_pretrained(DISTILBERT_PATH)
        os.mkdir(DISTILBERT_PATH)
        tmp_tok.save_pretrained(DISTILBERT_PATH)
        del tmp_tok

    # Testing the Tokenizer
    tokenizer = BertWordPieceTokenizer(f'{DISTILBERT_PATH}/vocab.txt', lowercase=True)

    train_dataset = Dataset(train_df, tokenizer)
    test_dataset = Dataset(test_df, tokenizer)

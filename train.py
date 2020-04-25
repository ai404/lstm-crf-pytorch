from model import *
from utils import *
from evaluate import *
from tqdm import tqdm

def load_data():
    data = dataloader()
    batch = []
    cti = load_tkn_to_idx(sys.argv[2]) # char_to_idx
    wti = load_tkn_to_idx(sys.argv[3]) # word_to_idx
    itt = load_idx_to_tkn(sys.argv[4]) # idx_to_tkn
    print("loading %s..." % sys.argv[5])
    with open(sys.argv[5], "r") as fo:
        text = fo.read().strip().split("\n" * (HRE + 1))
    for block in text:
        for line in block.split("\n"):
            x, y = line.split("\t")
            x = [x.split(":") for x in x.split(" ")]
            y = [int(y)] if HRE else [int(x) for x in y.split(" ")]
            xc, xw = zip(*[(list(map(int, xc.split("+"))), int(xw)) for xc, xw in x])
            data.append_item(xc = xc, xw = xw, y0 = y)
        data.append_row()
    data.strip()
    for _batch in data.split():
        xc, xw = data.tensor(_batch.xc, _batch.xw, _batch.lens)
        _, y0 = data.tensor(None, _batch.y0, sos = True)
        batch.append((xc, xw, y0))
    print("data size: %d" % len(data.y0))
    print("batch size: %d" % BATCH_SIZE)
    return batch, cti, wti, itt

def train():
    num_epochs = int(sys.argv[-1])
    if BERT:
        from prepare_dataset_challenge import Dataset, TAG_TO_IX
        from torch.utils.data import DataLoader
        from tokenizers import BertWordPieceTokenizer

        tokenizer = BertWordPieceTokenizer(f'{BERT_PATH}/vocab.txt', lowercase=True)

        train_dataset = Dataset(sys.argv[2], tokenizer)
        batch = DataLoader(train_dataset, batch_size=16,
                                shuffle=True, num_workers=4, drop_last=True)
        cti, wti, itt = {}, {}, TAG_TO_IX
    else:
        batch, cti, wti, itt = load_data()
    print(len(cti), len(wti), len(itt))
    model = rnn_crf(len(cti), len(wti), len(itt))
    optim = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    print(model)
    epoch = load_checkpoint(sys.argv[1], model) if isfile(sys.argv[1]) else 0
    filename = re.sub("\.epoch[0-9]+$", "", sys.argv[1])
    print("training model...")
    for ei in range(epoch + 1, epoch + num_epochs + 1):
        print(f"Epoch {ei}...")
        loss_sum = 0
        timer = time()
        for xc, xw, y0 in tqdm(batch):
            xw = xw.cuda() if CUDA else xw
            y0 = y0.cuda() if CUDA else y0
            loss = model(xc, xw, y0) # forward pass and compute loss
            loss.backward() # compute gradients
            optim.step() # update parameters
            loss_sum += loss.item()
        timer = time() - timer
        loss_sum /= len(batch)
        if ei % SAVE_EVERY and ei != epoch + num_epochs:
            save_checkpoint("", None, ei, loss_sum, timer)
        else:
            save_checkpoint(filename, model, ei, loss_sum, timer)
        if EVAL_EVERY and (ei % EVAL_EVERY == 0 or ei == epoch + num_epochs):
            args = [model, cti, wti, itt]
            evaluate(predict(sys.argv[-2], *args), True)
            model.train()
            print()

if __name__ == "__main__":
    if len(sys.argv) not in [4, 5, 7, 8]:
        sys.exit("Usage: %s model char_to_idx word_to_idx tag_to_idx training_data (validation_data) num_epoch" % sys.argv[0])
    if len(sys.argv) in [4, 7]:
        EVAL_EVERY = False
    train()

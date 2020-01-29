from model import *
from utils import *

def load_model():
    cti = load_tkn_to_idx(sys.argv[2]) # char_to_idx
    wti = load_tkn_to_idx(sys.argv[3]) # word_to_idx
    itt = load_idx_to_tkn(sys.argv[4]) # idx_to_tag
    model = rnn_crf(len(cti), len(wti), len(itt))
    print(model)
    load_checkpoint(sys.argv[1], model)
    return model, cti, wti, itt

def run_model(model, itt, data):
    data.sort()
    for batch in data.split():
        xc, xw = data.tensor(batch.xc, batch.xw, batch.lens)
        y1 = model.decode(xc, xw, batch.lens)
        data.y1.extend([[itt[i] for i in x] for x in y1])
    data.unsort()
    for x0, y0, y1 in zip(data.x0, data.y0, data.y1):
        if HRE:
            for x0, y0, y1 in zip(x0, y0, y1):
                yield x0, y0, y1
        else:
            yield x0[0], y0, y1

def predict(filename, model, cti, wti, itt):
    data = dataloader()
    with open(filename) as fo:
        text = fo.read().strip().split("\n" * (HRE + 1))
    for block in text:
        for x0 in block.split("\n"):
            if re.match("(\S+/\S+ ?))+$", x0): # word/tag
                x0, y0 = zip(*[re.split("/(?=[^/]+$)", x) for x in x0.split(" ")])
                x0 = " ".join(x0)
            elif re.match("(\S+ ?)*\t\S+$", x0): # sentence \t label
                x0, *y0 = x0.split("\t")
            else: # no ground truth provided
                y0 = []
            x1 = tokenize(x0)
            xc = [[cti[c] if c in cti else UNK_IDX for c in w] for w in x1]
            xw = [wti[w] if w in wti else UNK_IDX for w in x1]
            data.append_item(x0 = [x0], xc = [xc], xw = [xw], y0 = y0)
        data.append_row()
    data.strip()
    with torch.no_grad():
        model.eval()
        return run_model(model, itt, data)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: %s model char_to_idx word_to_idx tag_to_idx test_data" % sys.argv[0])
    for x0, y0, y1 in predict(sys.argv[5], *load_model()):
        if not TASK:
            print((x0, y0, y1) if y0 else (x0, y1))
        else: # word/sentence segmentation
            if y0:
                print(iob_to_txt(x0, y0))
            print(iob_to_txt(x0, y1))
            print()

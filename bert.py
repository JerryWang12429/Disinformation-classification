import pandas as pd
import numpy as np
import pdb

import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

PRETRAINED_MODEL_NAME = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
print("PyTorch 版本：", torch.__version__)

NUM_LABELS = 2


class FakeNewsDataset(Dataset):
    def __init__(self, mode, tokenizer):
        assert mode in ["train", "test"]
        self.mode = mode

        self.df = pd.read_csv("data/" + mode + ".tsv", sep="\t").fillna("")
        self.len = len(self.df)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        if self.mode == "test":
            text_a = self.df.text[idx]
            label_tensor = None
        else:
            text_a = self.df.text[idx]

            label_id = self.df.label[idx]
            label_tensor = torch.tensor(label_id)

        word_pieces = ["[CLS]"]
        tokens_a = self.tokenizer.tokenize(text_a)
        word_pieces += tokens_a + ["[SEP]"]
        len_a = len(word_pieces)

        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)

        segments_tensor = torch.tensor([1] * len_a, dtype=torch.long)

        return (tokens_tensor, segments_tensor, label_tensor)

    def __len__(self):
        return self.len


def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]

    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None

    # zero pad
    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)

    # attention masks，關注非zero padding位置
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)

    return tokens_tensors, segments_tensors, masks_tensors, label_ids


def get_predictions(model, dataloader, compute_acc=False):
    predictions = None
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():

        for data in dataloader:
            # 將tensors 移到 GPU 上
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]

            outputs = model(*data[:3])

            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)

            if compute_acc:
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()

            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))

    if compute_acc:
        acc = correct / total
        return predictions, acc
    return predictions


def restructure():
    data = pd.read_csv('data/data_2tsv.tsv', sep='\t')
    data = data[data['replyType'] != 'NOT_ARTICLE']
    types = data.replyType.unique()
    dic = {}
    for i, types in enumerate(types):
        dic[types] = i
    print(dic)
    data['type_id'] = data.replyType.apply(lambda x: dic[x])

    data_bert = pd.DataFrame({
        'id': range(len(data)),
        'label': data.type_id,
        'alpha': ['a'] * data.shape[0],
        'text': data.text.replace(r'\n', ' ', regex=True)
    })
    print(data_bert.head())
    print(data_bert.shape)

    train, test = np.split(data_bert.sample(frac=1),
                           [int(.8 * len(data_bert))])

    print(train.shape)
    print(test.shape)

    train.text = train.text.str.slice(0, 75)
    test.text = test.text.str.slice(0, 75)

    train.to_csv('data/train.tsv', sep='\t', index=False)
    test.to_csv('data/test.tsv', sep='\t', index=False)


if __name__ == "__main__":
    restructure()

    trainset = FakeNewsDataset("train", tokenizer=tokenizer)

    BATCH_SIZE = 64
    trainloader = DataLoader(trainset,
                             batch_size=BATCH_SIZE,
                             collate_fn=create_mini_batch)
    data = next(iter(trainloader))

    tokens_tensors, segments_tensors, \
        masks_tensors, label_ids = data

    print(f"""
    tokens_tensors.shape   = {tokens_tensors.shape}
    {tokens_tensors}
    ------------------------
    segments_tensors.shape = {segments_tensors.shape}
    {segments_tensors}
    ------------------------
    masks_tensors.shape    = {masks_tensors.shape}
    {masks_tensors}
    ------------------------
    label_ids.shape        = {label_ids.shape}
    {label_ids}
    """)
    model = BertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)

    print("""name            module
    ----------------------""")
    for name, module in model.named_children():
        if name == "bert":
            for n, _ in module.named_children():
                print(f"{name}:{n}")
        else:
            print("{:15} {}".format(name, module))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = model.to(device)
    _, acc = get_predictions(model, trainloader, compute_acc=True)
    print("classification acc:", acc)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    EPOCHS = 20

    Quantity = pd.read_csv('data/train.tsv', sep='\t')
    count = Quantity.shape[0]
    ave = count/BATCH_SIZE

    for epoch in range(EPOCHS):

        running_loss = 0.0
        for data in trainloader:

            tokens_tensors, segments_tensors, masks_tensors, labels = [
                t.to(device) for t in data
            ]

            # 將參數梯度歸零
            optimizer.zero_grad()

            # forward pass
            outputs = model(input_ids=tokens_tensors,
                            token_type_ids=segments_tensors,
                            attention_mask=masks_tensors,
                            labels=labels)
            loss = outputs[0]
            # backward
            loss.backward()
            optimizer.step()

            # 紀錄當前 batch loss
            running_loss += loss.item()
        # 計算分類準確率

        _, acc = get_predictions(model, trainloader, compute_acc=True)
        print('[epoch %d] loss: %.3f, acc: %.3f' %
              (epoch + 1, running_loss/ave, acc))

    model.save_pretrained('model/')
    tokenizer.save_pretrained('model/')

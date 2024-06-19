import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from dataset import BertDataset
from vocab import BertVocab
from models import BertModel


def train(train_loader, model, criterion, optimizer, scheduler, device, epoch):
    model.train()
    losses = torch.zeros(len(train_loader))
    for i, data in enumerate(train_loader):
        cur_lr = optimizer.param_groups[0]['lr']
        # data to device
        data = {key: value.to(device) for key, value in data.items()}
        # forward
        next_sentence, mask_language = model(data["bert_input"], data["segment_label"])
        # loss
        next_loss = criterion(next_sentence, data["is_next"])
        mask_loss = criterion(mask_language.reshape(-1, mask_language.shape[-1]), data["bert_label"].view(-1))

        loss = next_loss + mask_loss
        losses[i] = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i%100 == 0:
            print("epoch: {}, iter: {}/{}, cur_lr: {:.6f}, next loss: {:.3f}, mask loss: {:.3f}".format(
                epoch + 1, i, len(train_loader), cur_lr, next_loss.item(), mask_loss.item())
            )

    return torch.mean(losses)


def main():
    # params
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    corpus_path = 'data/corpus.small'
    max_seq_len = 32

    embedding_dim = 256
    num_heads = 8
    num_layers = 6
    dropout = 0.1

    lr = 1e-3
    batch_size = 64
    epochs = 100

    # load vocab
    vocab = BertVocab(corpus_path)

    # load dataset
    train_data = BertDataset(corpus_path, vocab, max_seq_len)
    data_loader = DataLoader(train_data, batch_size=batch_size, num_workers=4, shuffle=True)
    num_warmup_steps = int(len(data_loader) * epochs * 0.1)
    num_training_steps = int(len(data_loader) * epochs)

    # load model
    model = BertModel(len(vocab), embedding_dim, num_heads, num_layers, dropout)
    model = model.to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.999), weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    # training loop
    print('Training Loop')
    for epoch in range(epochs):
        train_loss = train(data_loader, model, criterion, optimizer, scheduler, device, epoch)
        print("epoch: {}, train avg loss: {:.2f}".format(epoch + 1, train_loss))
        torch.save(model.state_dict(), "weights/epoch_{}_loss_{:.2f}.pt".format(epoch + 1, train_loss))


if __name__ == "__main__":
    main()

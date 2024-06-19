import torch
import torch.nn.functional as F

from models import BertModel
from vocab import BertVocab


def sentence2token(sentence, sentence_mask_id, vocab):
    sentence_token = []
    for i, token in enumerate(sentence.split()):
        if i == sentence_mask_id:
            sentence_token.append(vocab.mask_index)
        else:
            sentence_token.append(vocab.stoi.get(token, vocab.unk_index))
    return sentence_token


def preprocess(input_sentence, sentence1_mask_id, sentence2_mask_id, vocab, max_seq_len):
    sentence1, sentence2 = input_sentence.split('\t')
    sentence1_token = sentence2token(sentence1, sentence1_mask_id, vocab)
    sentence2_token = sentence2token(sentence2, sentence2_mask_id, vocab)

    t1 = [vocab.sos_index] + sentence1_token + [vocab.eos_index]
    t2 = sentence2_token + [vocab.eos_index]

    segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:max_seq_len]
    bert_input = (t1 + t2)[:max_seq_len]

    padding = [vocab.pad_index for _ in range(max_seq_len - len(bert_input))]
    bert_input.extend(padding), segment_label.extend(padding)

    mask_ids = [sentence1_mask_id+1, len(t1)+sentence2_mask_id]

    return bert_input, segment_label, mask_ids


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    corpus_path = 'data/corpus.small'
    max_seq_len = 32

    embedding_dim = 256
    num_heads = 8
    num_layers = 6
    dropout = 0.1

    # load vocab
    vocab = BertVocab(corpus_path)

    # load model
    model = BertModel(len(vocab), embedding_dim, num_heads, num_layers, dropout)
    model.load_state_dict(torch.load('weights/epoch_100_loss_2.56.pt', map_location='cpu'))
    model.eval()
    model = model.to(device)

    # def input sentence
    input_sentence = 'Robbie album sold on memory card\tSinger Robbie Williams\' greatest hits album is to be sold on a memory card for mobile phones.'
    sentence1_mask_id, sentence2_mask_id = 2, 5 # sold, album

    # pre-process
    bert_input, segment_label, mask_ids = preprocess(input_sentence, sentence1_mask_id, sentence2_mask_id, vocab, max_seq_len)

    # to tensor
    bert_input = torch.tensor([bert_input]).to(device)
    segment_label = torch.tensor([segment_label]).to(device)

    # forward
    next_sentence, mask_language = model(bert_input, segment_label)

    # post-process
    next_sentence = F.softmax(next_sentence, dim=-1)
    mask_language = F.softmax(mask_language[0], dim=-1)
    next_score, is_next = next_sentence.max(dim=-1)
    _, mask1_word_id = mask_language[mask_ids[0]].max(dim=-1)
    _, mask2_word_id = mask_language[mask_ids[1]].max(dim=-1)

    # output
    print('is_next:', is_next.item(), '\tsocre:', next_score.item())
    print('mask1_word:', vocab.itos[mask1_word_id])
    print('mask2_word:', vocab.itos[mask2_word_id])


if __name__ == "__main__":
    main()
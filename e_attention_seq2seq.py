import torch
from torch import nn
from function.a_dataset import load_data_nmt
from function.b_encoder_decoder import AttentionDecoder, EncoderDecoder
from function.d_attention_score import AdditiveAttention
from c_seq2seq import Seq2SeqEncoder,train_seq2seq,bleu,predict_seq2seq


class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,dropout=0.0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = AdditiveAttention(num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # outputs的形状为(batch_size，num_steps，num_hiddens).
        # hidden_state的形状为(num_layers，batch_size，num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # enc_outputs的形状为(batch_size,num_steps,num_hiddens).
        # hidden_state的形状为(num_layers,batch_size,
        # num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # 输出X的形状为(num_steps,batch_size,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # query的形状为(batch_size,1,num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # context的形状为(batch_size,1,num_hiddens)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            # 在特征维度上连结
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # 将x变形为(1,batch_size,embed_size+num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # 全连接层变换后，outputs的形状为
        # (num_steps,batch_size,vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def __getitem__(self, item):
        return self.data[item]

if __name__ == '__main__':
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 250, torch.device('cuda:0')
    constrain=600

    train_iter, src_vocab, tgt_vocab,english,frash = load_data_nmt(batch_size, num_steps,num_examples=constrain)
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)

    decoder = Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)


    english = [' '.join(qwe) for qwe in english][:600]
    frash = [' '.join(qwe) for qwe in frash][:600]
    mse = Accumulator(2)
    q = 1
    for eng, fra in zip(english, frash):
        translation, attention_weight_seq = predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device)
        mse.add(bleu(translation, fra, k=2), q)
        print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')
    print(f'翻译准确率：{(mse[0] / mse[1]) * 100:.2f}%')





































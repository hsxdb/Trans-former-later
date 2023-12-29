import ast
import collections
import torch
from torch.utils.data import TensorDataset, DataLoader


class vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)  # 返回的字符统计字典
        """item转成列表,key=lambda x: x[1]: 这是一个关键函数，用于指定按照键-值对中的第二个元素（频次）进行排序。
        x[1] 表示访问每个键-值对的第二个元素，即频次。排序时会根据频次来比较元素。
        reverse=True: 指定排序的顺序。True 表示按照降序（从大到小）排序，也就是频次最高的标记排在前面。"""
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [token for token, freq in self.token_freqs if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, toekns):
        """如果传入的 tokens 参数是单个标记（字符串），则该方法尝试将这个标记转换为对应的索引。
        如果找到对应的索引，就返回这个索引；如果标记不存在，就返回 self.unk，表示未知标记的索引。"""
        if not isinstance(toekns, (list, tuple)):
            return self.token_to_idx.get(toekns, self.unk)
        """如果传入的 tokens 参数是一个包含多个标记的列表或元组，该方法使用列表解析逐个处理每个标记，然后返回对应的索引"""
        return [self.__getitem__(token) for token in toekns]

    def to_token(self, indices):
        try:
            if not isinstance(indices, (list, tuple)):
                return self.idx_to_token[indices]
            return [self.to_token(index) for index in indices]
        except:
            return "错误：索引超出边界"


def count_corpus(tokens):
    # 如果tokens为空，或者tokens的第一个元素是列表（即tokens是一个二维列表）
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 使用列表推导式将二维列表转换为一维列表
        # 对于tokens中的每一行（line），对于line中的每一个词（token）
        # 将这个词添加到新的一维列表中
        tokens = [token for line in tokens for token in line]
    # 使用collections.Counter对一维列表中的词进行计数
    # 返回一个字典，其中键是词，值是该词在tokens中出现的次数
    return collections.Counter(tokens)


def read_data_nmt():
    """载入“英语－法语”数据集"""
    with open('./data/fra.txt', 'r', encoding='utf-8') as f:
        return f.read()
    #     lines = f.readlines()
    #
    # sour = []
    # tga=[]
    #
    # for line in lines:
    #     # 使用制表符 '\t' 分割每行，取第二个制表符之前的内容作为翻译文本
    #     a=line.split('\t')
    #     translation_text = str(a[:1])
    #     tg=str(a[1:2])
    #     tga.append(tg)
    #     sour.append(translation_text)
    #
    # # 将处理后的文本列表连接成一个字符串，每行用换行符分隔
    # processed_data = [', '.join([f"'{char}'" for char in ast.literal_eval(item)]) for item in tga]
    # processed_dataq = [
    #     [' '.join(item.split()), ', '.join([f"'{char}'" for char in ast.literal_eval(item)[1]])]
    #     for item in sour
    # ]
    # cleaned_text = '\n'.join(cleaned_lines)
    #
    # return cleaned_text


def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""

    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char for i, char in enumerate(text)]
    return ''.join(out)


def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))


def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator.
    Defined in :numref:`sec_utils`"""
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)


def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab,source,target


if __name__ == '__main__':
    raw_text = read_data_nmt()
    text = preprocess_nmt(raw_text)
    print(text[:80])
    source, target = tokenize_nmt(text)
    print(source[:6], target[:6])

    src_vocab = vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    print(len(src_vocab))
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
    for X, X_valid_len, Y, Y_valid_len in train_iter:
        print('X:', X.type(torch.int32))
        print('X的有效长度:', X_valid_len)
        print('Y:', Y.type(torch.int32))
        print('Y的有效长度:', Y_valid_len)
        break

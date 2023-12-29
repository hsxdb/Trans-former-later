import hashlib
import math
import torch
from torch import nn
from loguru import logger
from function.a_dataset import load_data_nmt
from function.b_encoder_decoder import AttentionDecoder, Encoder, EncoderDecoder
from c_seq2seq import train_seq2seq, Accumulator, bleu, predict_seq2seq
from function.f_multihead_attention import MultiHeadAttention, PositionalEncoding
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox
import csv
import pandas as pd


class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""

    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    """残差连接后进行层规范化"""

    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):
    """Transformer编码器块"""

    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(Encoder):
    """Transformer编码器"""

    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                              norm_shape, ffn_num_input, ffn_num_hiddens,
                                              num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        b = self.embedding(X)
        c = math.sqrt(self.num_hiddens)
        a = b * c
        X = self.pos_encoding(a)
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X


class DecoderBlock(nn.Module):
    """解码器中第i个块"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的开头:(batch_size,num_steps),
            # 其中每一行是[1,2,...,num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # 自注意力
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 编码器－解码器注意力。
        # enc_outputs的开头:(batch_size,num_steps,num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


class TransformerDecoder(AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 DecoderBlock(key_size, query_size, value_size, num_hiddens,
                                              norm_shape, ffn_num_input, ffn_num_hiddens,
                                              num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights


if __name__ == '__main__':

    logger.add("./log/transformer.log", level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
    lr, num_epochs, device = 0.005, 5000, torch.device('cuda:0')
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]
    constrain = 99999999

    train_iter, src_vocab, tgt_vocab, english, frash = load_data_nmt(batch_size, num_steps, num_examples=constrain)

    encoder = TransformerEncoder(
        10012, key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    decoder = TransformerDecoder(
        17851, key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)
    # train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    net.load_state_dict(torch.load('./model_data/model_data_500.pth'))
    # english = [' '.join(qwe) for qwe in english]
    # frash = [' '.join(qwe) for qwe in frash]
    mse = Accumulator(2)
    q = 1
    net = net.to(device)


    def machine():
        def translate_text():
            source_language = source_text.get("1.0", tk.END).strip()
            translation, _ = predict_seq2seq(net, source_language, src_vocab, tgt_vocab, num_steps, device)
            target_text.config(state="normal")
            if 'unk' in translation:
                translation = '由于训练数据集不够大，导致没有识别的单词，因此模型需要在更大数据集上训练'
            target_text.delete("1.0", tk.END)  # Clear the entire content of the Text widget
            target_text.insert(tk.END, translation)  # Insert new text at the end
            target_text.config(state="disabled")

        def on_click(event):
            if source_text.get("1.0", tk.END).strip() == "输入需要翻译的英语":
                source_text.delete("1.0", tk.END)
                source_text.config(fg='black')

        def tr():
            window.destroy()

        window = tk.Tk()
        window.title("Transformer 机器翻译")
        window.resizable(False, False)
        # 设置窗口大小
        window.geometry("500x400")  # 你可以根据需要调整窗口大小

        # 背景图片
        background_image = Image.open("./log/img.png")  # 替换为你的背景图片路径
        background_photo = ImageTk.PhotoImage(background_image)

        # 创建标签来显示背景图片
        background_label = tk.Label(window, image=background_photo)
        background_label.place(relwidth=1, relheight=1)  # 让标签铺满整个窗口

        # 标题行
        title_label = tk.Label(window, text="Transformer 机器翻译", font=('Times New Roman', 25, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=10, sticky=tk.EW)  # 使用sticky=tk.EW让标题居中

        source_text = tk.Text(window, width=21, height=6, font=("Times New Roman", 12))
        source_text.insert("1.0", "输入需要翻译的英语")
        source_text.bind("<FocusIn>", on_click)
        source_text.grid(row=1, column=0, pady=5, sticky=tk.W)

        target_text = tk.Text(window, width=21, height=6, state="disabled", font=("Times New Roman", 12))
        target_text.grid(row=1, column=1, pady=5, sticky=tk.E)

        # 翻译按钮
        translate_button = tk.Button(window, text="翻译", font=('Helvetica', 17, 'bold'), command=translate_text)
        translate_button.grid(row=3, column=0, columnspan=2, sticky=tk.EW)  # 使用sticky=tk.EW让按钮居中
        # 翻译按钮
        translate_button1 = tk.Button(window, text="退出", font=('Helvetica', 17, 'bold'), command=tr)
        translate_button1.grid(row=4, column=0, columnspan=2, sticky=tk.EW)  # 使用sticky=tk.EW让按钮居中

        # 配置行和列的权重，使其居中
        window.grid_rowconfigure(0, weight=1)
        window.grid_rowconfigure(3, weight=1)
        window.grid_columnconfigure(0, weight=1)
        window.grid_columnconfigure(1, weight=1)
        window.eval('tk::PlaceWindow . center')

        # 运行程序
        window.mainloop()


    def hash_password(password):
        # 使用SHA-256哈希函数加密密码
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        return hashed_password


    def open_main_window():
        # 关闭登录窗口
        login_window.destroy()
        machine()


    def register(user=None):
        def registerq():
            username = entry_username.get()
            password = entry_password.get()
            df = pd.read_csv('./data/user_data.csv')
            # 使用 pandas 检查用户名是否已经存在
            if username in df['Username'].values:
                messagebox.showerror("注册失败", "用户名已存在")
                return
            if username and password:
                # 打开 CSV 文件，如果不存在则创建
                hashed_password = hash_password(password)
                with open('./data/user_data.csv', 'a', newline='') as file:
                    writer = csv.writer(file)

                    writer.writerow([username, hashed_password])

                messagebox.showinfo("注册成功", f"用户{username}注册成功！")
                register_window.destroy()
            else:
                messagebox.showerror("注册失败", "请输入用户名和密码")

        # 创建注册窗口
        register_window = tk.Toplevel(login_window)
        register_window.title("注册窗口")
        register_window.resizable(False, False)

        # 读取背景图像
        background_image = Image.open("./log/img_2.png")  # 替换为你的背景图片路径
        scale_percent = 50  # 50% 的缩放

        # 计算缩放后的大小
        width = int(background_image.width * scale_percent / 100)
        height = int(background_image.height * scale_percent / 100)

        # 缩放图像大小
        background_image = background_image.resize((width, height), Image.LANCZOS)

        background_photo = ImageTk.PhotoImage(background_image)

        # 创建标签来显示缩放后的背景图片
        background_label = tk.Label(register_window, image=background_photo)
        background_label.place(relwidth=1, relheight=1)

        # 设置窗口大小为背景图像大小
        register_window.geometry("{}x{}".format(background_image.width, background_image.height))

        # 设置组件相对位置，使其居中于背景图像
        tk.Label(register_window, text="用户名：", font=("Times New Roman", 14)).place(relx=0.5, rely=0.3, anchor=tk.CENTER)
        entry_username = tk.Entry(register_window, font=("Times New Roman", 14))
        entry_username.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
        entry_username.insert(0, f"{user}")

        tk.Label(register_window, text="密码：", font=("Times New Roman", 14)).place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        entry_password = tk.Entry(register_window, show="*", font=("Times New Roman", 14))
        entry_password.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

        # 注册按钮
        register_button = tk.Button(register_window, text="注册", command=registerq, font=("Times New Roman", 14))
        register_button.place(relx=0.5, rely=0.7, anchor=tk.CENTER)
        x = int((register_window.winfo_screenwidth() - register_window.winfo_reqwidth()) / 2)
        y = int((register_window.winfo_screenheight() - register_window.winfo_reqheight()) / 2)

        # 将窗口居中显示
        register_window.geometry(f"+{x - 100}+{y - 70}")
        register_window.mainloop()


    def login():
        username = entry_username.get()
        password = entry_password.get()
        # 这里可以添加你的登录验证逻辑，这里简单地使用用户名和密码进行验证

        if username and password:
            df = pd.read_csv('./data/user_data.csv')

            if not df.empty and (df['Username'] == username).any():
                hashed_password = hash_password(password)
                stored_hashed_password = df.loc[df['Username'] == username, 'Password'].iloc[0]
                if hashed_password == stored_hashed_password:
                    messagebox.showinfo("登录成功", "欢迎回来，{}".format(username))
                    open_main_window()
                else:
                    messagebox.showerror("登录失败", "用户名或密码错误")
            else:
                response = messagebox.askquestion("登录失败", f"用户{username}未注册。\n是否注册新用户？")
                if response == 'yes':
                    register(username)


    with open('./data/user_data.csv', 'a', newline='') as file:
        writer = csv.writer(file)

        # 如果文件为空，写入标题
        if file.tell() == 0:
            writer.writerow(['Username', 'Password'])
            writer.writerow(['root', 'root'])
    # 创建登录窗口
    login_window = tk.Tk()
    login_window.title("登录窗口")
    login_window.resizable(False, False)

    # 读取背景图像
    background_image = Image.open("./log/img_1.png")  # 替换为你的背景图片路径
    scale_percent = 50  # 50% 的缩放

    # 计算缩放后的大小
    width = int(background_image.width * scale_percent / 100)
    height = int(background_image.height * scale_percent / 100)

    # 缩放图像大小
    background_image = background_image.resize((width, height), Image.LANCZOS)

    background_photo = ImageTk.PhotoImage(background_image)

    # 创建标签来显示缩放后的背景图片
    background_label = tk.Label(login_window, image=background_photo)
    background_label.place(relwidth=1, relheight=1)  # 让标签铺满整个窗口

    # 设置窗口大小为背景图像大小
    login_window.geometry("{}x{}".format(background_image.width, background_image.height))

    # 设置组件相对位置，使其居中于背景图像
    tk.Label(login_window, text="Transformer 机器翻译", font=("Times New Roman", 17)).place(relx=0.5, rely=0.1,
                                                                                        anchor=tk.CENTER)

    # 用户名标签和输入框
    tk.Label(login_window, text="用户名：", font=("Times New Roman", 14)).place(relx=0.3, rely=0.3, anchor=tk.CENTER)
    entry_username = tk.Entry(login_window, font=("Times New Roman", 14))
    entry_username.place(relx=0.7, rely=0.3, anchor=tk.CENTER)

    # 密码标签和输入框
    tk.Label(login_window, text="密码：", font=("Times New Roman", 14)).place(relx=0.3, rely=0.4, anchor=tk.CENTER)
    entry_password = tk.Entry(login_window, show="*", font=("Times New Roman", 14))
    entry_password.place(relx=0.7, rely=0.4, anchor=tk.CENTER)

    # 登录按钮
    login_button = tk.Button(login_window, text="登录", command=login, font=("Times New Roman", 14))
    login_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    # 注册按钮
    register_button = tk.Button(login_window, text="注册", command=register, font=("Times New Roman", 14))
    register_button.place(relx=0.5, rely=0.6, anchor=tk.CENTER)
    login_window.eval('tk::PlaceWindow . center')
    login_window.mainloop()

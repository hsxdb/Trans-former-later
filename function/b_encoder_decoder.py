from torch import nn


class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()

    def forward(self, x, *args):
        raise NotImplementedError


class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__()

    def init_state(self, enc_output, *args):
        raise NotImplementedError

    def forward(self, x, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_x, dec_x, *args):
        enc_output = self.encoder(enc_x, *args)
        dec_state = self.decoder.init_state(enc_output, *args)
        return self.decoder(dec_x, dec_state)



class AttentionDecoder(Decoder):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError
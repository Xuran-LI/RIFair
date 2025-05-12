import keras
import tensorflow
from keras import layers
from tensorflow.python.ops.numpy_ops import shape, reshape, multiply


class AutoIntTransformerBlock(layers.Layer):
    def __init__(self, embed_dim=32, num_heads=4, ff_dim=32, rate=0.1, **kwargs):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "att": self.att,
            "ffn": self.ffn,
            "layernorm1": self.layernorm1,
            "layernorm2": self.layernorm2,
            "dropout1": self.dropout1,
            "dropout2": self.dropout2
        })
        return config


class AdultEmbedding(layers.Layer):
    def __init__(self, vocab_size=108, embed_size=32, rate=0.1, **kwargs):  # Adult
        super().__init__()
        self.index_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_size)
        self.dropout1 = layers.Dropout(rate)

    def call(self, embed_index, embed_value):
        embed_index = self.index_emb(embed_index)
        input_size = shape(embed_value)[-1]
        embed_value = reshape(embed_value, newshape=[-1, input_size, 1])
        embeddings = multiply(embed_index, embed_value)
        return self.dropout1(embeddings)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "index_emb": self.index_emb,
            "dropout1": self.dropout1
        })
        return config


class BankEmbedding(layers.Layer):
    def __init__(self, vocab_size=54, embed_size=32, rate=0.1, **kwargs):  # Adult
        super().__init__()
        self.index_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_size)
        self.dropout1 = layers.Dropout(rate)

    def call(self, embed_index, embed_value):
        embed_index = self.index_emb(embed_index)
        input_size = shape(embed_value)[-1]
        embed_value = reshape(embed_value, newshape=[-1, input_size, 1])
        embeddings = multiply(embed_index, embed_value)
        return self.dropout1(embeddings)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "index_emb": self.index_emb,
            "dropout1": self.dropout1
        })
        return config


class COMPASEmbedding(layers.Layer):
    def __init__(self, vocab_size=911, embed_size=32, rate=0.1, **kwargs):  # COMPAS
        super().__init__()
        self.index_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_size)
        self.dropout1 = layers.Dropout(rate)

    def call(self, embed_index, embed_value):
        embed_index = self.index_emb(embed_index)
        input_size = shape(embed_value)[-1]
        embed_value = reshape(embed_value, newshape=[-1, input_size, 1])
        embeddings = multiply(embed_index, embed_value)
        return self.dropout1(embeddings)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "index_emb": self.index_emb,
            "dropout1": self.dropout1
        })
        return config


class CreditEmbedding(layers.Layer):
    def __init__(self, vocab_size=63, embed_size=32, rate=0.1, **kwargs):  # COMPAS
        super().__init__()
        self.index_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_size)
        self.dropout1 = layers.Dropout(rate)

    def call(self, embed_index, embed_value):
        embed_index = self.index_emb(embed_index)
        input_size = shape(embed_value)[-1]
        embed_value = reshape(embed_value, newshape=[-1, input_size, 1])
        embeddings = multiply(embed_index, embed_value)
        return self.dropout1(embeddings)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "index_emb": self.index_emb,
            "dropout1": self.dropout1
        })
        return config


class ACSEmploymentEmbedding(layers.Layer):
    def __init__(self, vocab_size=100, embed_size=32, rate=0.1, **kwargs):  # ACSEmployment
        super().__init__()
        self.index_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_size)
        self.dropout1 = layers.Dropout(rate)

    def call(self, embed_index, embed_value):
        embed_index = self.index_emb(embed_index)
        input_size = shape(embed_value)[-1]
        embed_value = reshape(embed_value, newshape=[-1, input_size, 1])
        embeddings = multiply(embed_index, embed_value)
        return self.dropout1(embeddings)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "index_emb": self.index_emb,
            "dropout1": self.dropout1
        })
        return config


class ACSIncomeEmbedding(layers.Layer):
    def __init__(self, vocab_size=734, embed_size=32, rate=0.1, **kwargs):  # ACSIncome
        super().__init__()
        self.index_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_size)
        self.dropout1 = layers.Dropout(rate)

    def call(self, embed_index, embed_value):
        embed_index = self.index_emb(embed_index)
        input_size = shape(embed_value)[-1]
        embed_value = reshape(embed_value, newshape=[-1, input_size, 1])
        embeddings = multiply(embed_index, embed_value)
        return self.dropout1(embeddings)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "index_emb": self.index_emb,
            "dropout1": self.dropout1
        })
        return config


class ACSCoverageEmbedding(layers.Layer):
    def __init__(self, vocab_size=94, embed_size=32, rate=0.1, **kwargs):  # ACSCoverage
        super().__init__()
        self.index_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_size)
        self.dropout1 = layers.Dropout(rate)

    def call(self, embed_index, embed_value):
        embed_index = self.index_emb(embed_index)
        input_size = shape(embed_value)[-1]
        embed_value = reshape(embed_value, newshape=[-1, input_size, 1])
        embeddings = multiply(embed_index, embed_value)
        return self.dropout1(embeddings)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "index_emb": self.index_emb,
            "dropout1": self.dropout1
        })
        return config


class ACSMobilityEmbedding(layers.Layer):
    def __init__(self, vocab_size=119, embed_size=32, rate=0.1, **kwargs):  # ACSCoverage
        super().__init__()
        self.index_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_size)
        self.dropout1 = layers.Dropout(rate)

    def call(self, embed_index, embed_value):
        embed_index = self.index_emb(embed_index)
        input_size = shape(embed_value)[-1]
        embed_value = reshape(embed_value, newshape=[-1, input_size, 1])
        embeddings = multiply(embed_index, embed_value)
        return self.dropout1(embeddings)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "index_emb": self.index_emb,
            "dropout1": self.dropout1
        })
        return config


class ACSTravelEmbedding(layers.Layer):
    def __init__(self, vocab_size=710, embed_size=32, rate=0.1, **kwargs):  # ACSCoverage
        super().__init__()
        self.index_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_size)
        self.dropout1 = layers.Dropout(rate)

    def call(self, embed_index, embed_value):
        embed_index = self.index_emb(embed_index)
        input_size = shape(embed_value)[-1]
        embed_value = reshape(embed_value, newshape=[-1, input_size, 1])
        embeddings = multiply(embed_index, embed_value)
        return self.dropout1(embeddings)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "index_emb": self.index_emb,
            "dropout1": self.dropout1
        })
        return config


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim=32, num_heads=4, ff_dim=32, rate=0.1, **kwargs):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "att": self.att,
            "ffn": self.ffn,
            "layernorm1": self.layernorm1,
            "layernorm2": self.layernorm2,
            "dropout1": self.dropout1,
            "dropout2": self.dropout2
        })
        return config


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen=70, vocab_size=1500, embed_dim=32, **kwargs):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.maxlen = maxlen

    def call(self, x):
        positions = tensorflow.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "token_emb": self.token_emb,
            "pos_emb": self.pos_emb,
            "maxlen": self.maxlen
        })
        return config


class TokenAndPositionEmbeddingBank(layers.Layer):
    def __init__(self, maxlen=90, vocab_size=1500, embed_dim=32, **kwargs):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.maxlen = maxlen

    def call(self, x):
        positions = tensorflow.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "token_emb": self.token_emb,
            "pos_emb": self.pos_emb,
            "maxlen": self.maxlen
        })
        return config


class TokenAndPositionEmbeddingCOMPAS(layers.Layer):
    def __init__(self, maxlen=70, vocab_size=1500, embed_dim=32, **kwargs):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.maxlen = maxlen

    def call(self, x):
        positions = tensorflow.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "token_emb": self.token_emb,
            "pos_emb": self.pos_emb,
            "maxlen": self.maxlen
        })
        return config


class TokenAndPositionEmbeddingACS(layers.Layer):
    def __init__(self, maxlen=100, vocab_size=1500, embed_dim=32, **kwargs):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.maxlen = maxlen

    def call(self, x):
        positions = tensorflow.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "token_emb": self.token_emb,
            "pos_emb": self.pos_emb,
            "maxlen": self.maxlen
        })
        return config


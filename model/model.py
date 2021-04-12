import tensorflow as tf
import numpy as np
import random

class PooledBert(tf.keras.Model):
    def __init__(self,tokenizer,config):
        super(PooledBert,self).__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.EmbeddingLayer = EmbeddingLayer(config)

        self.encoder_16 = EncoderBlock(config,16)
        self.encoder_8 = EncoderBlock(config,8)
        self.encoder_4 = EncoderBlock(config,4)
        self.encoder_2 = EncoderBlock(config,2)
        self.encoder_1 = EncoderBlock(config,1)
        self.MLM_head = MaskedLanguageModel(self.tokenizer.vocab_size)
        

    def call(self,batch_data,train=False): # dataloader -> tokenizer -> Bert[embedding-> MHA-> head -> result]
        input_ids,attention_mask = self.tokenize(batch_data)
        input_ids_duplicated = tf.identity(input_ids) #tf.function 사용시 에러 
        if train:
            input_ids = self.mask_sentence(input_ids,attention_mask)
            loss_mask = tf.not_equal(input_ids_duplicated,input_ids)

        embeddings = self.EmbeddingLayer(input_ids)
        attention_mask = tf.cast(attention_mask,dtype=tf.float32)

        attention_weight_16, output_16 = self.encoder_16(embeddings,attention_mask)
        attention_weight_8, output_8 = self.encoder_8(output_16,attention_mask)
        attention_weight_4, output_4 = self.encoder_4(output_8,attention_mask)
        attention_weight_2, output_2 = self.encoder_2(output_4,attention_mask)
        attention_weight_1, output_1 = self.encoder_1(output_2,attention_mask)

        output = self.MLM_head(output_1)

        if train:
            label = tf.boolean_mask(input_ids,loss_mask)
            return output , loss_mask , label

        return output

    def tokenize(self,batch_data):
        output = self.tokenizer.batch_encode_plus(batch_data,padding='max_length',max_length=self.config.max_len,
                                                  return_token_type_ids=False)
        return output['input_ids'],output['attention_mask']
    
    def mask_sentence(self,batch_input_ids,batch_attention_mask):
        batch_masked_ids = []
        for input_id,attention_mask in zip(batch_input_ids,batch_attention_mask):
            coin = random.random()
            words = tf.math.reduce_sum(attention_mask)

            if coin < self.config.no_mask_ratio:
                batch_masked_ids.append(input_id)

            elif self.config.no_mask_ratio < coin < self.config.no_mask_ratio + self.config.mask_ratio:
                mask_idx = random.randint(1,words-2)
                input_id[mask_idx] = self.tokenizer.mask_token_id
                batch_masked_ids.append(input_id)

            elif self.config.no_mask_ratio + self.config.mask_ratio < coin < self.config.no_mask_ratio + self.config.mask_ratio + self.config.random_mask_ratio:
                mask_idx = random.randint(1,words-2)
                alternative_token_id = random.randint(0,self.tokenizer.vocab_size)
                input_id[mask_idx] = alternative_token_id
                batch_masked_ids.append(input_id)

            else:
                iteration = random.randint(1,words-2)
                for n in range(iteration):
                    mask_idx = random.randint(1,words-2)
                    input_id[mask_idx] = self.tokenizer.mask_token_id
                batch_masked_ids.append(input_id)

        return tf.constant(batch_masked_ids)


class MultiHeadAttention(tf.keras.Model):
    def __init__(self,config,pool_size):
        super(MultiHeadAttention,self).__init__()
        self.config = config
        self.d_model = self.config.d_model
        self.head_num = self.config.head_num

        self.w_q = tf.keras.layers.Dense(self.config.d_model)
        self.w_k = tf.keras.layers.Dense(self.config.d_model)
        self.w_v = tf.keras.layers.Dense(self.config.d_model)

        self.scale = tf.keras.layers.Lambda(lambda x: x/np.sqrt(self.config.d_model))
        self.linear = tf.keras.layers.Dense(self.config.d_model/pool_size)

    def call(self,X,mask): # X = q,k,v / mask = (batch_size,time_sequence)
        assert X.shape[-1] % self.head_num == 0

        wq = tf.reshape(self.w_q(X),(self.head_num,X.shape[0],X.shape[1],-1)) * tf.reshape(mask,(mask.shape[0],mask.shape[1],1))
        wk = tf.reshape(self.w_k(X),(self.head_num,X.shape[0],X.shape[1],-1))
        wv = tf.reshape(self.w_v(X),(self.head_num,X.shape[0],X.shape[1],-1))
        # wq.shape == wk.shape == wv.shape == (12,10,256,64)
        scaled_attention_logit = self.scale(tf.matmul(wq,wk,transpose_b=True))
        # scaled_attention_logit = head,batch,ts,ts
        attention_weight = tf.nn.softmax(scaled_attention_logit,axis=-1)
        output = tf.reshape(tf.matmul(attention_weight,wv),(X.shape[0],X.shape[1],-1))
        output = self.linear(output)
        return attention_weight,output

class EncoderBlock(tf.keras.Model):
    def __init__(self,config,pool_size):
        super(EncoderBlock,self).__init__()
        self.config = config
        self.hidden_size = self.config.d_model/pool_size # pool_size = 16,8,4,2,1로 고정
        self.pool = tf.keras.layers.AveragePooling2D((pool_size,pool_size))
        self.up_pool = tf.keras.layers.UpSampling2D((pool_size,pool_size))
        self.pool_mask = tf.keras.layers.AveragePooling1D(pool_size)
        self.MultiHeadAttention = MultiHeadAttention(self.config,pool_size)
        self.MHA_Dropout = tf.keras.layers.Dropout(self.config.dropout)
        self.MHA_Normalization = tf.keras.layers.LayerNormalization(epsilon=self.config.layer_norm_epsilon)

        self.FFN = tf.keras.Sequential([
            tf.keras.layers.Dense(self.config.d_model * 4),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(self.config.d_model)])
        self.FFN_Dropout = tf.keras.layers.Dropout(self.config.dropout)
        self.FFN_Normalization = tf.keras.layers.LayerNormalization(epsilon=self.config.layer_norm_epsilon)

    def call(self,X,mask):
        pooled_X = tf.squeeze(self.pool(tf.expand_dims(X,axis=-1)))
        normalized_X = self.MHA_Normalization(pooled_X)
        mask = self.pool_mask(tf.expand_dims(mask,axis=-1))
        attention_weight, attention_output = self.MultiHeadAttention(normalized_X,mask)
        attention_output = self.up_pool(tf.expand_dims(self.MHA_Dropout(attention_output),axis=-1))
        attention_output = X + tf.squeeze(attention_output)

        normalized_attention_output = self.FFN_Normalization(attention_output)
        FFN_output = X + self.FFN_Dropout(self.FFN(normalized_attention_output))

        return attention_weight, FFN_output

class EmbeddingLayer(tf.keras.Model):
    def __init__(self,config):
        super(EmbeddingLayer, self).__init__()
        self.config = config
        self.embedding = tf.keras.layers.Dense(self.config.d_model)

    def call(self,batch_data):  # batch_data (batch,ts)
        batch_data = tf.expand_dims(batch_data,axis=-1)
        embeddings = self.embedding(batch_data)
        embeddings = tf.add(embeddings, self.positional_encoding(self.config.max_len, self.config.d_model))
        return embeddings

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, max_length, hidden_size):
        angle_rads = self.get_angles(np.arange(max_length)[:, np.newaxis], np.arange(hidden_size)[np.newaxis, :],
                                     hidden_size)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    
class MaskedLanguageModel(tf.keras.Model):
    def __init__(self,vocab_size):
        super(MaskedLanguageModel,self).__init__()
        self.linear = tf.keras.layers.Dense(vocab_size)
    def call(self, x):
        return tf.nn.softmax(self.linear(x),axis=-1)

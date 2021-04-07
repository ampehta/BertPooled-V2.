import numpy as np
import tensorflow as tf

class MultiHeadAttention(tf.keras.Model):
    def __init__(self,d_model,head_num):
        super(MultiHeadAttention,self).__init__()
        self.d_model = d_model
        self.head_num = head_num

        self.w_q = tf.keras.layers.Dense(d_model)
        self.w_k = tf.keras.layers.Dense(d_model)
        self.w_v = tf.keras.layers.Dense(d_model)

        self.scale = tf.keras.layers.Lambda(lambda x: x/np.sqrt(d_model))
        self.linear = tf.keras.layers.Dense(d_model)

    def call(self,X,mask): # X = q,k,v / mask = (batch_size,time_sequence)
        assert X.shape[-1] % self.head_num == 0

        wq = tf.reshape(self.w_q(X),(self.head_num,X.shape[0],X.shape[1],-1)) * tf.reshape(mask,(mask.shape[0],mask.shape[1],1))
        wk = tf.reshape(self.w_k(X),(self.head_num,X.shape[0],X.shape[1],-1))
        wv = tf.reshape(self.w_v(X),(self.head_num,X.shape[0],X.shape[1],-1))
        # wq.shape == wk.shape == wv.shape == (12,10,256,64)
        scaled_attention_logit = self.scale(tf.matmul(wq,wk,transpose_b=True))
        # scaled_attention_logit = head,batch,ts,ts
        attention_weight = tf.nn.softmax(scaled_attention_logit,axis=-1)
        output = tf.reshape(tf.matmul(attention_weight,wv),X.shape)
        output = self.linear(output)

        return output,attention_weight,scaled_attention_logit

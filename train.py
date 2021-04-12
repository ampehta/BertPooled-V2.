class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

class PooledBertTrainer():
    def __init__(self,tokenizer,config):
        self.Bert = PooledBert(tokenizer,config)
        self.config = config
        self.learning_rate = CustomSchedule(self.config.d_model,warmup_steps=1000)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98,epsilon=1e-9)
        self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
        
    def call(self,data):
        steps = 0
        for epoch in range(self.config.epochs):
            self.MLM_train_step(data)
            steps+=1
            print(steps)

        return self.Bert


    def MLM_train_step(self,data):
        X = data# X: masked
        with tf.GradientTape() as tape:
            prediction,loss_mask,label = self.Bert(X,train=True)
            loss = self.MLM_loss_func(label,prediction,loss_mask)

        gradients = tape.gradient(loss, self.Bert.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.Bert.trainable_variables))
        

    def MLM_loss_func(self,label,prediction,loss_mask):
        prediction = tf.boolean_mask(prediction,loss_mask,axis=0)
        return self.loss_func(label,prediction)

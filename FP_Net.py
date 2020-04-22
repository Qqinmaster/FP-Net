import os
from dataset import Dataset
from common  import cf
import tensorflow as tf
from tensorflow.python.framework import ops
from Vector_cal_tf import NB_algorithm

import math
import numpy as np

tf.reset_default_graph()


def get_batch_dataset(data):
    dataset = Dataset(data)                  
    dataset  = dataset.load()
    data_iter = dataset.batch_iter()
    return data_iter,dataset 


class FlipGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0
    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls
        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]
        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)
        self.num_calls += 1
        return y


flip_gradient = FlipGradientBuilder()


class Net(object):
    def __init__(self,sess):
        self.sess = sess
        # define placeholders
        self.data1 = tf.placeholder(tf.int32,shape = [None,cf.max_sentence_len],name = 'data1')
        self.data1_len = tf.placeholder(tf.int32,shape = [None],name = 'data1_len')
        self.label = tf.placeholder(tf.int32,shape = [None],name = 'label')
        self.label_onehot = tf.one_hot(self.label,cf.num_class,name = 'label_onehot')
        self.is_training  = tf.placeholder(tf.bool, name='is_training')
        # moment lr
        self.l = tf.placeholder(tf.float32, [], name='l')
        self.lr = tf.placeholder(tf.float32, [], name='lr')
        # word embedding
        self.word_embedding_initializer = tf.random_normal_initializer(stddev = 0.1)
        self.embedding1 = tf.get_variable(name = 'word_embedding',
                                          shape = [cf.vocab_size,cf.embedding_size],
                                          dtype = tf.float32,
                                          initializer = self.word_embedding_initializer)
        self.embedding2 = tf.Variable(initial_value = self.embedding1,
                                      trainable = False,
                                      name = 'embedding2')
        # build model
        self.pred_label,self.ts_loss,self.or_loss = self.build(self.embedding1,self.embedding1,self.data1,self.data1_len,self.label_onehot,self.lr,self.l,self.is_training)

        self.global_steps = tf.Variable(0,trainable = False)
        self.Opt = tf.train.AdamOptimizer(cf.learning_rate_base)
        self.train_opt1 = self.Opt.minimize(self.or_loss,global_step = self.global_steps)
        self.train_opt2 = self.Opt.minimize(self.ts_loss, global_step=self.global_steps)
        # self.train_opt2 = tf.train.MomentumOptimizer(self.lr,0.9).minimize(self.ts_loss)

    def build(self,embedding1,embedding2,sentence,sentence_len,label_onehot,lr,l,is_training):  
        ts_loss,ts_output = self.Trivial_semantics(embedding2,sentence,sentence_len,label_onehot,lr,l,is_training)
        with tf.variable_scope('main_mold'):
            politary_stack,politary_emb = self.encoder(embedding1,sentence,sentence_len,'main_mold_transformer')
            grl_shape = politary_emb.get_shape().as_list()
            total_filter = grl_shape[1]*grl_shape[2]
            politary_emb = tf.reshape(politary_emb,shape =[-1,total_filter])
            politary_output = NB_algorithm(politary_emb,ts_output)
            politary_output = self.bn_layer(politary_output,is_training = self.is_training)
            regularizer = tf.contrib.layers.l2_regularizer(cf.regularizer)

            with tf.variable_scope('or_layer1',reuse = tf.AUTO_REUSE):
                fc1_weights = tf.get_variable('weight',[total_filter,cf.layer1_size],initializer = tf.truncated_normal_initializer(stddev = 0.1))
                fc1_biases = tf.get_variable('biases',[cf.layer1_size],initializer = tf.constant_initializer(0.1))
                fc1 = tf.nn.relu(tf.tensordot(politary_output,fc1_weights,axes = 1) + fc1_biases)
                if is_training is not None:
                    tf.add_to_collection('losses1',regularizer(fc1_weights))
                    fc1 = tf.nn.dropout(fc1,0.7)
                        
            with tf.variable_scope('or_layer2',reuse = tf.AUTO_REUSE):
                fc2_weights = tf.get_variable('weight',[cf.layer1_size,cf.num_class],initializer = tf.truncated_normal_initializer(stddev = 0.1))
                fc2_biases = tf.get_variable('biases',[cf.num_class],initializer = tf.constant_initializer(0.1))
                logits = tf.tensordot(fc1,fc2_weights,axes = 1) + fc2_biases
                if is_training is not None:
                    tf.add_to_collection('losses1',regularizer(fc2_weights))
                    
        loss = tf.nn.softmax_cross_entropy_with_logits(labels = tf.reshape(label_onehot,(cf.minibatch_size,cf.num_class)),logits = logits)
        loss_reg = tf.add_n(tf.get_collection('losses1'))
        or_loss = tf.reduce_mean(loss) + loss_reg
        pred_label = tf.argmax(tf.nn.softmax(logits),axis =-1)
        return pred_label,ts_loss,or_loss
    
    def Trivial_semantics(self,embedding,sentence,sentence_len,label_onehot,lr,l,is_training):
        sen_emb_stack,sen_emb = self.encoder(embedding,sentence,sentence_len,name = 'Trivial_stack')
        grl = flip_gradient(sen_emb,l)
        grl_shape = grl.get_shape().as_list()
        total_filter = grl_shape[1]*grl_shape[2]
        grl = tf.reshape(grl,shape =[-1,total_filter])
        regularizer = tf.contrib.layers.l2_regularizer(cf.regularizer)
        with tf.variable_scope('domain_layer1',reuse = tf.AUTO_REUSE):
            fc1_weights = tf.get_variable('weight',[total_filter,cf.layer1_size],initializer = tf.truncated_normal_initializer(stddev = 0.1))
            fc1_biases = tf.get_variable('biases',[cf.layer1_size],initializer = tf.constant_initializer(0.1))
            fc1 = tf.nn.relu(tf.tensordot(grl,fc1_weights,axes = 1) + fc1_biases)
            if is_training is not None:
                tf.add_to_collection('losses2',regularizer(fc1_weights))
                fc1 = tf.nn.dropout(fc1,0.7)
                    
        with tf.variable_scope('domain_layer2',reuse = tf.AUTO_REUSE):
            fc2_weights = tf.get_variable('weight',[cf.layer1_size,cf.num_class],initializer = tf.truncated_normal_initializer(stddev = 0.1))
            fc2_biases = tf.get_variable('biases',[cf.num_class],initializer = tf.constant_initializer(0.1))
            logits = tf.tensordot(fc1,fc2_weights,axes = 1) + fc2_biases
            if is_training is not None:
                tf.add_to_collection('losses2',regularizer(fc2_weights))
        loss = tf.nn.softmax_cross_entropy_with_logits(labels = tf.reshape(label_onehot,(cf.minibatch_size,cf.num_class)),logits = logits)
        loss_reg = tf.add_n(tf.get_collection('losses2'))
        loss = tf.reduce_mean(loss) + loss_reg
        sem_out = tf.reshape(sen_emb,shape =[-1,total_filter]) 
        return loss,sem_out

    def bn_layer(self,x,is_training,name='BatchNorm',moving_decay=0.9,eps=1e-5):

        shape = x.shape
        assert len(shape) in [2,4]
        param_shape = shape[-1]
        with tf.variable_scope(name):
            gamma = tf.get_variable('gamma',param_shape,initializer=tf.constant_initializer(1))
            beta = tf.get_variable('beat', param_shape,initializer=tf.constant_initializer(0))
            axes = list(range(len(shape)-1))
            batch_mean, batch_var = tf.nn.moments(x,axes,name='moments')
            ema = tf.train.ExponentialMovingAverage(moving_decay)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean,batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(tf.equal(is_training,True),mean_var_with_update,lambda:(ema.average(batch_mean),ema.average(batch_var)))
        return tf.nn.batch_normalization(x,mean,var,beta,gamma,eps)
    
    def encoder(self,word_embedding,sentence,sentence_len,name):
        sen_emb = tf.nn.embedding_lookup(word_embedding,sentence)
        with tf.variable_scope(name):
            if cf.is_positional and cf.stack_num > 0:
                with tf.variable_scope('positional',reuse = tf.AUTO_REUSE):
                    sen_emb = self.positional_encoding_vector(sen_emb, max_timescale = 10)
        
            sen_emb_stack = [sen_emb]
            for index in range(cf.stack_num):
                with tf.variable_scope('self_stack_' + str(index),reuse = tf.AUTO_REUSE):
                    sen_emb = self.block(sen_emb,sen_emb,sen_emb,Q_lengths = sentence_len,K_lengths = sentence_len)
                    sen_emb_stack.append(sen_emb)

        return sen_emb_stack,sen_emb

    def positional_encoding_vector(self,x,min_timescale = 1.0,max_timescale = 1.0e4,value = 0):
        length = x.shape[1]
        channels = x.shape[2]
        _lambda = tf.get_variable(name = 'lambda',
                                  shape = [length],
                                  dtype = tf.float32,
                                  initializer = tf.constant_initializer(value))
        _lambda = tf.expand_dims(_lambda,axis = -1)
        position = tf.to_float(tf.range(length))
        num_timescales = channels // 2    
        log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / 
                                   (tf.to_float(num_timescales) - 1))
        inv_timescales = min_timescale * tf.exp(tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
        scaled_time = tf.expand_dims(position,1) * tf.expand_dims(inv_timescales,0)
        signal = tf.concat([tf.sin(scaled_time),tf.cos(scaled_time)],axis= 1)
        signal = tf.pad(signal,[[0,0],[0,tf.mod(channels,2)]])
        signal = tf.multiply(_lambda,signal)
        signal = tf.expand_dims(signal,axis = 0)
        return x + signal
    
    def dot_sim(self,x,y,is_nor = True):
        assert x.shape[-1] == y.shape[-1]
        sim = tf.einsum('bik,bjk->bij',x,y)
        if is_nor:
            scale = tf.sqrt(tf.cast(x.shape[-1],tf.float32))
            scale = tf.maximum(1.0,scale)
            return sim / scale
        else:
            return sim
    
    def mask(self,row_lengths,col_lengths,max_row_lengrh,max_col_length):
        row_mask = tf.sequence_mask(row_lengths,max_row_lengrh)
        col_mask = tf.sequence_mask(col_lengths,max_col_length)
        
        row_mask = tf.cast(tf.expand_dims(row_mask,-1),tf.float32)
        col_mask = tf.cast(tf.expand_dims(col_mask,-1),tf.float32)
        
        return tf.einsum('bik,bjk->bij',row_mask,col_mask)
    
    def weighted_sum(self,weight,values):
        return tf.einsum('bij,bjk->bik',weight,values)
    
    def attention(self,Q,K,V,Q_lengths,K_lengths,attention_type = 'dot',is_mask = True,mask_value = -2**32+1,drop_prob = None):
        assert attention_type in('dot','bilinear')
        if attention_type == 'dot':
            assert Q.shape[-1] == K.shape[-1]
        Q_time = Q.shape[1]
        K_time = K.shape[1]
        if attention_type == 'dot':
            logits = self.dot_sim(Q,K)
        if is_mask:
            mask = self.mask(Q_lengths,K_lengths,Q_time,K_time)
            logits = mask * logits + (1-mask) * mask_value
        attention = tf.nn.softmax(logits)
        if drop_prob is not None:
            attention = tf.nn.dropout(attention,drop_prob)
        return self.weighted_sum(attention,V)
    
    def layer_norm_debug(self,x,axis = None,epsilon = 1e-6):
        if axis is None:
            axis = [-1]
        shape = [x.shape[i] for i in axis]
        
        scale = tf.get_variable(name= 'scale',shape = shape,dtype = tf.float32,initializer = tf.ones_initializer())
        bias = tf.get_variable(name = 'bias',shape = shape,dtype = tf.float32,initializer = tf.zeros_initializer())
        
        mean = tf.reduce_mean(x,axis= axis ,keepdims = True)
        variance = tf.reduce_mean(tf.square(x-mean),axis = axis,keepdims = True)
        norm = (x - mean) * tf.rsqrt(variance + epsilon)
        return scale * norm + bias

    def dense(self,x, out_dimension = None, add_bias = True):
        if out_dimension is None:
            out_dimension = x.shape[-1]
    
        W = tf.get_variable(
            name='weights',
            shape=[x.shape[-1], out_dimension],
            dtype=tf.float32,
            initializer=tf.orthogonal_initializer())
        if add_bias:
            bias = tf.get_variable(
                name='bias',
                shape=[1],
                dtype=tf.float32,
                initializer=tf.zeros_initializer())
            return tf.einsum('bik,kj->bij', x, W) + bias
        else:
            return tf.einsum('bik,kj->bij', x, W)
    
    def FNN(self,x,out_dimension_0 = None,out_dimension_1 = None):
        with tf.variable_scope('FFN_1'):
            y = self.dense(x, out_dimension_0)
            y = tf.nn.relu(y)
        with tf.variable_scope('FFN_2'):
            z = self.dense(y, out_dimension_1)
        return z
    
    def block(self,Q,K,V,Q_lengths,K_lengths,attention_type = 'dot',is_layer_norm = True,is_mask = True,mask_value = -2**32+1,drop_prob = None):
        att = self.attention(Q,K,V,Q_lengths,K_lengths,attention_type = 'dot',is_mask = is_mask,mask_value = mask_value,drop_prob = drop_prob)
        if is_layer_norm:
            with tf.variable_scope('attention_layer_norm'):
                y = self.layer_norm_debug(Q+att)
        else:
             y = Q + att
             
        z = self.FNN(y)
        if is_layer_norm:
            with tf.variable_scope('FNN_layer_norm'):
                w = self.layer_norm_debug(y+z)
        else:
            w = y+z
        return w

    def loss_acc(self,x,y,num_classes = cf.num_class,is_clip = False,clip_value = 10):
        assert isinstance(num_classes,int)
        assert num_classes >= 2
        y = tf.cast(y,tf.int32)
        y_hot = tf.one_hot(y,num_classes)
        w = tf.get_variable(name = 'weights',shape = [x.shape[-1],num_classes],
                            initializer = tf.orthogonal_initializer())
        bias = tf.get_variable(name = 'bias',shape = [num_classes],
                               initializer = tf.zeros_initializer())
        logits = tf.matmul(x,w) + bias

        pre = tf.nn.softmax(logits,name = 'pred')
        correct = tf.equal(tf.argmax(pre,axis = -1),tf.argmax(y_hot,axis = -1))
        accuracy =  tf.reduce_mean(tf.cast(correct,tf.float32))

        loss = tf.nn.softmax_cross_entropy_with_logits(labels = y_hot,logits = logits)
        loss = tf.reduce_mean(loss)
        if is_clip:
            loss = tf.clip_by_value(loss,clip_value)
        return accuracy,loss,logits
    
    def train(self,sentence,label,sentence_len,lr,l,is_training):
        return self.sess.run([self.pred_label,self.train_opt1,self.train_opt2,self.or_loss],feed_dict = {
                self.data1:sentence,
                self.label:label,
                self.data1_len:sentence_len,
                self.l:l,
                self.lr:lr,
                self.is_training:is_training})
    
    def get_pred_label(self,sentence,label,sentence_len,lr,l,is_training):
        return self.sess.run([self.pred_label,self.or_loss],feed_dict = {
                self.data1:sentence,
                self.label:label,
                self.data1_len:sentence_len,
                self.l:l,
                self.lr:lr,
                self.is_training:is_training})

       
def main():
    tf.set_random_seed(1234)
    global_cnt  = 0
    epoch_start = 0
    tf_config = tf.ConfigProto(allow_soft_placement = True,log_device_placement = False)  
    tf_config.gpu_options.allow_growth = True 
    with tf.Session(config = tf_config) as sess:
        model = Net(sess)
        sess.run(tf.global_variables_initializer())

        Fina_test_acc = []
        for epoch in range(epoch_start+1, cf.nr_epoch+1):
            p   = epoch / float(cf.nr_epoch)
            lp  = 2. / (1. + np.exp(-10*p)) - 1
            lr  = 0.01 / (1 + 10.*p)**0.75
            lp2 = [0.05,0.1,0.2,0.4,0.8,1.0]
            train_data,traindata = get_batch_dataset('train')
            stage = np.arange(0,traindata.minibatchs_per_epoch,int(traindata.minibatchs_per_epoch / len(lp2))).tolist()
            
            for t in range(traindata.minibatchs_per_epoch):
                global_cnt += 1
                train_batch_gnr = next(train_data)
                try:
                    for u in range(len(lp2)):
                        if t in np.arange(stage[u],stage[u+1]):
                            lp = lp2[u]
                except IndexError:
                    for u in range(len(lp2)-1):
                        if t in np.arange(stage[u],stage[u+1]):
                            lp = lp2[u]
                prelabel,train1_opt,train2_opt,loss_t= model.train(train_batch_gnr[0],train_batch_gnr[1],train_batch_gnr[2].reshape((cf.minibatch_size,)),lr,lp,is_training = True)
                train_acc = np.mean(train_batch_gnr[1] == prelabel)
                if global_cnt % cf.show_interval == 0:
                    print('In epoch {} after {} steps, the loss is {} ,train_acc is {}'.format(epoch,global_cnt,loss_t,train_acc))
                    test_data,testdata = get_batch_dataset('test')
                    all_loss = 0
                    True_label = []
                    Pred_lavel = []
                    for _ in range(testdata.minibatchs_per_epoch):
                        test_batch_gnr = next(test_data)
                        prelabel_test,loss_test= model.get_pred_label(test_batch_gnr[0],test_batch_gnr[1],test_batch_gnr[2].reshape((cf.minibatch_size,)),lr,lp,is_training = False)
                        all_loss += loss_test
                        True_label.extend(test_batch_gnr[1].tolist())
                        Pred_lavel.extend(prelabel_test.tolist())
                        
                    test_acc = np.mean(np.array(True_label)==np.array(Pred_lavel))
                    ave_loss = all_loss / testdata.minibatchs_per_epoch
                    Fina_test_acc.append(test_acc)
                    print('-------------------------------******************-----------------------------')
                    print('After {} epoch,the test-loss is {},the test_acc is {}'.format(epoch,ave_loss,test_acc))
                    print('-------------------------------******************-----------------------------')
                    
        Fina_test_acc = np.array(Fina_test_acc)
        print('***************Finally Fina_test_acc.max is {}*************'.format(Fina_test_acc.max()))

if __name__ == "__main__":
    main()
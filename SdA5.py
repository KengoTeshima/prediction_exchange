# -*- coding: utf-8 -*-

import numpy as np
from chainer import cuda, Variable, optimizers, Chain
import chainer.functions as F
import chainer.links as L
from chainer import serializers

class DA:
    def __init__(
            self,
            rng,
            data,
            n_inputs=784,
            n_hidden=784,
            corruption_level=0.05,
            optimizer=optimizers.Adam,
            gpu=-1
    ):
        """
        Denoising AutoEncoder
        data: data for train
        n_inputs: a number of units of input layer and output layer
        n_hidden: a number of units of hidden layer
        corruption_level: a ratio of masking noise
        """

        self.model = Chain(encoder=L.Linear(n_inputs, n_hidden),
                           decoder=L.Linear(n_hidden, n_inputs))

        if gpu >= 0:
            self.model.to_gpu()
            self.xp = cuda.cupy
        else:
            self.xp = np

        self.gpu = gpu

        self.x_train, self.x_test = data

        self.n_train = len(self.x_train)
        self.n_test = len(self.x_test)

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden

        self.optimizer = optimizer()
        self.optimizer.setup(self.model)
        self.corruption_level = corruption_level
        self.rng = rng

    def forward(self, x_data, train=True):
        y_data = x_data
        # add noise (masking noise)
        x_data = self.get_corrupted_inputs(x_data, train=train)

        x, t = Variable(x_data.reshape(y_data.shape)), Variable(y_data.reshape(x_data.shape))

        # encode
        h = self.encode(x)
        # decode
        y = self.decode(h)
        # compute loss
        loss = F.mean_squared_error(y, t)
        return loss

    def compute_hidden(self, x_data):
        # x_data = self.xp.asarray(x_data)
        x = Variable(x_data)
        h = self.encode(x)
        # return cuda.to_cpu(h.data)
        return h.data

    def predict(self, x_data):
        x = Variable(x_data)
        # encode
        h = self.encode(x)
        # decode
        y = self.decode(h)
        return cuda.to_cpu(y.data)

    def encode(self, x):
        return F.relu(self.model.encoder(x))

    def decode(self, h):
        return F.relu(self.model.decoder(h))

    def encoder(self):
        initialW = self.model.encoder.W
        initial_bias = self.model.encoder.b

        return L.Linear(self.n_inputs,
                        self.n_hidden,
                        initialW=initialW,
                        initial_bias=initial_bias)

    def decoder(self):
        return self.model.decoder

    def to_cpu(self):
        self.model.to_cpu()
        self.xp = np

    def to_gpu(self):
        if self.gpu < 0:
            print "something wrong"
            raise
        self.model.to_gpu()
        self.xp = cuda.cupy

    # masking noise
    def get_corrupted_inputs(self, x_data, train=True):
        if train and self.corruption_level != 0.0:
            mask = self.rng.binomial(size=x_data.shape, n=1, p=1.0-self.corruption_level)
            mask = mask.astype(np.float32)
            mask = self.xp.asarray(mask)
            ret = mask * x_data
            # return self.xp.asarray(ret.astype(np.float32))
            return ret
        else:
            return x_data


    def train_and_test(self, n_epoch=5, batchsize=100):
        self.save_accuracy = self.xp.tile([1000.0],2000)
        self.best_loss = 1.0

        for epoch in xrange(1, n_epoch+1):
            print 'epoch', epoch

            perm = self.rng.permutation(self.n_train)
            sum_loss = 0
            for i in xrange(0, self.n_train, batchsize):
                x_batch = self.xp.asarray(self.x_train[perm[i:i+batchsize]])

                real_batchsize = len(x_batch)

                self.optimizer.zero_grads()
                loss = self.forward(x_batch)
                loss.backward()
                self.optimizer.update()

                sum_loss += float(loss.data) * real_batchsize

            print 'train mean loss={}'.format(sum_loss/self.n_train)

            # evaluation
            sum_loss = 0
            for i in xrange(0, self.n_test, batchsize):
                x_batch = self.xp.asarray(self.x_test[i:i+batchsize])

                real_batchsize = len(x_batch)
                loss = self.forward(x_batch, train=False)

                sum_loss += float(loss.data) * real_batchsize

            print 'test mean loss={}'.format(sum_loss/self.n_test)

            if (sum_loss/self.n_test) < self.best_loss:
                self.best_loss = sum_loss/self.n_test
                self.best_epoch = epoch
                serializers.save_hdf5('dae.model', self.model)
                print("update best loss")

            #早期終了？
            if self.xp.mean(self.save_accuracy) < sum_loss:
                print("early stopping done")
                break

            #早期終了用配列にsum_accuracyを追加
            self.save_accuracy = self.save_accuracy[1:]
            append = self.xp.array([float(sum_loss)])
            self.save_accuracy = self.xp.hstack((self.save_accuracy,append))

        print ("best_epoch: %d" % (self.best_epoch))
        serializers.load_hdf5("dae.model", self.model)

class SDA:
    def __init__(
            self,
            rng,
            data,
            target,
            n_inputs=784,
            n_hidden=[784,784,784,784,784],
            n_outputs=1,
            corruption_levels=[0.1,0.1,0.1,0.1,0.1],
            gpu=-1):

        self.model = Chain(l1=L.Linear(n_inputs, n_hidden[0]),
                           l2=L.Linear(n_hidden[0], n_hidden[1]),
                           l3=L.Linear(n_hidden[1], n_hidden[2]),
                           l4=L.Linear(n_hidden[2], n_hidden[3]),
                           l5=L.Linear(n_hidden[3], n_hidden[4]),
                           l6=L.Linear(n_hidden[4], n_outputs))

        if gpu >= 0:
            self.model.to_gpu()
            self.xp = cuda.cupy
        else:
            self.xp = np

        self.rng = rng
        self.gpu = gpu
        self.data = data
        self.target = target

        self.x_train, self.x_test = data
        self.y_train, self.y_test = target

        self.n_train = len(self.y_train)
        self.n_test = len(self.y_test)

        self.corruption_levels = corruption_levels
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.hidden_size = len(n_hidden)

        self.dae1 = None
        self.dae2 = None
        self.dae3 = None
        self.dae4 = None
        self.dae5 = None
        self.optimizer = None
        self.setup_optimizer()

    def setup_optimizer(self):
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

    def dae_train(self, rng, n_epoch, batchsize, dae_num, data, n_inputs, n_hidden, corruption_level, gpu):
        #initialize
        dae = DA(rng=rng,
                 data=data,
                 n_inputs=n_inputs,
                 n_hidden=n_hidden,
                 corruption_level=corruption_level,
                 gpu=gpu)

        #train
        print "--------DA%d training has started!--------" % dae_num
        dae.train_and_test(n_epoch=n_epoch, batchsize=batchsize)
        dae.to_cpu()
        # compute outputs for next dAE
        tmp1 = dae.compute_hidden(data[0])
        tmp2 = dae.compute_hidden(data[1])
        if gpu >= 0:
            dae.to_gpu()
        next_inputs = [tmp1,tmp2]
        return dae,next_inputs

    def pre_train(self, n_epoch=20, batchsize=40, sda_name="SDA"):
        first_inputs = self.data
        n_epoch1 = n_epoch
        batchsize1 = batchsize

        # initialize first dAE
        self.dae1, second_inputs = self.dae_train(self.rng,
                                       n_epoch=n_epoch,
                                       batchsize=batchsize,
                                       dae_num=1,
                                       data=first_inputs,
                                       n_inputs=self.n_inputs,
                                       n_hidden=self.n_hidden[0],
                                       corruption_level=self.corruption_levels[0],
                                       gpu=self.gpu)

        self.dae2, third_inputs = self.dae_train(self.rng,
                                       n_epoch=int(n_epoch),
                                       batchsize=batchsize,
                                       dae_num=2,
                                       data=second_inputs,
                                       n_inputs=self.n_hidden[0],
                                       n_hidden=self.n_hidden[1],
                                       corruption_level=self.corruption_levels[1],
                                       gpu=self.gpu)

        self.dae3, forth_inputs = self.dae_train(self.rng,
                                       n_epoch=int(n_epoch),
                                       batchsize=batchsize,
                                       dae_num=3,
                                       data=third_inputs,
                                       n_inputs=self.n_hidden[1],
                                       n_hidden=self.n_hidden[2],
                                       corruption_level=self.corruption_levels[2],
                                       gpu=self.gpu)

        self.dae4, fifth_inputs = self.dae_train(self.rng,
                                       n_epoch=int(n_epoch),
                                       batchsize=batchsize,
                                       dae_num=4,
                                       data=forth_inputs,
                                       n_inputs=self.n_hidden[2],
                                       n_hidden=self.n_hidden[3],
                                       corruption_level=self.corruption_levels[3],
                                       gpu=self.gpu)

        self.dae5, sixth_inputs = self.dae_train(self.rng,
                                       n_epoch=int(n_epoch),
                                       batchsize=batchsize,
                                       dae_num=5,
                                       data=fifth_inputs,
                                       n_inputs=self.n_hidden[3],
                                       n_hidden=self.n_hidden[4],
                                       corruption_level=self.corruption_levels[4],
                                       gpu=self.gpu)

        # update model parameters
        self.model.l1 = self.dae1.model.encoder
        self.model.l2 = self.dae2.model.encoder
        self.model.l3 = self.dae3.model.encoder
        self.model.l4 = self.dae4.model.encoder
        self.model.l5 = self.dae5.model.encoder

        self.setup_optimizer()

        model_file = "%s.model"%sda_name
        state_file = "%s.state"%sda_name

        serializers.save_hdf5(model_file,self.model)
        serializers.save_hdf5(state_file,self.optimizer)

    def forward(self, x_data, y_data, train=True, output=False):
        x, t = Variable(x_data), Variable(y_data)
        h1 = F.dropout(F.relu(self.model.l1(x)), train=train)
        h2 = F.dropout(F.relu(self.model.l2(h1)), train=train)
        h3 = F.dropout(F.relu(self.model.l3(h2)), train=train)
        h4 = F.dropout(F.relu(self.model.l4(h3)), train=train)
        h5 = F.dropout(F.relu(self.model.l5(h4)), train=train)
        y = F.tanh(self.model.l6(h5))
        if output:
            return y
        else:
            return F.mean_squared_error(y, t)

    def fine_tune(self, n_epoch=20, batchsize=50):
        train_accs = []
        test_accs = []

        #早期終了用配列
        self.save_accuracy = self.xp.tile([1000.0],100)

        #ベストLOSS定義
        self.best_loss = 1000.0

        for epoch in xrange(1, n_epoch+1):
            print 'fine tuning epoch ', epoch

            perm = self.rng.permutation(self.n_train)
            sum_loss = 0
            for i in xrange(0, self.n_train, batchsize):
                x_batch = self.xp.asarray(self.x_train[perm[i:i+batchsize]])
                y_batch = self.xp.asarray(self.y_train[perm[i:i+batchsize]])

                real_batchsize = len(x_batch)

                self.optimizer.zero_grads()
                loss = self.forward(x_batch, y_batch)

                loss.backward()
                self.optimizer.update()

                sum_loss += float(cuda.to_cpu(loss.data)) * real_batchsize

            print 'fine tuning train mean loss={}'.format(sum_loss/self.n_train)
            train_accs.append(sum_loss/self.n_train)

            # evaluation
            sum_loss = 0
            for i in xrange(0, self.n_test, batchsize):
                x_batch = self.xp.asarray(self.x_test[i:i+batchsize])
                y_batch = self.xp.asarray(self.y_test[i:i+batchsize])

                real_batchsize = len(x_batch)

                loss = self.forward(x_batch, y_batch, train=False)

                sum_loss += float(cuda.to_cpu(loss.data)) * real_batchsize

            print 'fine tuning test mean loss={}'.format(sum_loss/self.n_test)
            test_accs.append(sum_loss/self.n_test)

            if sum_loss < self.best_loss:
                self.best_loss = sum_loss
                self.best_epoch = epoch
                serializers.save_hdf5('mlp.model', self.model)
                print("update best loss")

            #早期終了？
            if self.xp.mean(self.save_accuracy) < sum_loss:
                print("early stopping done")
                break

            #早期終了用配列にsum_accuracyを追加
            self.save_accuracy = self.save_accuracy[1:]
            append = self.xp.array([float(sum_loss)])
            self.save_accuracy = self.xp.hstack((self.save_accuracy,append))

        print ("best_epoch: %d" % (self.best_epoch))
        serializers.load_hdf5("mlp.model", self.model)

        return train_accs, test_accs

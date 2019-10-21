import time
import numpy as np
import tensorflow as tf
from models import GAT
from utils import process
import argparse
import os

# checkpt_file = 'pre_trained/cora/mod_cora.ckpt'

# NUM_PARALLEL_EXEC_UNITS = 4
# os.environ['OMP_NUM_THREADS'] = str(NUM_PARALLEL_EXEC_UNITS)
# os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
#os.environ["KMP_AFFINITY"] = "verbose" # no affinity
#os.environ["KMP_AFFINITY"] = "none" # no affinity
os.environ["KMP_AFFINITY"] = "disabled" # completely disable thread pools

# Training settings
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

parser.add_argument('--dataset', type=str, default='cora', help='Which dataset to use ["cora", "citeseer", "Pubmed"]')
parser.add_argument('--data_dir', type=str, default='./data/', help='Directory for data.')

args = parser.parse_args()

# dataset = 'cora'

# training params
batch_size = 1
# nb_epochs = 100000
# patience = 100
# lr = 0.005  # learning rate
# l2_coef = 0.0005  # weight decay
# hid_units = [8] # numbers of hidden units per each attention head in each layer
# n_heads = [8, 1] # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu

model = GAT

print('Dataset: ' + args.dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(args.lr))
print('l2_coef: ' + str(args.weight_decay))
print('----- Archi. hyperparams -----')
# print('nb. layers: ' + str(len(args.hidden)))
print('nb. units per layer: ' + str(args.hidden))
print('nb. attention heads: ' + str(args.nb_heads))
# print('residual: ' + str(residual))
# print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

checkpt_file = f'pre_trained/{args.dataset}/best_model.ckpt'
if not os.path.exists(os.path.dirname(checkpt_file)):
    os.makedirs(os.path.dirname(checkpt_file))

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_image_data('./data/', args.dataset)
print(y_train)
print(y_train.shape)
print(train_mask)
# features, spars = process.preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]

adj = adj.todense()

features = features[np.newaxis]
adj = adj[np.newaxis]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)

with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
        bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes))
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
        attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        is_train = tf.placeholder(dtype=tf.bool, shape=())

    logits = model.inference(ftr_in, nb_classes, nb_nodes, is_train,
                             attn_drop, ffd_drop,
                             bias_mat=bias_in,
                             hid_units=[args.hidden], n_heads=[args.nb_heads, 1],
                             residual=residual, activation=nonlinearity)
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

    train_op = model.training(loss, args.lr, args.weight_decay)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    with tf.Session() as sess:
        sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

        for epoch in range(args.epochs):
            tr_step = 0
            tr_size = features.shape[0]

            while tr_step * batch_size < tr_size:
                _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
                    feed_dict={
                        ftr_in: features[tr_step*batch_size:(tr_step+1)*batch_size],
                        bias_in: biases[tr_step*batch_size:(tr_step+1)*batch_size],
                        lbl_in: y_train[tr_step*batch_size:(tr_step+1)*batch_size],
                        msk_in: train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                        is_train: True,
                        attn_drop: 0.6, ffd_drop: 0.6})
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1

            vl_step = 0
            vl_size = features.shape[0]

            while vl_step * batch_size < vl_size:
                loss_value_vl, acc_vl = sess.run([loss, accuracy],
                    feed_dict={
                        ftr_in: features[vl_step*batch_size:(vl_step+1)*batch_size],
                        bias_in: biases[vl_step*batch_size:(vl_step+1)*batch_size],
                        lbl_in: y_val[vl_step*batch_size:(vl_step+1)*batch_size],
                        msk_in: val_mask[vl_step*batch_size:(vl_step+1)*batch_size],
                        is_train: False,
                        attn_drop: 0.0, ffd_drop: 0.0})
                val_loss_avg += loss_value_vl
                val_acc_avg += acc_vl
                vl_step += 1

            print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                    (train_loss_avg/tr_step, train_acc_avg/tr_step,
                    val_loss_avg/vl_step, val_acc_avg/vl_step), flush=True)

            if val_acc_avg/vl_step >= vacc_mx or val_loss_avg/vl_step <= vlss_mn:
                if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
                    vacc_early_model = val_acc_avg/vl_step
                    vlss_early_model = val_loss_avg/vl_step
                    saver.save(sess, checkpt_file)
                vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
                vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == args.patience:
                    print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx, flush=True)
                    print('Early stop model validation loss: ',
                          vlss_early_model, ', accuracy: ', vacc_early_model, flush=True)
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

        saver.restore(sess, checkpt_file)

        ts_size = features.shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        while ts_step * batch_size < ts_size:
            loss_value_ts, acc_ts = sess.run([loss, accuracy],
                feed_dict={
                    ftr_in: features[ts_step*batch_size:(ts_step+1)*batch_size],
                    bias_in: biases[ts_step*batch_size:(ts_step+1)*batch_size],
                    lbl_in: y_test[ts_step*batch_size:(ts_step+1)*batch_size],
                    msk_in: test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                    is_train: False,
                    attn_drop: 0.0, ffd_drop: 0.0})
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1

        print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step, flush=True)

        sess.close()

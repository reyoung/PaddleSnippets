import paddle.v2 as paddle
import numpy

paddle.init(use_gpu=False, trainer_count=3)
img = paddle.layer.data(name='image', type=paddle.data_type.dense_vector(784))
hidden = paddle.layer.fc(input=img, size=200)

label = paddle.layer.data(name='label', type=paddle.data_type.integer_value(10))

with paddle.layer.mixed(size=10, act=paddle.activation.Softmax(),
                        bias_attr=paddle.attr.Param('softmax.b')) as prediction:
    prediction += paddle.layer.trans_full_matrix_projection(input=hidden,
                                                            param_attr=paddle.attr.Param(
                                                                name='softmax.w'))

cost_nce = paddle.layer.nce(input=hidden, label=label, num_classes=10,
                            param_attr=paddle.attr.Param(name='softmax.w'),
                            bias_attr=paddle.attr.Param("softmax.b"))

params = paddle.parameters.create([cost_nce, prediction])

optimizer = paddle.optimizer.Momentum(
    learning_rate=0.1 / 128.0,
    momentum=0.9,
    regularization=paddle.optimizer.L2Regularization(rate=0.0005 * 128))

trainer = paddle.trainer.SGD(cost=cost_nce,
                             parameters=params,
                             update_equation=optimizer)


def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % 100 == 0:
            print "Pass %d, Batch %d, Cost %f" % (
                event.pass_id, event.batch_id, event.cost)


trainer.train(
    reader=paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=8192),
        batch_size=128),
    event_handler=event_handler,
    num_passes=20)

mat = []
labels = []
for img, lbl in paddle.dataset.mnist.train()():
    mat.append((img,))
    labels.append(lbl)

probs = paddle.infer(output_layer=prediction, parameters=params, input=mat)

err_cnt = 0
for i in xrange(len(probs)):
    if numpy.argmax(probs[i]) != labels[i]: err_cnt += 1

print 'Testing error rate %.2f%%' % (float(err_cnt) * 100 / len(probs))

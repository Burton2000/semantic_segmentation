import tensorflow as tf


def lr_schedule(epoch):
    """Returns a custom learning rate that decreases as epochs progress."""

    learning_rate = 0.01
    if epoch > 180:
        learning_rate = 0.001
    if epoch > 200:
        learning_rate = 0.001
    if epoch > 300:
        learning_rate = 0.005

    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)

    return learning_rate

from preprocess import Preprocess
from config import Config
import loss_functions
from dafl import SSCAModule, GAMModule, DAFLModel
from F1score import F1Score
import tensorflow as tf
import matplotlib.pyplot as plt


config = Config()
preprocess = Preprocess()
x, X_test, y, y_test = preprocess.process()

x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)

x_val_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_val_tensor = tf.convert_to_tensor(y_test, dtype=tf.float32)

train_dataset = tf.data.Dataset.from_tensor_slices((x_tensor, y_tensor))
val_dataset = tf.data.Dataset.from_tensor_slices((x_val_tensor, y_val_tensor))

buffer_size = len(x_tensor)

train_dataset = train_dataset.shuffle(buffer_size).batch(config.batch_size)

buffer_size = len(x_val_tensor)

val_dataset = val_dataset.shuffle(buffer_size).batch(config.batch_size)


optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = config.learning_rate)
model = DAFLModel(config.num_attributes, config.num_groups, config.momentum, config.group_indices_list, config.num_cascades)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
# train_f1_score = F1Score(name='train_f1_score')
# train_precision = tf.keras.metrics.Precision(name='train_precision')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')
# val_f1_score = F1Score(name='val_f1_score')
# val_precision = tf.keras.metrics.Precision(name='val_precision')

checkpoint_path = config.checkpoint_path + "cp-{epoch:04d}.ckpt"
training_losses = []
validation_losses = []

# @tf.function
def train_step(model, x, y, optimizer):
    with tf.GradientTape() as tape:
        classification_output, embeddings, group_attentions, attention_map = model(x, config.num_attributes, training=True)
        classificationLoss = loss_functions.classification_loss(y, classification_output)
        triplet_loss = loss_functions.compute_triplet_loss(embeddings, y, config.num_attributes, classification_output)
        group_consistency_loss = loss_functions.compute_group_consistency_loss(group_attentions, attention_map, config.group_indices_list)
        
        loss = classificationLoss + triplet_loss + group_consistency_loss

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, classificationLoss, triplet_loss, group_consistency_loss

for epoch in range(config.num_epochs):
    train_loss.reset_states()
    train_accuracy.reset_states()
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        loss, classificationLoss, triplet_loss, group_consistency_loss = train_step(model, x_batch, y_batch, optimizer)
        train_loss.update_state(loss)
        train_accuracy.update_state(y_batch, model(x_batch, config.num_attributes, training=False)[0])
        
        if step % 100 == 0:  # Log every 10 steps
            print(f'Epoch {epoch + 1}, Step {step}, Loss: {train_loss.result().numpy()}, Accuracy: {train_accuracy.result().numpy() * 100}%')

    
    print(f'Epoch {epoch + 1} completed. Avg Loss: {train_loss.result().numpy()}, Avg Accuracy: {train_accuracy.result().numpy() * 100}%')
        # Logging and tracking loss, accuracy, etc.
    model.save_weights(checkpoint_path.format(epoch=epoch))
    val_loss.reset_states()
    val_accuracy.reset_states()
    training_losses.append(train_loss.result())
    
    for x_val, y_val in val_dataset:
        predictions = model(x_val, config.num_attributes, training=False)[0]
        v_loss = loss_functions.classification_loss(y_val, predictions)

        val_loss.update_state(v_loss)
        val_accuracy.update_state(y_val, predictions)

    print(f'Validation Loss: {val_loss.result().numpy()}, Validation Accuracy: {val_accuracy.result().numpy() * 100}%')
    validation_losses.append(val_loss.result())
    

def plot_losses(losses, train_loss):
    """
    Plots the loss values over steps.

    Args:
    losses (list or array): A list or array containing loss values at each step or epoch.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.title('Loss over Steps')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    isExist = os.path.exists(config.plot_save_path)
    if not isExist:
    	os.makedirs(config.plot_save_path)
    	
    if train_loss:
    	plt.savefig(config.plot_save_path + 'train_loss.png')
    else:
    	plt.savefig(config.plot_save_path + 'validation_loss.png')

print(type(training_losses[0]))
for i in training_losses:
    print(i)
plot_losses(list(training_losses), True)
plot_losses(validation_losses, False)


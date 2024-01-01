import tensorflow as tf
from keras import models, regularizers, layers, optimizers, losses, metrics

class SSCAModule(tf.keras.layers.Layer):
    def __init__(self, num_attributes):
        super(SSCAModule, self).__init__()
        self.num_attributes = num_attributes
        self.query_embedding = layers.Dense(units=num_attributes)
        self.feature_transform = layers.Conv2D(filters=2048, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True)
        self.feature_transform1 = layers.Conv2D(filters=2048, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True)
        self.feature_embedding = layers.Dense(units=num_attributes)

    def call(self, queries, features):
        Q = self.query_embedding(queries)
        F = self.feature_transform(features)
        spacial_feature_input = self.feature_transform1(features)
        F_new = tf.reshape(F, [F.shape[0], F.shape[1]*F.shape[2], F.shape[3]])
        reshaped_attention_map = tf.nn.softmax(tf.matmul(F_new, Q) / tf.sqrt(tf.cast(self.num_attributes, tf.float32)), axis=-1)
        attention_map = tf.reshape(reshaped_attention_map, [F.shape[0], F.shape[1], F.shape[2], Q.shape[2]])
        reshaped_spacial_feature_input = tf.reshape(spacial_feature_input, [F.shape[0], -1, spacial_feature_input.shape[3]])
        reshaped_spacial_feature_input = tf.transpose(reshaped_spacial_feature_input, perm = [0, 2, 1])
        return attention_map, tf.matmul(reshaped_spacial_feature_input, reshaped_attention_map)


class GAMModule(tf.keras.layers.Layer):
    def __init__(self, num_groups, momentum, group_indices_list):
        super(GAMModule, self).__init__()
        self.num_groups = num_groups
        self.group_indices_list = group_indices_list
        self.momentum = momentum
        self.group_memory = [tf.Variable(tf.zeros(shape=(1, 7, 7)), trainable=False) 
                             for _ in range(num_groups)]

    def call(self, query_attention_maps):
        query_attention_maps = query_attention_maps[0]
        group_attentions = []

        for group_idx, indices in enumerate(self.group_indices_list):
            aggregated_attention = tf.add_n([query_attention_maps[i] for i in indices]) / len(indices)
            self.group_memory[group_idx].assign(
                (1 - self.momentum) * self.group_memory[group_idx] +
                self.momentum * tf.reduce_mean(aggregated_attention, axis=0, keepdims=True)
            )
            group_attentions.append(self.group_memory[group_idx])

        return group_attentions


class DAFLModel(tf.keras.Model):
    def __init__(self, num_attributes, num_groups, momentum, group_indices_list, num_cascades):
        super(DAFLModel, self).__init__()
        self.num_attributes = num_attributes
        self.resnet = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
        self.ssca_modules = [SSCAModule(num_attributes) for _ in range(num_cascades)]
        self.gam_module = GAMModule(num_groups, momentum, group_indices_list)
        self.classifiers = [layers.Dense(1, activation='sigmoid') for _ in range(num_attributes)]

    def call(self, inputs, num_attributes):
        feature_map = self.resnet(inputs)
        
        queries = tf.random.normal(shape=(inputs.shape[0], tf.shape(feature_map)[-1], num_attributes))
        attention_maps = []
        for ssca in self.ssca_modules:
            attention_map, queries = ssca(queries, feature_map)
            attention_map = tf.transpose(attention_map, perm = [0, 3, 1, 2])
            attention_maps.append(attention_map)
        group_attentions = self.gam_module(attention_map)
        outputs = []
        for i in range(self.num_attributes):
            # Process each attribute's features through its classifier
            attribute_features = queries[:, :, i]  # Shape: [batch_size, input_dim]
            attribute_output = self.classifiers[i](attribute_features)  # Shape: [batch_size, 1]
            outputs.append(attribute_output)

        classification_output = tf.concat(outputs, axis=-1)
        return classification_output, queries, group_attentions, attention_map
    
    
import tensorflow as tf
import numpy as np


def compute_triplet_loss(embeddings, labels, num_attributes, y_pred, margin=1.0):
    triplet_loss = 0

    for attr_idx in range(num_attributes):
        attribute_embeddings = embeddings[:, :, attr_idx]
        attribute_labels = labels[:, attr_idx]
        y_pred_attr = y_pred[:, attr_idx]
        positive_mask = labels[:, attr_idx] == 1
        negative_mask = labels[:, attr_idx] == 0

        positive_embeddings = tf.boolean_mask(attribute_embeddings, positive_mask)
        negative_embeddings = tf.boolean_mask(attribute_embeddings, negative_mask)

        if tf.size(positive_embeddings) == 0 or tf.size(negative_embeddings) == 0:
            continue 
        lowest_probability_index = np.argmin(y_pred_attr[positive_mask])
        highest_probability_index = np.argmax(y_pred_attr[negative_mask])
        
        hardest_positive_feature = positive_embeddings[lowest_probability_index]
        hardest_negative_feature = negative_embeddings[highest_probability_index]
        
        pos_loss = 0
        neg_loss = 0
        
        for i in range(len(positive_embeddings)):
            pos_loss += max(0, np.linalg.norm(positive_embeddings[i] - hardest_positive_feature) - np.linalg.norm(positive_embeddings[i] - hardest_negative_feature))
            
        for i in range(len(negative_embeddings)):
            neg_loss += max(0, np.linalg.norm(negative_embeddings[i] - hardest_negative_feature) - np.linalg.norm(negative_embeddings[i] - hardest_positive_feature))

        triplet_loss += (pos_loss + neg_loss) / embeddings.shape[0]

    triplet_loss /= tf.cast(num_attributes, tf.float32)

    return triplet_loss


def classification_loss(y_true, y_pred):
    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    return bce_loss(y_true, y_pred)


def compute_group_consistency_loss(group_attentions, query_attention_maps, group_indices_list):
    loss = 0.0
    query_attention_maps = query_attention_maps[0]
    for group_idx, indices in enumerate(group_indices_list):
        group_memory = group_attentions[group_idx]
        for attr_idx in indices:
            # Calculate MSE between the group memory and each attribute's attention map
            loss += tf.reduce_mean(tf.square(group_memory - query_attention_maps[attr_idx]))

    # Normalize the loss by the number of attributes
    loss /= sum(len(indices) for indices in group_indices_list)
    return loss

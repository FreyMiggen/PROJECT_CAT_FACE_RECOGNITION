import tensorflow as tf
import os
import numpy as np


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.linalg.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.cast(tf.equal(distances, 0.0),dtype=tf.float32)
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)
    
    # REGULARIZATION: ADD THE DOT PRODUCT TERM
    return distances, dot_product



def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask


def batch_all_triplet_loss(labels, embeddings, margin=0.2,alpha=0, squared=False):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist,dot_product = _pairwise_distances(embeddings, squared=squared)
    
    #REGULARIZATION: compute regulazation term so  that dot_pro_expand[i,j,k] contains embed_i(T)*emb_k
    dot_pro_expand=tf.reshape(tf.tile(dot_product,(1,labels.shape[0])),[labels.shape[0],labels.shape[0],labels.shape[0]])

    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    mask = tf.cast(mask,dtype=tf.float32)
    triplet_loss = tf.multiply(mask, triplet_loss)
    
    #REGULARIZATION:put to zero to invalid triplets
    regularizer=tf.multiply(mask,dot_pro_expand)
    M1=tf.math.square(regularizer)
    M2=tf.maximum(M1-(1/32),0)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)
    #REGULARIZATION: add regularization term
    triplet_loss=triplet_loss+alpha*(M1+M2)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16),dtype=tf.float32)
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss


def accuracy_triplet(labels,embeddings,margin=0.2,squared=False):
    # Get the pairwise distance matrix
    pairwise_dist,dot_product = _pairwise_distances(embeddings, squared=squared)
    
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    mask = tf.cast(mask,dtype=tf.float32)
    triplet_loss = tf.multiply(mask, triplet_loss)
    #triplet_loss = tf.maximum(triplet_loss, 0.0)
    
    #TRIPLET WITH LOSS<0
    
    negative_triplets=tf.cast(tf.less(triplet_loss,0),tf.float32)
    num_negative_triplets=tf.math.reduce_sum(negative_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    return num_negative_triplets/num_valid_triplets

def veri_accuracy(labels,embeddings,lower_thres=0.5,upper_thres=1.0,squared=False):
    """
    Return a 2D mask where mask[a, b] is True if the (a, b) is of same label and false if (a,b) is of different label.
    """
    pos_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    neg_equal=tf.cast(tf.logical_not(pos_equal),dtype=tf.float32)
    pos_equal=tf.cast(pos_equal,dtype=tf.float32)
    
    pairwise_dist,dot_product = _pairwise_distances(embeddings, squared=squared)
    
    pos_dist=tf.math.multiply(pos_equal,pairwise_dist)
    neg_dist=tf.math.multiply(neg_equal,pairwise_dist)
    
    #filter out neg_dist which is set to 0
  
    checked_pos=tf.cast(tf.less(pos_dist,lower_thres),dtype=tf.float32)
    filtered_checked_pos=tf.math.reduce_sum(checked_pos)-tf.math.reduce_sum(neg_equal)
    
    checked_neg=tf.cast(tf.greater(neg_dist,upper_thres),dtype=tf.float32)
    checked_neg=tf.math.reduce_sum(checked_neg)
    return filtered_checked_pos,checked_neg,tf.math.reduce_sum(pos_equal)

################################################
def process_img(img,label):
    img=tf.keras.applications.resnet_v2.preprocess_input(img)
    img=tf.image.random_brightness(img,0.2)
    img=tf.image.random_saturation(img,2,5)
    img=tf.image.random_flip_left_right(img)
    return img,label

def parse_img_path(filepath):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img,channels=3)
    #img = tf.image.convert_image_dtype(img, tf.float32)
    img=tf.image.resize(img,(224,224))
    img=tf.keras.applications.resnet_v2.preprocess_input(img) 
    return img

def process_img_batch(filepaths,model):
    testset=tf.data.Dataset.from_tensor_slices(filepaths)
    testset=testset.map(parse_img_path)
    testset=testset.batch(len(filepaths)) 
    emb=model.predict(testset)
    return emb


def add_face_to_db(img_dir,db_path,model=None,model_path='model/model_90.h5',add_to_img_db=False,img_db_path=None):
    
    """
    Usage: Given a folder containing at least 5 images of a cat face, calculate 
    the 3 best embedding vectors and store them in the embedding db; optionally
    store original images in image db
    Notice: file in vector db and sub folder in image db have the same name and is
           also the name of img_dir
    Args:
        img_dir: path of the folder containing images
        db_path: path to the folder that saves embedding vectors, 
                   defaul is 'run/{count}/img_db'
    """
    if model is None:
        model=tf.keras.models.load_model(model_path)
        
    img_dir=os.path.normpath(img_dir)
    db_path=os.path.normpath(db_path)
    filepaths=[os.path.join(img_dir,filename) for filename in os.listdir(img_dir)]
    cat_emb=process_img_batch(filepaths,model)
    
    dist_cat= _pairwise_distances(cat_emb, squared=False)[0]
    mean_emb=np.mean(dist_cat,axis=0)
    ind=np.argsort(mean_emb)

    data= [cat_emb[ind[0]],cat_emb[ind[1]],cat_emb[ind[2]]]
    data=np.asarray(data)
    
    #savename: run/count/emb_db/name_of_the_sub_dir.csv
    
    saved_name=os.path.join(db_path,img_dir.split('\\')[-1])
    np.savetxt(f'{saved_name}.csv',data,delimiter=',')
    
    if add_to_img_db==True:
        if img_db_path is None:
            saved_sub_fol=os.path.join('0/0',img_dir.split('\\')[-1])
        else:
            saved_sub_fol=os.path.join(img_db_path,img_dir.split('\\')[-1])
        for fname in filepaths:
            shutil.copy(fname,saved_sub_fol)
        
        
    
    
# def add_face_to_db(sub_dir,model=None,model_path='model/model_90.h5'):
#     if model is None:
#         model=tf.keras.models.load_model(model_path)
        
#     img_names=os.listdir(sub_dir)
#     img_paths=[os.path.join(sub_dir,fname) for fname in img_names]
    
#     select_anchor_vects(sub_dir
    # embeddings=process_img_batch(img_paths,model)
    
    
    
    
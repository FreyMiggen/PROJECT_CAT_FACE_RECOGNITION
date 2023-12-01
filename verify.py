import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
#import os
import argparse
import random

def processed_for_test(image_path,model):
    img = tf.keras.utils.load_img(image_path)
    img = tf.keras.utils.img_to_array(img)
    img=tf.image.resize(img,(224,224))
    
    input_arr=tf.keras.applications.resnet_v2.preprocess_input(img)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    return img/255.0,predictions

def test_result(filename,result):
    #result is an embedding vector of shape (1,32)
    # return False if 2 out of 3 vector return result>0.5
    anchors=np.loadtxt(filename,delimiter=',')
    dist=anchors-result
    dist=np.square(dist)
    dist=np.sum(dist,axis=1)
    dist=np.sqrt(dist)
    dist_05=np.less(dist,0.5)
    dist_06=np.less(dist,0.6)
    if np.sum(dist_06)>=2 or np.sum(dist_05)>=1:
        return True
    else:
        return False
    
def most_similar(filename,result):
    anchors=np.loadtxt(filename,delimiter=',')
    dist=anchors-result
    dist=np.square(dist)
    dist=np.sum(dist,axis=1)
    dist=np.sqrt(dist)
    return np.sum(dist)

def verify(img_path,model_version='03_60',model_path=None,db_fol=None):
    
    """
    Usage: Use model to verify whether a specific cat is stored in the db
    
    Args:
    model_path: path to the model that you want to use to verify. If you choose to select this option,
                then you must provide the path to the embedding vector db that corresponds to the model
                
    db_fol: path to the db that stores embedding vectors (user only need to specifies it when model_path is not None)
    
    model_version: if you do not have any model, specify the model version to use
                   one of the default models. 
    
    Returns:
    existed: boolean value whether the cat given is stored in the embedding db or not
    chosen: the file path to the image db that is decide my the model as the cat/the most similar cat
            stored in the image db. 
    """
    # print(model_version)
    if model_path is None:
        if model_version in ['05_60','00_60','03_60','03_90']:
            model_path=f'model/model_{model_version}.h5'
            model=tf.keras.models.load_model(model_path)
            filedirs=[os.path.join(f'emb_db/{model_version}',filename) for filename in os.listdir(f'emb_db/{model_version}')]
        else:
            print('Please specify a model version or to provide the path to the model')
            return None
    else:
        if db_fol is None:
            print('Please provide the path to the embedding vector folder')
            return None
        else:
            model_path=os.path.normpath(model_path)
            model=tf.keras.models.load_model(model_path)
            filedirs=[os.path.join(db_fol,fname) for fname in os.listdir(dp_fol)]

    img,test_emb=processed_for_test(img_path,model)
    min_dist=10
    chosen=None
    existed=False

    for file in filedirs:
        if test_result(file,test_emb)==1:
            existed=True
            chosen=file

    if not existed:        
        for file in filedirs:
            if most_similar(file,test_emb)<=min_dist:
                min_dist=most_similar(file,test_emb)
                chosen=file
    # from the file in the vector db, file the corresponding sub fol in the image db
    # In order for this to perform properly, they have to have the same name
    chosen=os.path.normpath(chosen)
    fname=chosen.split('\\')[-1].split('.')[0]
    chosen=os.path.join('img_db',fname)     
    
    return existed,chosen

def Main():
    parser=argparse.ArgumentParser()
    parser.add_argument('img_path',help='Add the image path of the cat image you want to test')
    parser.add_argument('-m','--model_version',help='If you dont want to use the default model, declare the model version you want to use \n option: 00_40, 03_40,05_40, 00_60, 03_60,05_60 \n syntax: alpha_epoch',
    default='03_60')
    
    args=parser.parse_args()
    existed,chosen_fol=verify(args.img_path,args.model_version)
    if existed:
  
        print(f'The cat is already stored in the database in {chosen_fol}')
        img_list=[os.path.join(chosen_fol,fname) for fname in os.listdir(chosen_fol)]
        size=len(img_list)-1
        user_ans=input('Would you like to see another image of the cat?[y,n]: ')
        if user_ans=='y':
            ind=random.randint(0,size)
            img=Image.open(img_list[ind])
            img.show()
    else:
        print('The cat is not yet recorded in our database!')
        user_ans=input('Would you like to see image of the cat that is decided as most similar by our model?[y,n]: ')
        if user_ans=='y':
            img_list=[os.path.join(chosen_fol,fname) for fname in os.listdir(chosen_fol)]
            size=len(img_list)-1
            ind=random.randint(0,size)
            img=Image.open(img_list[ind])
            img.show()


if __name__=='__main__':
    Main()




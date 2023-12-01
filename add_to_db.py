import utils 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from PIL import Image
import numpy as np
#import os
import argparse
import shutil


def add_to_db(img_dir,model_path='model/model_03_60.h5',db_path='emb_db/03_60',add_to_img_db=True,img_db_path=None):
    
    """
    Usage: Given a folder containing at least 5 images of a cat face, calculate 
    the 3 best embedding vectors and store them in the embedding db; optionally
    store original images in image db
    Notice: file in vector db and sub folder in image db have the same name and is
           also the name of img_dir
    Args:
        img_dir: path of the folder containing images
        db_path: path to the folder that saves embedding vectors, 
                  
    """
 
    model=tf.keras.models.load_model(model_path)
    img_dir=os.path.normpath(img_dir)
    db_path=os.path.normpath(db_path)
    filepaths=[os.path.join(img_dir,filename) for filename in os.listdir(img_dir)]
    cat_emb=utils.process_img_batch(filepaths,model)
    
    dist_cat= utils._pairwise_distances(cat_emb, squared=False)[0]
    mean_emb=np.mean(dist_cat,axis=0)
    ind=np.argsort(mean_emb)

    data= [cat_emb[ind[0]],cat_emb[ind[1]],cat_emb[ind[2]]]
    data=np.asarray(data)
    
    cat_name=img_dir.split('\\')[-1]
    saved_name=os.path.join(db_path,cat_name)
    np.savetxt(f'{saved_name}.csv',data,delimiter=',')
    
    print(f'Embedding vectors of {cat_name} is already saved in {saved_name}.csv !')
    
    if add_to_img_db==True:
        if img_db_path is None:
            saved_sub_fol=os.path.join('img_db',cat_name)
        else:
            saved_sub_fol=os.path.join(img_db_path,cat_name)
        shutil.copytree(img_dir,saved_sub_fol)
        print('Original images of {} is already stored in {}'.format(cat_name,saved_sub_fol))




def Main():
    parser=argparse.ArgumentParser()
    parser.add_argument('img_dir',help='Add the path of the folder containing images of cat face that you want to add to db')
    parser.add_argument('-m','--model_path',help='Add the path to the model you want to use',default='model/model_03_60.h5')
    parser.add_argument('-db','--db_path',help='Add the path to the folder that saves embedding vectors corresponding to the model in used',default='emb_db/03_60')
    parser.add_argument('-a','--add_to_img_db',help='Boolean value: whether you want to save original images to image db or nor?', default=True)
    parser.add_argument('-i','--img_db_path')
    args=parser.parse_args()
    add_to_db(args.img_dir,args.model_path,args.db_path,args.add_to_img_db,args.img_db_path)
    
    
if __name__=='__main__':
    Main()
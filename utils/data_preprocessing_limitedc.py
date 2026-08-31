import torch
import os 
import json
import numpy as np
from re import L
import matplotlib.pyplot as plt
import random 
from PIL import Image
import torchvision
from torchvision import transforms
from torchvision.models._utils import IntermediateLayerGetter
from tqdm import tqdm

def hyper_parameters(sampling_rate, frame_img, stride_img, frame_k, stride_k):
    """
    we will now set the hyper-parameters for processing data.

    sampling_rate: let sampling_rate=N, we sample each Nth img in a img list(surrounding_pathlist)
    frame_img: sliding window length for s_imgs
    stride_img: stride of the sliding windows applied on only s_imgs
    frame_k: sliding window length for raw_keypoints and raw_trajectory
    stride_k: stride of the sliding windows applied on both raw_keypoints and raw_trajectory
    pad_num: zeros padding length for three-type data  
    """
    hypers = {'sampling_rate': sampling_rate, 'frame_img':frame_img, 'stride_img':stride_img, 'frame_k':frame_k, 'stride_k':stride_k}
    return hypers

def get_14keypoints(a_sample):
    head = (np.array(a_sample['nose'])+np.array(a_sample['left_eye'])+np.array(a_sample['right_eye']))/3
    thorax = (np.array(a_sample['left_shoulder'])+np.array(a_sample['right_shoulder']))/2
    leftshoulder = np.array(a_sample['left_shoulder'])
    rightshoulder = np.array(a_sample['right_shoulder'])
    leftelbow = np.array(a_sample['left_elbow'])
    rightelbow = np.array(a_sample['right_elbow'])
    leftwrist = np.array(a_sample['left_wrist'])
    rightwrist = np.array(a_sample['right_wrist'])
    lefthip = np.array(a_sample['left_hip'])
    righthip = np.array(a_sample['right_hip'])
    leftknee = np.array(a_sample['left_knee'])
    rightknee = np.array(a_sample['right_knee'])
    leftankle = np.array(a_sample['left_ankle'])
    rightankle = np.array(a_sample['right_ankle'])

    preprocessed_keypoints = np.array([head,thorax, leftshoulder, rightshoulder, leftelbow, rightelbow, leftwrist, rightwrist, lefthip, righthip, leftknee, rightknee, leftankle, rightankle])
    preprocessed_keypoints = np.swapaxes(preprocessed_keypoints, 1, 0)    
    return preprocessed_keypoints

    

def sliding_window(x, n, s): #n: width of window, s=stride
    window_start = 0
    window_end = n-1
    samples = []
    while window_end <= x.shape[0]-1:
        sample = x[window_start:window_end+1] #numpy array
        samples.append(sample) #list
        window_start+=s
        window_end+=s
            
    samples = np.stack(samples, axis=0)

    return samples


def padding(pad, data): 
    if len(data.shape) == 2: # trajectory: frame, 5; categorical frame, 4 
        pad_zeros = np.zeros((pad, data.shape[1]))
    elif len(data.shape) == 3: #preproccessed keypoints shape: frame, 14, 2
        pad_zeros = np.zeros((pad, data.shape[1], data.shape[2]))
    else:
        print('error: data shape is not valid')
        raise SystemExit()
    
    pad_data = [data, pad_zeros]
    pad_data = np.concatenate(pad_data, axis=0)
    pad_data = np.expand_dims(pad_data, axis=0) 
    return pad_data



def get_observed_frames(img_list):
    observed_frames = []
    for path in img_list:
        img = path.split('/')[-1]
        frame = int(img.split('_')[3])
        observed_frames.append(frame)
    return observed_frames

def get_sample_last_frames(observed_frames, n, s):
    window_start_idx = 0
    window_end_idx = n-1
    last_frame_indices = []
    while window_end_idx <= len(observed_frames)-1:
        last_frame_indices.append(window_end_idx)
        window_start_idx+=s
        window_end_idx+=s
        
    sample_last_frames = np.array(observed_frames)[last_frame_indices].tolist()
    return sample_last_frames

def get_TTE(observed_frames, sample_last_frames):
    crossing_moment = observed_frames[-1]+1
    TTE_list = np.array(crossing_moment) - sample_last_frames
    return TTE_list
    
                               
def concatenate_both_cases(data_placeholder):
    """
    we will now read three-type data from from data_placeholder 
    """
    data_pos_k, data_neg_k = data_placeholder['pos']['k'], data_placeholder['neg']['k']
    data_pos_t, data_neg_t = data_placeholder['pos']['t'], data_placeholder['neg']['t']
    data_pos_c, data_neg_c = data_placeholder['pos']['c'], data_placeholder['neg']['c']
    data_pos_g, data_neg_g = data_placeholder['pos']['g'], data_placeholder['neg']['g']

    
    """
    we will now concatenate each data type list to a numpy matrix
    """
    data_pos_k = np.concatenate(data_pos_k, axis=0)
    data_neg_k = np.concatenate(data_neg_k, axis=0)
    data_pos_t = np.concatenate(data_pos_t, axis=0)
    data_neg_t = np.concatenate(data_neg_t, axis=0)
    data_pos_c = np.concatenate(data_pos_c, axis=0)
    data_neg_c = np.concatenate(data_neg_c, axis=0)      
    data_pos_g = np.concatenate(data_pos_g, axis=0)
    data_neg_g = np.concatenate(data_neg_g, axis=0)
    
    print('pos sample len: {0}'.format(data_pos_k.shape[0]))
    print('neg sample len: {0}'.format(data_neg_k.shape[0]))


    """
    we will now calculate a maximum value for both x1,x2 in keypoints data (**_k). 
    """
    max_x1_pos = torch.max(torch.tensor(data_pos_k[:,:,:,0]))
    max_x2_pos = torch.max(torch.tensor(data_pos_k[:,:,:,1]))
    max_x1_neg = torch.max(torch.tensor(data_neg_k[:,:,:,0]))
    max_x2_neg = torch.max(torch.tensor(data_neg_k[:,:,:,1]))


    x1_max = max(max_x1_pos, max_x1_neg)
    x2_max = max(max_x2_neg, max_x2_pos)


    """
    we will now normalize both train and test set for keypoint data. The first dimension will be normalized by x1_max. The second dimension will be normalized by x2_max
    """
    dims = [0, 1] 
    for dim in dims:
        if dim == 0:
            (data_pos_k[:,:,:,dim]) = (data_pos_k[:,:,:,dim]) /x1_max
            (data_neg_k[:,:,:,dim]) = (data_neg_k[:,:,:,dim]) /x1_max

        else:
            (data_pos_k[:,:,:,dim])= (data_pos_k[:,:,:,dim]) /x2_max
            (data_neg_k[:,:,:,dim])= (data_neg_k[:,:,:,dim]) /x2_max

    """
    we will now prepare labels for three-type data based on number of postive and negative cases.
    """
    def generate_y(data_pos_shape, data_neg_shape):
        labels_pos = np.ones((data_pos_shape[0],1))
        labels_neg = np.zeros((data_neg_shape[0],1))

        labels = np.concatenate([labels_pos, labels_neg], axis=0)
        
        return labels
    
    #keypoints    
    k_labels = generate_y(data_pos_k.shape, data_neg_k.shape)

#     #trajactory
#     t_labels = generate_y(data_pos_t.shape, data_neg_t.shape)

#     #categorical
#     c_labels = generate_y(data_pos_c.shape, data_neg_c.shape)

    """
    we merge postive and negative in three-type data. Then, we put three-type data in lists (X_train, Y_train, X_test, Y_test).
    """
    k = np.concatenate([data_pos_k, data_neg_k], axis=0)
    t = np.concatenate([data_pos_t, data_neg_t], axis=0)
    c = np.concatenate([data_pos_c, data_neg_c], axis=0)
    g = np.concatenate([data_pos_g, data_neg_g], axis=0)

    X_data = [k,t,c,g]
    Y_data = [k_labels]

    return X_data, Y_data
        
        
def placeholder_k_t_c(cases):
    #this function creates a dictionary to store preprocessed three-type data
    num_cases = len(cases)
    if num_cases:
        if num_cases==2:
            data_placeholder = {'pos':{'k':[], 't':[], 'c':[], 'TTE':[]}, 'neg':{'k':[], 't':[], 'c':[]}}
        else:
            print("Only support 2 classes!")
            raise SystemExit()
        return data_placeholder #at least one case 
    else:
        print("Number of classes not found!")
        raise SystemExit()

def placeholder_k_t_c_g(cases):
    #this function creates a dictionary to store preprocessed three-type data
    num_cases = len(cases)
    if num_cases:
        if num_cases==2:
            data_placeholder = {'pos':{'k':[], 't':[], 'c':[], 'g':[],'TTE':[]}, 'neg':{'k':[], 't':[], 'c':[], 'g':[]}}
        else:
            print("Only support 2 classes!")
            raise SystemExit()
        return data_placeholder #at least one case 
    else:
        print("Number of classes not found!")
        raise SystemExit()

def get_x_y_z_list(raw_trajectory):
    x_list = list(raw_trajectory[:,0])
    y_list = list(raw_trajectory[:,1])
    z_list = list(raw_trajectory[:,2])
    return x_list, y_list, z_list

def change_of_location(x_list, y_list):
    speed_list = []
    for i, x in enumerate(x_list):
        if i>0:
            x_change_square= (x_list[i]-x_list[i-1])**2
            y_change_square= (y_list[i]-y_list[i-1])**2
            distance = (x_change_square+y_change_square)**(1/2)
            speed = distance/1 #one frame
            speed_list.append(speed)
        else:
            speed_list.append(0)#the very first frame of pedestrian track does not have its previous-frame location information, so we pad the speed of first frame with 0
    return speed_list


def change_of_size(z_list):
    size_change_list = []
    for i, size in enumerate(z_list):
        if i>0:
            size_change = z_list[i]-z_list[i-1]
            size_change_list.append(size_change)
        else:
            size_change_list.append(0)#pad the size change of first frame with 0
    return size_change_list

def read_split_ids(split_ids_root):
    
    def _txt2list(split_ids_root, f):        
        with open(split_ids_root+f) as f:
            li = f.read().splitlines()
        return li
    
    train_ids = _txt2list(split_ids_root, 'train.txt')
    val_ids = _txt2list(split_ids_root, 'val.txt')
    test_ids = _txt2list(split_ids_root, 'test.txt')
    return train_ids, val_ids, test_ids

def get_data_add_categorical(hypers, cases, raw_data, save_np_folder, subset = 'train', split_ids_root='./split_ids/'):    
    """
    The get_data takes the following arguments as inputs:

    (1) hypers: a dictionary of values of hyper-parameters
    (2) cases: a list of cases, e.g, ['pos', 'neg'] or ['pos'] or ['neg']
    (3) raw_data: a dictionary stores unprocessed keypoints, trajectory and full surrounding image path before sampling
    (4) save_np_folder: a path that saves 5-fold numpy data for each of three-type and synchronization dictionaries that link datapoints to pedestrian IDs
    (5) restrict_jaadnobeh_dict: if True, we will get data generated from bbox whose size is >= lowerbounds.
    (6) add_trajectory_features: if True, we will add speed and the change bbox size along with bbox coordinates
    (7) subset: define which subset we want to produce, 'train' or 'test'
    we will now extract values of hyper-parameters. 
    """
    frame_k = hypers['frame_k']
    stride_k = hypers['stride_k']

    data_placeholder = placeholder_k_t_c_g(cases)#  we put pre-processed (keypoints,trajectory,surrounding imags) data in a dictionary called "data_placeholder". The dict will be updated everytime a new ID is given. Initually, it is empty and with no information.  
    sync = {'pos':{}, 'neg':{}}#The synchronization dictionary saves pedestrian IDs and the corresponding number of datapoints for each of three-type data. 
    
    """
    we will now load raw data.
    
    Raw data dict structure:
        data = {
                ‘pos’:
                      {ID_1:{‘keypoints’:{}, ‘imgs’:[], ‘trajectory’:[]}, 
                       ID_2…}, 
                ‘neg’:…
               }

    """
    raw_data = open(raw_data)
    data = json.load(raw_data) #load raw data from all_labels_beh.json, all_labels_nobeh.json or jaadall_labels.json
    
        

        
    """
    Create a folder to save numpy arrays (preprocessed data) 
    """
        
    cwd = os.getcwd()                            
    folder = cwd+'/preprocessed_data/'+save_np_folder
    
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    if subset == 'train': 
        folder = folder + 'train/'
    elif subset == 'test': 
        folder = folder + 'test/'
    else:
        print('Not support validation set nor other names!')
        raise SystemExit()
        
    
    if not os.path.exists(folder):
        os.mkdir(folder)    
    
    
    """
    start to process data from each pedID
    """
    
    for case in cases:  # we will process k, t, s data for each ID in either positive or negative case. The two cases are processed in the same way but we keep track of IDs and cases for later.
        IDs = list(data[case].keys())
        print('Processing {0} cases'.format(case))
        train_ids, val_ids, test_ids = read_split_ids(split_ids_root)
        train_pool = []
        train_pool.extend(train_ids)
        train_pool.extend(val_ids)
        
        for i in tqdm(range(len(IDs))): # iterate over the set of IDs for the case we've selected
            ID = IDs[i]

            """
            check which subset we're producing, train or test
            """            
            a = int(ID.split('_')[1])
            vid = 'video_'+ f"{a:04}"
            if subset == 'train': #we merge train and val in the single train set 
                if vid not in train_pool:
                    continue
            else:#subset == 'test'
                if vid not in test_ids:
                    continue
            
            if ID not in sync[case].keys():
                sync[case][ID] = 0
            
            
            """
            load data
            """
            total_frame = len(data[case][ID]['keypoints']['nose'])#Get total number of frames for an ID
            if total_frame==0: # if any of these conditions happen, jump to the next ID
                continue
            img_list = data[case][ID]['imgs']
            observed_frames = get_observed_frames(img_list)

            """
            we will now extract three-type data from the raw dict. 

            The keypoints and trajectory do not need further processing but surrounding images will pass a sampling process and compressed by vgg model in the function called 'img_sampling_and_compression'.
            """
            raw_keypoints = get_14keypoints(data[case][ID]['keypoints'])
            raw_trajectory = np.array(data[case][ID]['trajectory'])
            
            """
            now add additional categorical data 
            """
#             raw_reaction = np.array(data[case][ID]['reaction'])
#             raw_look = np.array(data[case][ID]['look'])
#             raw_action = np.array(data[case][ID]['action'])
            
#             raw_vehspeed = np.array(data[case][ID]['veh_speed'])
            raw_light = np.array(data[case][ID]['tra_light'])
            raw_vehspeed = np.array(data[case][ID]['veh_speed'])
            
            try:
                raw_inter = np.array(data[case][ID]['intersection'])
            except:
                raw_inter=2#intersection=unknown
                
            raw_roadtype = np.array(data[case][ID]['road_type'])
            
            
            
#             print(raw_vehspeed.shape, raw_light.shape, raw_look.shape, raw_gesture.shape)
#             raw_categorical = np.stack((raw_look, raw_action, raw_light, raw_reaction, raw_vehspeed),axis=1)
            raw_categorical = np.stack((raw_light,raw_vehspeed),axis=1)
            raw_global_env = [[raw_roadtype, raw_inter]]#to multiply samples based on preprocessed k len 
#             print(raw_categorical.shape)
            """
            add features for trajectory
            """
            x_list,y_list, z_list = get_x_y_z_list(raw_trajectory)
            raw_speed = np.array(change_of_location(x_list, y_list))
            raw_speed = np.expand_dims(raw_speed,axis=-1)
            raw_size_change = np.array(change_of_size(z_list))
            raw_size_change = np.expand_dims(raw_size_change,axis=-1)
#                 raw_trajectory = raw_trajectory[1:]# because raw_speed and raw_size_change have N-1 datapoint compared to N datapoint of raw_trajectory, we have to trash the first datapoint of raw_trajectory

            raw_trajectory = np.concatenate((raw_trajectory, raw_size_change, raw_speed),axis=-1)

            """
            We will now apply sliding_window/ padding function for three-type data to ensure each window has the fixed length specified in the hyperparameters. 

            The keypoints and trajectory will have sliding window length specified in hyper-parameter 'frame_k'

            """
        
            if len(raw_trajectory) == len(raw_categorical) == total_frame:
                if total_frame<frame_k:
                    pad_num = frame_k - total_frame
                    k = padding(pad_num, raw_keypoints)
                    t = padding(pad_num, raw_trajectory)
                    c = padding(pad_num, raw_categorical)
                    if case == 'pos':
                        sample_last_frames = np.array([observed_frames[-1]])
                    
                else:
                    k = sliding_window(raw_keypoints, frame_k, stride_k)
                    t = sliding_window(raw_trajectory, frame_k, stride_k)
                    c = sliding_window(raw_categorical, frame_k, stride_k)
                    if case == 'pos':
                        sample_last_frames = get_sample_last_frames(observed_frames, frame_k, stride_k)
                
                if case=='pos':
                    TTE_list = get_TTE(observed_frames, sample_last_frames)#observed_frames: frames indicated by img name, sample_last_frames: [the last frames of all samples, each sample has a last frame] 
                    if len(c)!=len(TTE_list):
                        raise SystemExit('{0} has inconsistent sample len {1} and TTE_list len {2}'.format(ID, len(c), len(TTE_list)))
                        

            else:
                print("all types should have the same length")
                print(len(raw_trajectory), len(raw_categorical), total_frame)
                raise SystemExit()                

            """
            we will now save updated three-type data to data_placeholder and update sync_(train/val) dictionaries.

            sync dictionary structure:

                sync = {"ID_1":{'keypoints_len':, 'surroundings_len':, 'trajectory_len':},
                        "ID_2":{...},
                        ...
                }

            """

            data_placeholder[case]['k'].append(k)
            data_placeholder[case]['t'].append(t)
            data_placeholder[case]['c'].append(c)
            data_placeholder[case]['g'].append(raw_global_env*len(k))
            
            if case=='pos':
                data_placeholder[case]['TTE'].extend(TTE_list.tolist())
            
            sync[case][ID]+=len(k)
                                  
    """
    we will now concatenate the data from both cases extracted from data_placeholder. 
    
    In addition, we also generate Y_train and Y_test by a function either "concatenate_both_cases" or "concatenate_single_case", depends on our raw data (all_labels_trajactory.json)  
    """
    X_data, Y_data = concatenate_both_cases(data_placeholder) 

    """
    we will now save three-type data separately in .npy format.  
    """
    x_k, x_t, x_c, x_g = X_data[0],X_data[1],X_data[2], X_data[3]
    y_k = Y_data[0]
    tte = data_placeholder['pos']['TTE']
    np.save(folder+'x_k.npy', x_k)
    np.save(folder+'x_t.npy', x_t)
    np.save(folder+'x_c.npy', x_c)
    np.save(folder+'x_g.npy', x_g)
    np.save(folder+'TTE.npy', np.array(tte))
    np.save(folder+'y.npy', y_k)
    
    """
    save sync
    """
    print('save ID2sample.json')
    save_json_name = 'ID2sample.json'
    a_file = open(folder+save_json_name, "w")
    json.dump(sync, a_file)
    a_file.close()
    

    print('Data shape: ')
    print(x_k.shape, x_t.shape, x_c.shape, x_g.shape, np.array(tte).shape)
    print('Label shape: ')
    print(y_k.shape)
    print('Data preparation done!')
    print('='*40)
    return data_placeholder

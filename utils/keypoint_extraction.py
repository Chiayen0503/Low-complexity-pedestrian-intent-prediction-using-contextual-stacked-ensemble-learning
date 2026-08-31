import sys
from mmpose.apis import MMPoseInferencer
from PIL import Image
import matplotlib.pyplot as plt

def find_center(bbox):
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    return (x1+x2)/2, (y1+y2)/2

def find_keypoints_center(keypoints):
    keypoints = np.array(keypoints)
    min_x, max_x=min(keypoints[:,0]), max(keypoints[:,0])
    min_y, max_y=min(keypoints[:,1]), max(keypoints[:,1])
    center = (min_x+max_x)/2, (min_y+max_y)/2
    return center

def distance(keypoints_center, shift_bbox_center):
    x1, y1 = keypoints_center[0], keypoints_center[1]
    x2, y2 = shift_bbox_center[0], shift_bbox_center[1]
    return (abs(x1-x2)**2+(abs(y1-y2)**2))**(1/2)

def predict_allkeypoints(inferencer, img_path):
    # creating a prediction generator when given input
    result_generator = inferencer(img_path, show=False)
    result = next(result_generator)
    result = result['predictions'][0]#a list of multiple keypoint dict
    people_keypoints = [dic['keypoints'] for dic in result]
    return people_keypoints

def find_true_keypoints(inferencer, img_path,df):
    people_keypoints = predict_allkeypoints(inferencer, img_path)
#     print(len(people_keypoints))
    dist_li = []
    
    f_img = img_path.split('/')[-1]
    example = df[f_img]
    """
    find the surrounding bbox and the center of the ped_bbox we're looking for, 
    only 1 surrounding bbox and 1 ped_bbox available
    """
    a, b = example['surounding_bbox'][0], example['surounding_bbox'][1]# x-> shift a, y->shift b
    bbox_center = find_center(example['ped_bbox'])#find raw pedbbox center
    shift_bbox_center = bbox_center[0]-a, bbox_center[1]-b
    
    if len(people_keypoints)>1:
        for person_keys in people_keypoints:
            keypoints_center = find_keypoints_center(person_keys)#find keypoint center
            dist = distance(keypoints_center, shift_bbox_center)
            print(dist)
            dist_li.append(dist)
                
        return shift_bbox_center, people_keypoints[np.argmin(dist_li)]
    
    elif len(people_keypoints)==1:
        return shift_bbox_center, people_keypoints[0]
        
    else:
        sys.exit('something wrong, should have at least a person keypoints')

def plot_keys_and_centre(img_path, real_keys, shift_bbox_center):
    f = img_path.split('/')[-1]
    img = np.asarray(Image.open(img_path))
    imgplot = plt.imshow(img)
    shift_x, shift_y = shift_bbox_center[0], shift_bbox_center[1]
    keys = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    for i, k in enumerate(real_keys):
        plt.plot(k[0],k[1], 'y*')
        if keys[i] == 'nose' or keys[i]== 'right_elbow':
            plt.text(k[0],k[1]+5, str(keys[i]) , fontsize = 10, color='r')

    plt.plot(shift_x, shift_y, 'r*')


def find_shift_bbox(example):
    a, b = example['surounding_bbox'][0], example['surounding_bbox'][1]# x-> shift a, y->shift b
    x1, y1, x2, y2 = example['ped_bbox']#find raw pedbbox center
    shift_bbox = x1-a, y1-b, x2-a, y2-b
    return shift_bbox

def stackback_keypoints(case, ID, labels_df, frame_idx):
    keys = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    keys_coords = []
    for k in keys:
        a_key = labels_df[case][ID]['keypoints'][k][frame_idx]
        keys_coords.append(a_key)
    return keys_coords


def count_legit_keypoints(df, labels_df):
    cases = ['pos', 'neg']
    correct_count = 0
    wrong_count = 0
    total_imgs = 0
    for case in cases:
        try:
            IDs = labels_df[case].keys()
            for ID in IDs:
                imgs = labels_df[case][ID]['imgs']
                for i, img in enumerate(imgs):
                    total_imgs+=1
                    f_img = img.split('/')[-1]
                    f_img = f_img.split('.')[0]
                    f_img = f_img[:-3]+'.png'
                    example = df[f_img]
                    x1, y1, x2, y2 = find_shift_bbox(example)#
                    keys_coords=stackback_keypoints(case, ID, labels_df, frame_idx=i)
                    key_center_x, key_center_y =  find_keypoints_center(keys_coords)
                    if x2>=key_center_x and x1<=key_center_x and y2>=key_center_y and y1<=key_center_y:
                        correct_count+=1
                    else:
                        wrong_count+=1
        except:
            continue

    print('correct: {0}'.format(correct_count))
    print('wrong: {0}'.format(wrong_count))
    print('total: {0}'.format(total_imgs))
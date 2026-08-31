from utils.data_loader import load_dict
import matplotlib.pyplot as plt
import numpy as np
import collections

def load_tte(root):
    raw_data = root +'ID2sample_update.json'
    
    df = load_dict(raw_data)
    tte_array = np.load(root+'TTE.npy')    
    tte_array = np.array([tte_array]*5)#since we'll validate 5-fold trained weight, tte_array is repeated 5 times, meaning the tte of positive samples is repeated 5 times.  
    tte_array = np.concatenate(tte_array)#tte
    return tte_array

def update_count(a, b, c, count):#count correct, wrong, acc with time t defined in tte_array
    """
    a: gts
    b: tte_array
    c: preds
    """
    for i, t in enumerate(b):#iterate TTE
        if t not in count.keys():
            count[t] = {'correct':0, 'wrong':0}
            
        if c[i]!=a[i]:#if pred!=gt
            count[t]['wrong']+=1
        else:
            count[t]['correct']+=1
    
    for j in count.keys():
        result = count[j]['correct']/(count[j]['correct']+count[j]['wrong'])
        result = float("{:.2f}".format(result))
        count[j]['acc'] = result
    
    return count

def get_x_y_li(count):
    od = collections.OrderedDict(sorted(count.items()))#sort count based on its keys, i.e, tte (in frames)
    x_li = list(od.keys())
    y_li = []
    for t in x_li:
        y_li.append(od[t]['acc'])
    return x_li, y_li, od

def plot_tte(count):
    """
    count: count correct and wrong based on tte (in frames)
    od: sorted dictionary of count
    x_li: time list
    y_li: acc list
    """
    x_li, y_li, od = get_x_y_li(count)
    print('TTE to accuracy: ')
    plt.plot(x_li, y_li)
    plt.xlabel('TTE in frames')
    plt.ylabel('Accuracy')
    plt.show()
    print('Avg: {0}'.format(sum(y_li)/len(x_li)))
    print('Sample count in TTE')
    dist = {}
    for k in od.keys():
        if k not in dist.keys():
            dist[k] = 0
        dist[k]+= od[k]['correct']+od[k]['wrong']
    plt.bar(dist.keys(), dist.values())
    plt.show()
    
    print('60 to 90th frame to accuracy:')
    plt.plot(x_li[60:91], y_li[60:91])#2sec before the event and sample for 1 sec
    plt.xlabel('TTE in frames')
    plt.ylabel('Accuracy')
    plt.show()
    print('Avg: {0}'.format(sum(y_li[60:91])/len(x_li[60:91])))
    
    print('='*40)


def get_time_results(x_li, y_li, time_li):
    for time in time_li:
        print(sum(y_li[time[0]:time[1]])/ len(y_li[time[0]:time[1]]))
    
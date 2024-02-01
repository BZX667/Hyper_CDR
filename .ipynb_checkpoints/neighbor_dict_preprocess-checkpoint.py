import json
import random
import pandas as pd

def load_neighbor_dict(order, dataset):
    file_path = '/home/featurize/work/Hyper_CDR/data/' + dataset + '/neighbor_{}_dict.json'.format(order)
    with open(file_path, 'r') as file:
        neighbor_indices_dict = json.load(file)
    return neighbor_indices_dict

if __name__ == "__main__":
    list = []
    dataset = "Amazon-CD"
    user_num = 29135
    overlap_size = 6591
    # dict1 = load_neighbor_dict(1, dataset)
    # dict2 = load_neighbor_dict(2, dataset)
    dict3 = load_neighbor_dict(3, dataset)
    
    for i in range(overlap_size):
        i = str(i)
        # n1 = dict1[i].split(" ")
        # n2 = dict2[i].split(", ")
        n3 = dict3[i].split(", ")

        # new_n2 = [x for x in n2 if x not in n1]
        # new_n3 = [int(x) for x in n3 if x not in n2]
        new_n3 = [int(x) for x in n3]
        new_n3 = [x for x in new_n3 if x < user_num]
        list.append(new_n3[random.randint(0, len(new_n3) - 1)])
    df = pd.DataFrame({'3nd_neighbor': list})
    df.to_csv('/home/featurize/work/Hyper_CDR/data/'+dataset+'/neighbor_3_b.csv')
        
#     with open('/home/featurize/work/Hyper_CDR/data/' + dataset + '/neighbor_2_dict_new.json', 'w') as file:
#         json.dump(dict2, file)
    
#     with open('/home/featurize/work/Hyper_CDR/data/' + dataset + '/neighbor_3_dict_new.json', 'w') as file:
#         json.dump(dict3, file)
    
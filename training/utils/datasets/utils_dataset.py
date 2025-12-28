import torch
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from .files_utils import*
import re

def get_ids_in_folder(base_path="./data/public/1.0.1/train/"):
    """
    Get a list of IDs from .npz files present in the specified directory.

    Parameters:
    - base_path (str): Path to the directory containing the .npz files.

    Returns:
    - list of ints: IDs extracted from filenames.
    """
    ids = []
    pattern = re.compile(r"^(\d{4})_d5000\.npz$")

    for filename in os.listdir(base_path):
        match = pattern.match(filename)
        if match:
            ids.append(int(match.group(1)))

    return sorted(ids)

def load_npz_by_id(id, base_path="./data/public/1.0.1/train/"):
    """
    Load an .npz file by its ID.

    Parameters:
    - id (int): The identifier number XXXX in the filename.
    - base_path (str): Path to the directory containing the .npz files.

    Returns:
    - dict-like object: The content of the loaded .npz file.
    """
    filename = f"{id:04d}_d5000.npz"
    filepath = os.path.join(base_path, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filepath}' not found.")

    data = np.load(filepath)
    return data

def compute_channel_mean_std(mode="train"):
    type_convert = np.float64

    sen1_sum = np.zeros(3, dtype=type_convert)
    sen1_sum_sq = np.zeros(3, dtype=type_convert)
    sen1_count = np.zeros(3, dtype=type_convert)

    sen2_sum = np.zeros(10, dtype=type_convert)
    sen2_sum_sq = np.zeros(10, dtype=type_convert)
    sen2_count = np.zeros(10, dtype=type_convert)

    l7_sum = np.zeros(6, dtype=type_convert)
    l7_sum_sq = np.zeros(6, dtype=type_convert)
    l7_count = np.zeros(6, dtype=type_convert)

    modis_sum = np.zeros(7, dtype=type_convert)
    modis_sum_sq = np.zeros(7, dtype=type_convert)
    modis_count = np.zeros(7, dtype=type_convert)

    alos_sum = np.zeros(3, dtype=type_convert)
    alos_sum_sq = np.zeros(3, dtype=type_convert)
    alos_count = np.zeros(3, dtype=type_convert)

    ids = get_ids_in_folder(base_path=f"./data/public/1.0.1/{mode}/")

    for id in tqdm(ids):
        data = load_npz_by_id(id, base_path=f"./data/public/1.0.1/{mode}/")

        s1, s1_mask = data["s1"], data["s1_mask"]
        s2, s2_mask = data["s2"], data["s2_mask"]
        l7, l7_mask = data["l7"], data["l7_mask"]
        modis, modis_mask = data["modis"], data["modis_mask"]
        alos, alos_mask = data["alos"], data["alos_mask"]

        for img_idx in range(s2.shape[0]):
            for c in range(s1.shape[-1]):
                mask = s1_mask[img_idx, :, :,:, c].astype(bool)
                valid_s1 = s1[img_idx, :, :,:, c][mask]
                sen1_sum[c] += valid_s1.sum()
                sen1_sum_sq[c] += (valid_s1 ** 2).sum()
                sen1_count[c] += mask.sum()

            for c in range(s2.shape[-1]):
                mask = s2_mask[img_idx, :, :,:, c].astype(bool)
                valid_s2 = s2[img_idx, :, :,:, c][mask]
                sen2_sum[c] += valid_s2.sum()
                sen2_sum_sq[c] += (valid_s2 ** 2).sum()
                sen2_count[c] += mask.sum()

            for c in range(l7.shape[-1]):
  
                mask = l7_mask[img_idx, :, :,:, c].astype(bool)
                valid_l7 = l7[img_idx, :, :,:, c][mask]
                l7_sum[c] += valid_l7.sum()
                l7_sum_sq[c] += (valid_l7 ** 2).sum()
                l7_count[c] += mask.sum()

            for c in range(modis.shape[-1]):
                mask = modis_mask[img_idx, :, :,:, c].astype(bool)
                valid_modis = modis[img_idx, :, :,:, c][mask]
                modis_sum[c] += valid_modis.sum()
                modis_sum_sq[c] += (valid_modis ** 2).sum()
                modis_count[c] += mask.sum()

            for c in range(2):
                mask = (alos_mask[img_idx, :, :,:, c] == 1) & (alos[img_idx, :,: ,:, c] != 0)
                valid_alos = alos[img_idx, :, :,:, c][mask]
                alos_in_db = 10 * np.log(valid_alos ** 2) - 83.0
                alos_sum[c] += alos_in_db.sum()
                alos_sum_sq[c] += (alos_in_db ** 2).sum()
                alos_count[c] += mask.sum()
            
            mask = (alos_mask[img_idx, :, :,:, 2] == 1) & (alos[img_idx, :, :,:, 2] != 0)
            valid_alos = alos[img_idx, :, :,:, 2][mask]
            alos_in_db = 10 * np.log(valid_alos ** 2) - 83.0
            alos_sum[2] += alos_in_db.sum()
            alos_sum_sq[2] += (alos_in_db ** 2).sum()
            alos_count[2] += mask.sum()

    s1_mean = sen1_sum / sen1_count
    s1_std = np.sqrt(np.maximum(sen1_sum_sq / sen1_count - s1_mean ** 2, 0))

    s2_mean = sen2_sum / sen2_count
    s2_std = np.sqrt(np.maximum(sen2_sum_sq / sen2_count - s2_mean ** 2, 0))

    l7_mean = l7_sum / l7_count
    l7_std = np.sqrt(np.maximum(l7_sum_sq / l7_count - l7_mean ** 2, 0))

    modis_mean = modis_sum / modis_count
    modis_std = np.sqrt(np.maximum(modis_sum_sq / modis_count - modis_mean ** 2, 0))

    alos_mean = alos_sum / alos_count
    alos_std = np.sqrt(np.maximum(alos_sum_sq / alos_count - alos_mean ** 2, 0))


    print("s1 :",s1_mean,s1_std)
    print("alos :",alos_mean,alos_std)
    print("s2 :",s2_mean,s2_std)

    torch.save(torch.from_numpy(s1_mean), "./data/normalisation/s1_mean.pt")
    torch.save(torch.from_numpy(s1_std), "./data/normalisation/s1_std.pt")
    torch.save(torch.from_numpy(s2_mean), "./data/normalisation/s2_mean.pt")
    torch.save(torch.from_numpy(s2_std), "./data/normalisation/s2_std.pt")
    torch.save(torch.from_numpy(l7_mean), "./data/normalisation/l7_mean.pt")
    torch.save(torch.from_numpy(l7_std), "./data/normalisation/l7_std.pt")
    torch.save(torch.from_numpy(modis_mean), "./data/normalisation/modis_mean.pt")
    torch.save(torch.from_numpy(modis_std), "./data/normalisation/modis_std.pt")
    torch.save(torch.from_numpy(alos_mean), "./data/normalisation/alos_mean.pt")
    torch.save(torch.from_numpy(alos_std), "./data/normalisation/alos_std.pt")


def plot_histo_dico(dico):
    # Assuming 'dico_labels' is your dictionary
    plt.figure(figsize=(20, 10))  # Make the figure wider
    plt.bar(dico.keys(), dico.values())
    plt.xlabel('Categories')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)  # Rotate labels if needed
    plt.tight_layout()       # Adjust layout to fit labels

    plt.savefig('histogram.pdf', format='pdf')  # Saves the figure as a PDF file
    plt.close()  # Close the plot to avoid displaying it

def get_labels(ds,id):

    row=ds.iloc[id]

    # Access the "labels" column
    labels = row['labels']  # Replace 'labels' with the actual column name if different

    return labels

def get_split(ds,id):

    row=ds.iloc[id]

    # Access the "labels" column
    split = row['split']  # Replace 'labels' with the actual column name if different

    return split

def get_one_hot_indices(tensor):
    """
    Returns a list of indices of labels encoded in a one-hot encoded tensor.
    
    Args:
        tensor (torch.Tensor): A 2D one-hot encoded tensor (batch_size x num_classes)
        
    Returns:
        List[int]: Indices of the labels for each row.
    """
    # Use argmax to find the indices of the maximum value (1) in each row
    indices = torch.nonzero(tensor == 1, as_tuple=True)[0]
    return indices.tolist()


def get_tiny_dataset(ds,df,MAX_IDs=200,mode="train"):

    idxs=dict()
    dico_stats=dict()
    dico_tmp_counts=dict()


    for i in range(len(ds)):
        split=get_split(df,i)
        if split!=mode:
            continue
        _, tmp_lbl = ds[i]
        tmp_id=get_one_hot_indices(tmp_lbl)

    

        if len(tmp_id)>1:
            continue

        

        if not tmp_id[0] in dico_stats:
            dico_stats[tmp_id[0]]=1
        else:

            if dico_stats[tmp_id[0]]>=MAX_IDs:
                continue
            dico_stats[tmp_id[0]]+=1
        if not tmp_id[0] in idxs:
            idxs[tmp_id[0]]=[]

        idxs[tmp_id[0]].append(i)

    


    for i in range(len(ds)):
        split=get_split(df,i)
        if split!=mode:
            continue

        if i in idxs:
            continue

        _, tmp_lbl = ds[i]
        L_id=get_one_hot_indices(tmp_lbl)

        min_rep=-1
        min_rep_id=-1
        for tmp_idx in L_id:
            if not tmp_idx in dico_stats:
                dico_stats[tmp_idx]=0
                
            if dico_stats[tmp_idx]>=MAX_IDs:
                continue
            if min_rep==-1 or dico_stats[tmp_idx]<min_rep:
                min_rep=dico_stats[tmp_idx]
                min_rep_id=tmp_idx



        if min_rep_id==-1:
            continue

    
        
        
        dico_stats[min_rep_id]+=1

        if not min_rep_id in idxs:
            idxs[min_rep_id]=[]
        
        idxs[min_rep_id].append(i)

    print("summary dico stats")
    print(dico_stats)
    

    return idxs,dico_stats
    

        
    
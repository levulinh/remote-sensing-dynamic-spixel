import os

data_root = '/home/ncl/vlle/linh/datasets/AID/'
train_dir = os.path.join(data_root, 'train')
val_dir = os.path.join(data_root, 'val')

for sub_dir in os.listdir(train_dir):
    sub_dir_full = os.path.join(train_dir, sub_dir)
    for file_name in os.listdir(sub_dir_full):
        f_idx = int(file_name.split('_')[1][:-4])
        if f_idx > 100:
            file_full_path = os.path.join(sub_dir_full, file_name)
            os.remove(file_full_path)
            print(f'Removed {file_name}!!!')

for sub_dir in os.listdir(val_dir):
    sub_dir_full = os.path.join(val_dir, sub_dir)
    for file_name in os.listdir(sub_dir_full):
        f_idx = int(file_name.split('_')[1][:-4])
        if f_idx > 100:
            file_full_path = os.path.join(sub_dir_full, file_name)
            os.remove(file_full_path)
            print(f'Removed {file_name}!!!')  
import os
import shutil

src_root_folder = 'flower_photos'
dest_root_folder = 'dataset'
train_test_ratio = [9,1]
flower_name_list = os.listdir(src_root_folder)
for flower in flower_name_list:
    photo_list = os.listdir(os.path.join(src_root_folder, flower))
    train_ratio = train_test_ratio[0]/(train_test_ratio[0]+train_test_ratio[1])
    offset = int(train_ratio*len(photo_list))
    train_list = photo_list[:offset]
    test_list = photo_list[offset:]
    #保存训练集
    for photo in train_list:
        src_path = os.path.join(src_root_folder, flower, photo)
        dest_path = os.path.join(dest_root_folder, 'train', flower, photo)
        if not os.path.exists(os.path.join(dest_root_folder, 'train', flower)):
            os.makedirs(os.path.join(dest_root_folder, 'train', flower))
        shutil.copy(src_path, dest_path)
    # 保存测试
    for photo in test_list:
        src_path = os.path.join(src_root_folder, flower, photo)
        dest_path = os.path.join(dest_root_folder, 'test', flower, photo)
        if not os.path.exists(os.path.join(dest_root_folder, 'test', flower)):
            os.makedirs(os.path.join(dest_root_folder, 'test', flower))
        shutil.copy(src_path, dest_path)
        
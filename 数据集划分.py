import random
import os
import json

def read_split_data(root, val_rate=0.2):
    random.seed(0)
    assert os.path.exists(root), "dataset root:{} does not exists!!!".format(root)
    classes_name = [cls for cls in os.listdir(root) if os.path.isdir(os.path.join(root, cls))]
    class_indices = dict((k, v) for v, k in enumerate(classes_name))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()))
    with open('class_indices.json', 'w') as f:
        f.write(json_str)
    train_images_path, train_labels_path = [], []
    val_images_path, val_labels_path = [], []
    every_class_num = []
    for cls in classes_name:
        cls_path = os.path.join(root, cls)
        images = [os.path.join(root, cls, name) for name in os.listdir(cls_path) if name.endswith('.jpg')]
        # 类别对应的编号
        image_class_num = class_indices[cls]
        every_class_num.append(len(images))
        valid_path = random.sample(images, k=int(len(images)*val_rate))
        for img_path in images:
            if img_path in valid_path:
                val_images_path.append(img_path)
                val_labels_path.append(image_class_num)
            else:
                train_images_path.append(img_path)
                train_labels_path.append(image_class_num)


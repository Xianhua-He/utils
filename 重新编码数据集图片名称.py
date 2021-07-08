import shutil
import os
import os.path as osp
import pdb

if __name__ == '__main__':

  root_img = 'share_data/image_bak'
  root_label = '/share_data/annotation_bak'

  out_img = 'share_data/image'
  out_label = 'share_data/annotation'

  task_list = [1, 2]
  successed_list = []
  failed_list=[]
  
  for task_id in task_list:

    task_id = str(task_id)
    img_path = osp.join(root_img, task_id)
    label_path = osp.join(root_label, task_id)

    print('Processing task_%s ......'%task_id)
    print('-'*60)

    out_image_path = osp.join(out_img, task_id)
    out_label_path = osp.join(out_label, task_id)

    if not os.path.exists(img_path):
        print('%s is not existed! check the task_id!!!'%img_path)
        failed_list.append(task_id)
        continue
    if not os.path.exists(label_path):
        print('%s is not existed! check the task_id!!!'%label_path)
        failed_list.append(task_id)
        continue
    
    # 1.get images list
    img_list = []
    for root, dirs, files in os.walk(img_path):
      for f in files:
            img_list.append(os.path.join(root, f))
    img_list = sorted(img_list)

    # 2.get labels list
    label_list = []
    for root, dirs, files in os.walk(label_path):
      for f in files:
            label_list.append(os.path.join(root, f))
    label_list = sorted(label_list)

    # 3. check label/image ids to make sure label_ids == image_ids
    img_ids = [_.split('/')[-1].split('.')[0] for _ in img_list]
    label_ids = [_.split('/')[-1].split('.')[0] for _ in label_list]
    if img_ids == label_ids:
        pass
    else:
        failed_list.append(task_id)
        print('%s img_ids!=label_ids, rename passed this task'%task_id)
        continue

    if not os.path.exists(out_image_path):
        os.makedirs(out_image_path)
        print('%s is created!'%out_image_path)

    if not os.path.exists(out_label_path):
        os.makedirs(out_label_path)
        print('%s is created!'%out_label_path)

    # 4. copy image/label to new floder and rename
    count = 0

    for img_name,label_name in zip(img_list, label_list):
        if not osp.getsize(img_name) or not osp.getsize(label_name):
            print('%s has been removed because it is empty (0KB)'%img_name)
            os.remove(img_name)
            os.remove(label_name)
            continue
        count = count+1
        res_img   = osp.join(out_image_path, '%s_%06d.jpg'%(task_id,count))
        res_label = osp.join(out_label_path, '%s_%06d.json'%(task_id,count))
        shutil.copy(img_name, res_img)
        shutil.copy(label_name, res_label)
    print('Task_%s rename successed!(%d images)'%(task_id,count))

    successed_list.append(task_id)
print('successed_list:', successed_list)
print('failed_list:', failed_list)
	

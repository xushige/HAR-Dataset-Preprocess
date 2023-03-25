import requests
import shutil
from clint.textui import progress
import os
# 数据集下载
def download_dataset(dataset_name, file_url, dir_path):
    print('\n==================================================【 %s 数据集下载】===================================================\n'%(dataset_name))
    print('url: %s\n'%(file_url))

    if dataset_name == 'UniMiB-SHAR' and file_url[-4:] == '.git': # 由于unimib数据集无法直接访问下载，这里我把unimib数据集上传到github进行访问clone
        if os.path.exists(os.path.join(dir_path, dataset_name)):
            shutil.rmtree(os.path.join(dir_path, dataset_name))
        os.system('git clone %s %s/%s' % (file_url, dir_path, dataset_name))
        
    else: # 其他数据集
        r = requests.get(url=file_url, stream=True)
        with open(os.path.join(dir_path, 'dataset.zip'),mode='wb') as f:  # 需要用wb模式
            total_leng = int(r.headers.get('content-length'))
            for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_leng / 1024) + 1):
                if chunk:
                    f.write(chunk)
                    f.flush()
        for format in ["zip", "tar", "gztar", "bztar", "xztar"]:
            try:
                shutil.unpack_archive(filename=os.path.join(dir_path, 'dataset.zip'),extract_dir=dir_path, format=format)
                break
            except:
                continue
        os.remove(os.path.join(dir_path, r'dataset.zip'))

    print()

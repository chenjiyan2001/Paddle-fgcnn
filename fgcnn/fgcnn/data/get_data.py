# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    print('load data...')
    data = pd.read_csv('./train.csv')
    print('memory useage:', round(data.memory_usage(deep=True).sum() / 1024 ** 3, 4), 'GB')
    
    sparse_features = [
        'hour',
        'C1',
        'banner_pos',
        'site_id',
        'site_domain',
        'site_category',
        'app_id',
        'app_domain',
        'app_category',
        'device_id',
        'device_model',
        'device_type',
        'device_conn_type',  # 'device_ip',
        'C14',
        'C15',
        'C16',
        'C17',
        'C18',
        'C19',
        'C20',
        'C21',
    ]

    data['day'] = data['hour'].apply(lambda x: str(x)[4:6])
    data['hour'] = data['hour'].apply(lambda x: str(x)[6:])

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    print('get label encoder:')
    line = ''
    for feat in sparse_features:
        print('\t' + feat)
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
        data[feat] = data[feat].astype('category')
        drop = [k for k, v in data[feat].value_counts().items() if v < 5]
        data[feat].cat.remove_categories(drop)
        new_cate = -1 if str(data[feat].cat.categories.dtype) == 'int64' else '-1'
        data[feat].cat.add_categories(new_cate).fillna(new_cate)
        line += str(len(lbe.classes_)) + ','


    # 2. get vacob file
    print('get vacob file')
    vacob_file = open('vacob_file.txt', 'w')
    vacob_file.write(line)
    vacob_file.close()

    # 3. split dataset
    print('split dataset')
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,2:], data.iloc[:,1], test_size=0.25, random_state=42)
    X_train['click'] = y_train
    X_test['click'] = y_test
    del data
    del y_train
    del y_test
    
    # 4. save
    print('save...')
    X_train['click'].to_hdf('./train/train_data.h5', key='label', format='t')
    X_train.iloc[:, :-1].to_hdf('./train/train_data.h5', key='data', format='t')
    print('train data done')

    X_test['click'].to_hdf('./test/test_data.h5', key='label', format='t')
    X_test.iloc[:, :-1].to_hdf('./test/test_data.h5', key='data', format='t')
    print('test data done')

    print('done')
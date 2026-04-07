m=500
config = {
    'exp_name' : f'timestamp_{m}_k=5_test',
    'worker' : 1,
    'epoch' : 17,
    'lr' : 0.1,
    'batch' : 32,
    'test_only' : False,
    'resume' : '',
    'train_dataset' : "./dataset/casia_train_lmdbdataset_path",
    'test_dataset': "./dataset/ic13_test_lmdbdataset_path",
    'pos_entropy_path' : "./radical_encoder/radical_entropy.pt",
    'radical_encode' : "./radical_encoder/radical_features.pt",
    'data_classify' : "./data/text_features_classify.pt",
    'alpha_path' : './data/charlist_3755.txt',
    'radical_path': './data/radical_all_Chinese.txt',
    'decompose_path': './data/decompose.txt',
}

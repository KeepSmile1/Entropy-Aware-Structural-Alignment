import lmdb

env = lmdb.open('hwdb1x_train', readonly=True)
with env.begin() as txn:
    num_samples = int(txn.get(b'num-samples'))
    print(f"sample count: {num_samples}")


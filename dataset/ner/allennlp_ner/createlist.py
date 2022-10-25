from glob import glob
from os.path import exists
p='data_path/train/*'
outfile='trainlist.txt'
assert not exists(outfile)


fp=open(outfile, 'w')

files = glob(p + '/*.json')

for f in files:
    fp.write(f + '\n')

fp.close()

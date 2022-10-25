from glob import glob
from os.path import exists
p='/home/s1959796/csqparsing/dataversion_aug27_2022/CSQA_v9_skg.v6_compar_spqres9_subkg2_tyTop_nelctx_cleaned/valid/*'
outfile='validlist.txt'
assert not exists(outfile)


fp=open(outfile, 'w')

files = glob(p + '/*.json')

for f in files:
    fp.write(f + '\n')

fp.close()

p='/home/s1959796/csqparsing/dataversion_aug27_2022/CSQA_v9_skg.v6_compar_spqres9_subkg2_tyTop_nelctx_cleaned/test/*'
outfile='testlist.txt'
assert not exists(outfile)


fp=open(outfile, 'w')

files = glob(p + '/*.json')

for f in files:
    fp.write(f + '\n')

fp.close()

p='/home/s1959796/csqparsing/dataversion_aug27_2022/CSQA_v9_skg.v6_compar_spqres9_subkg2_tyTop_nelctx_cleaned/train/*'
outfile='trainlist.txt'
assert not exists(outfile)


fp=open(outfile, 'w')

files = glob(p + '/*.json')

for f in files:
    fp.write(f + '\n')

fp.close()

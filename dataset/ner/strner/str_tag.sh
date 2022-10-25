start=0
end=-1
python str_nel.py  \
   -data_path "datapath/" \
  -save_path "strnel_data/" \
  -file_path "validlist.txt" \
  -dataset 'valid' \
  -start $start  \
  -end $end 

python str_nel.py  \
   -data_path "datapath/" \
  -save_path "strnel_data/" \
  -file_path "testlist.txt" \
  -dataset 'test' \
  -start $start  \
  -end $end \
  -n_cpus 5

python str_nel.py  \
   -data_path "datapath" \
  -save_path "strnel_data/" \
  -file_path "trainlist.txt" \
  -dataset 'train' \
  -start $start  \
  -end $end \
  -n_cpus 5

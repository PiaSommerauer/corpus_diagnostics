




#for prop in lay_eggs blue green yellow used_in_cooking; do
#prop=lay_eggs
path=../../data/datasets/model/wiki

for prop in lay_eggs blue; do
  python3 extract_contexts.py "$prop" "$path" &
done
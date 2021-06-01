for prop in lay_eggs blue green yellow used_in_cooking; do
  python extract_contexts "$prop" ../../data/datasets/models/wiki &
done
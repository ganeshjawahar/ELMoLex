import sys
import os

dest_path = sys.argv[1]

epochs=0
for i in range(5):
  cur_fold_dest = dest_path + '/fold' + str(i)
  assert(os.path.exists(cur_fold_dest+"/out_train"))
  best_epoch = 0
  with open(cur_fold_dest+"/out_train") as f:
    for line in f:
      content = line.strip()
      if content.startswith("las:"):
        items = content.split()
        best_epoch = int(items[5])
  epochs+=best_epoch
epochs=epochs//5
print(epochs)

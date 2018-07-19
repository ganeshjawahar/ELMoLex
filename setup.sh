conda create -n conll18 python=3.6
# assuming CUDA 9.0! if not, find the right command from: https://pytorch.org/
conda install pytorch torchvision cuda90 -c pytorch
conda install tqdm
conda install networkx
conda install xlwt
source activate conll18
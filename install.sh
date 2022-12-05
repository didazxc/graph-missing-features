# for windows, you can use this cmd to run the shell script:
# C:\'Program Files'\Git\bin\bash.exe

TORCH=1.13.0
CUDA=cu116
RUN_CMD="conda run -n py39"

$RUN_CMD pip install tqdm numpy pandas scipy
$RUN_CMD pip install torch==${TORCH}+${CUDA} -f https://download.pytorch.org/whl/torch_stable.html
$RUN_CMD pip install torch-scatter==2.1.0 -f "https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html"
$RUN_CMD pip install torch-sparse==0.6.15 -f "https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html"
$RUN_CMD pip install torch-cluster==1.6.0 -f "https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html"
$RUN_CMD pip install torch-spline-conv==1.2.1 -f "https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html"
$RUN_CMD pip install torch_geometric==2.2.0
$RUN_CMD pip install ogb==1.3.5

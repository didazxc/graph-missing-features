# for windows, you can use this cmd to run the shell script:
# C:\'Program Files'\Git\bin\bash.exe

TORCH=1.13.0
CUDA=cpu
RUN_CMD="conda run -n py39"

# conda create -n py39 python=3.9 -y

$RUN_CMD pip install tqdm numpy pandas scipy
$RUN_CMD pip install torch==${TORCH}+${CUDA} -f https://download.pytorch.org/whl/torch_stable.html
$RUN_CMD pip install pyg-lib==0.1.0 \
torch-scatter==2.1.0 \
torch-sparse==0.6.15 \
torch-cluster==1.6.0 \
torch-spline-conv==1.2.1 \
torch_geometric==2.2.0 \
-f "https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html"
$RUN_CMD pip install ogb==1.3.5

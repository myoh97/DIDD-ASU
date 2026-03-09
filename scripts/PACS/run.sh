DEVICE=6
lambda_a=1e-4
lambda_b=1e-4
for ipc in 1 10 20; do
    CUDA_VISIBLE_DEVICES=$DEVICE python dil_condense_m3d.py \
    --ipc=$ipc \
    --dataset=PACS \
    --asym \
    --lambda_alpha $lambda_a \
    --lambda_beta $lambda_b \
    --flag=asym_${lambda_a}_${lambda_b}
done


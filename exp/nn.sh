points=(50    100   200   500   800   1000  2000  5000)
canvas=(10000 10000 10000 10000 10000 10000 10000 10000)

m=15
kb=4
k=10
set="m\=15_kb\=4-threshold\=0.95"
model_path="models/m\=15_kb\=4.pt"
device="cuda:0"

dist="uniform"

for i in "${!points[@]}"; do
    num=${points[$i]}
    width=${canvas[$i]}
    set="${num}_${width}x${width}-$dist"
    test_set="points/point$set-100-pt"
    output="exp_out/$set-$set"
    cmd="python -m arora flow=nn_exp \
    quadtree.m=$m quadtree.kb=$kb \
    model.device=$device nn_exp.model=$model_path \
    nn_exp.test_set=$test_set nn_exp.output=$output nn_exp.k=$k \
    quadtree.bbox.width=$width quadtree.bbox.height=$width"
    echo $cmd
    $cmd
done
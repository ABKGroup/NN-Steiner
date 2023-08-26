points=(100 100 100 100   200 200 400 500 800 800 1000 2000)
canvas=(100 200 800 10000 140 200 200 500 280 500 1000 2000)


for i in "${!points[@]}"; do
    num=${points[$i]}
    width=${canvas[$i]}
    set="${num}_${width}x${width}"
    test_set="points/point$set-100-pt"
    output="exp_out/$set"
    cmd="python -m arora flow=mst_exp \
    mst_exp.test_set=$test_set mst_exp.output=$output"
    # echo $cmd
    $cmd
done
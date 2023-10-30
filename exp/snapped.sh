# points=(50    100   200   500   800   1000)
# canvas=(10000 10000 10000 10000 10000 10000)
points=(50    100   200   500   800   1000)
canvas=(10000 10000 10000 10000 10000 10000)

dist="uniform"

for i in "${!points[@]}"; do
    num=${points[$i]}
    width=${canvas[$i]}
    set="${num}_${width}x${width}-$dist"
    test_set="points/point$set-100-pt"
    output="exp_out/$set"
    cmd="python -m arora flow=snapped_exp \
    snapped_exp.test_set=$test_set snapped_exp.output=$output \
    quadtree.bbox.width=$width quadtree.bbox.height=$width"
    echo $cmd
    $cmd
done
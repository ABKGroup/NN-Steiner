points=(50    100   200   500   800   1000  2000  5000)
canvas=(10000 10000 10000 10000 10000 10000 10000 10000)

tree_dir_path="flute_trees"
dist="uniform"

for i in "${!points[@]}"; do
    num=${points[$i]}
    width=${canvas[$i]}
    set="${num}_${width}x${width}-$dist"
    point_dir="points/point$set-100-pt"
    tree_dir="${tree_dir_path}flute${set}"
    output="exp_out/$set-flute"
    cmd="python -m arora flow=refine \
    refine.point_dir=$point_dir refine.tree_dir=$tree_dir \
    refine.output=$output"
    echo $cmd
    $cmd
done
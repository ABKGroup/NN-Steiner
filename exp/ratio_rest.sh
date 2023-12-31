points=(50    100   200   500   800   1000  2000  5000)
canvas=(10000 10000 10000 10000 10000 10000 10000 10000)

dist="uniform"

base="m=15_kb=4-solved"
path="REST_out"


for i in "${!points[@]}"; do
    num=${points[$i]}
    width=${canvas[$i]}
    set="${num}_${width}x${width}-$dist"
    comp_set="$path/$set.txt"
    base_set="exp_out/$set-$base-norm.txt"
    cmd="python evaluator/evaluateRatio.py $comp_set $base_set"
    echo $cmd
    $cmd
done
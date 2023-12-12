points=(50    100   200   500   800   1000  2000  5000)
canvas=(10000 10000 10000 10000 10000 10000 10000 10000)
# points=(50    100   200   500   800   1000)
# canvas=(10000 10000 10000 10000 10000 10000)

dist="uniform"

# comp="snapped"
comp="m=15_kb=4-threshold=0.95-cell-solved"
# comp="flute-refined"
# comp="golden"
path="flute_out"


for i in "${!points[@]}"; do
    num=${points[$i]}
    width=${canvas[$i]}
    set="${num}_${width}x${width}-$dist"
    comp_set="exp_out/$set-$comp.txt"
    base_set="$path/flute$set.txt"
    cmd="python evaluator/evaluateRatio.py $comp_set $base_set"
    echo $cmd
    $cmd
done
points=(50    100   200   500   800   1000  2000  5000)
canvas=(10000 10000 10000 10000 10000 10000 10000 10000)

dist="uniform"

for i in "${!points[@]}"; do
    num=${points[$i]}
    width=${canvas[$i]}
    cmd="python -m arora flow=data_gen \
    data_gen.num_points=$num data_gen.dist=$dist"
    echo $cmd
    $cmd
done
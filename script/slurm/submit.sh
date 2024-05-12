

for k in 10 20 30
do
    for h in 512 768 1280
    do
        sbatch --export=K=$k,H=$h --job-name=sol_"$k"_"$h" script/slurm/run_ft.slurm
    done
done
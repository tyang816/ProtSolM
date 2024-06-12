

for k in 10 20 30
do
    for h in 512
    do
        sbatch --export=K=$k,H=$h --job-name=sol_"$k"_"$h",--nodelist=ZSGPU10 script/slurm/run_ft.slurm
    done
done

for k in 20
do
    for h in 512
    do
        sbatch --export=K=$k,H=$h --job-name=sol_"$k"_"$h",--nodelist=ZSGPU10 script/slurm/run_ft_feature.slurm
    done
done

for i in {9126..9131}
do
    scancel $i
done
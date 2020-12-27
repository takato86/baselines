for rho in 0
do
for eta in 1
do
for i in 10 40 50
do
echo mpiexec -n 6 python -m baselines.run --alg=her_rs --env=FetchPickAndPlace-v1 --log_path=~/logs/202011_subgoal-based-seed=$i-eta=$eta-rho=$rho --num_timesteps=5e5 --seed=$i --eta=$eta --rho=$rho
mpiexec -n 6 python -m baselines.run --alg=her_rs --env=FetchPickAndPlace-v1 --log_path=~/logs/202011_subgoal-based-seed=$i-eta=$eta-rho=$rho --num_timesteps=5e5 --seed=$i --eta=$eta --rho=$rho
done
done
done
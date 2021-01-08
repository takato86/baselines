for i in 20 30 40 50
do
echo mpiexec -n 6 python -m baselines.run --alg=her_rs --env=FetchPickAndPlace-v1 --log_path=~/logs/202011_sarsa-rs-seed=$i --num_timesteps=5e5 --seed=$i 
mpiexec -n 6 python -m baselines.run --alg=her_rs --env=FetchPickAndPlace-v1 --log_path=~/logs/202101_sarsa-rs-seed=$i --num_timesteps=5e5 --seed=$i 
done

# --oversubscribe
# python -m baselines.run --alg=her_rs --env=FetchPickAndPlace-v1 --log_path=~/logs/202101_sarsa-rs-seed=10 --num_timesteps=5e5 --seed=10 
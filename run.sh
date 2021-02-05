ETA=1
RHO=0
VINIT=0
ENV=SingleFetchPickAndPlace-v0
SHAPING=dta
for i in 20 30 40 50
do
# Methods
LOG_PATH=~/logs/202102_m2s_$SHAPING-vinit=$VINIT-eta=$ETA-seed=$i
echo mpiexec -n 10 python -m baselines.run --alg=her_rs --env=$ENV --shaping=$SHAPING --log_path=$LOG_PATH --num_timesteps=5e5 --seed=$i --vinit=$VINIT --eta=$ETA --rho=$RHO --is_ddpg=True
mpiexec -n 10 python -m baselines.run --alg=her_rs --env=$ENV --shaping=$SHAPING --log_path=$LOG_PATH --num_timesteps=5e5 --seed=$i --vinit=$VINIT --eta=$ETA --rho=$RHO --is_ddpg=True
# Baseline
# LOG_PATH=~/logs/202102_m2s_ddpg-seed=$i
# echo mpiexec -n 10 python -m baselines.run --alg=her_rs --env=$ENV --log_path=$LOG_PATH --num_timesteps=5e5 --seed=$i --is_ddpg=True
# mpiexec -n 10 python -m baselines.run --alg=her_rs --env=$ENV --log_path=$LOG_PATH --num_timesteps=5e5 --seed=$i --is_ddpg=True
done
# mpiexec -n 6 python -m baselines.run --alg=her_rs --env=FetchPickAndPlace-v1 --log_path=~/logs/202101_dta-seed=0 --num_timesteps=5e5 --seed=0
# --oversubscribe
# python -m baselines.run --alg=her_rs --env=FetchPickAndPlace-v1 --log_path=~/logs/202101_sarsa-rs-seed=10 --num_timesteps=5e5 --seed=10 
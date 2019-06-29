# rl-study

1. dqn_agent.py没搞定，不收敛，使用dqn_openai.py
2. a2c_openai.py没搞定，eplenmean and eprewmean always nan，使用如下方法
  * python run.py --alg=a2c --env=SnakeEnv-v111 --num_timesteps=2e6 --num_env=8 --save_path=models/a2c
  * python run.py --alg=a2c --env=SnakeEnv-v111 --num_timesteps=0 --num_env=1 --load_path=models/a2c --play
  * 但是效果不如dqn好



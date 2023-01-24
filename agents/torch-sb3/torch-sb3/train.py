import stable_baselines3 as SB3
import gym
import os
import multiprocess
import torchSB3.env.RLteacherWrapper as RLteacherEnv

def train_SB3(
        env_id,
        algorithm,
        predictor,
        make_env=gym.make,
        workers=1,
        total_timesteps=10000
):
    # Tensorflow is not fork-safe, so we must use spawn instead
    # https://github.com/tensorflow/tensorflow/issues/5448#issuecomment-258934405
    # We use multiprocess rather than multiprocessing because Keras sets a multiprocessing context
    if not os.environ.get("SET_PARALLEL_TRPO_START_METHOD"): # Use an env variable to prevent double-setting
        multiprocess.set_start_method('spawn')
        os.environ['SET_PARALLEL_TRPO_START_METHOD'] = "1"

    # TODO: create environment
    model = getattr(SB3, algorithm)(
        "MlpPolicy", # TODO: custom model available
        RLteacherEnv(make_env(env_id), predictor),
        verbose=1)
    
    model.learn(total_timesteps=total_timesteps)

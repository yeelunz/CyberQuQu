from app.main import create_app

import logging
import ray
from utils.var_update import init_global_var

from ray.rllib.algorithms.ppo import PPOConfig
from utils.global_var import globalVar as gv

app = create_app()


if __name__ == '__main__':
    init_global_var()
    # test
    app.run(host='0.0.0.0', port=5000, debug=False)




from app.main import create_app

import logging
import ray
import os
import sys

from ray.rllib.algorithms.ppo import PPOConfig



app = create_app()


if __name__ == '__main__':

    # test
    app.run(host='0.0.0.0', port=5000, debug=False)




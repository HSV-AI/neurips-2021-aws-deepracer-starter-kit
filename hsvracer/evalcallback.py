from pytorch_lightning.callbacks import Callback
import torch
from icecream import ic
import cv2
import numpy as np

class EvaluationCallback(Callback):
    def __init__(self, env):
        self.env = env

    # def on_init_start(self, trainer):
    #     print("Starting to init trainer!")

    # def on_init_end(self, trainer):
    #     print("trainer is init now")

    # def on_train_end(self, trainer, pl_module):
    #     print("do something when training ends")

    # def on_batch_end(self, trainer, pl_module):
    #     print("Batch End")
    #     # self.evaluate_model(trainer=trainer, model=pl_module)

    def on_epoch_end(self, trainer, pl_module):
        self.evaluate_model(trainer=trainer, model=pl_module)

    @torch.no_grad()
    def evaluate_model(self, trainer, model):

        steps_completed = 0
        total_reward = 0

        # torch.set_grad_enabled(False)
        # model.eval()

        # Do the initial reset        
        observations = self.env.reset()

        done = False

        while not done:

            # Get the action to take from the model
            observation = observations['STEREO_CAMERAS']
            observation = np.swapaxes(observation, 0, 2)
            state = torch.FloatTensor(observation).to(model.device)

            pi, action, value = model(state)
            # action = model(state)

            action = int(torch.argmax(action).item())

            # Pass the action to the env step and get the observations and done
            observations, reward, done, info = self.env.step(action)

            steps_completed += 1 
            total_reward += reward

        avg_reward = total_reward / steps_completed

        model.log("eval_steps", steps_completed, prog_bar=True, on_step=False, on_epoch=True)
        model.log("eval_total_reward", total_reward, prog_bar=True, on_step=False, on_epoch=True)
        model.log("eval_avg_reward", avg_reward, prog_bar=True, on_step=False, on_epoch=True)

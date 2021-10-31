from pytorch_lightning.callbacks import Callback
import torch
from icecream import ic

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
    #     self.evaluate_model(trainer=trainer, model=pl_module)

    def on_epoch_end(self, trainer, pl_module):
        self.evaluate_model(trainer=trainer, model=pl_module)

    def evaluate_model(self, trainer, model):

        steps_completed = 0
        total_reward = 0

        torch.set_grad_enabled(False)
        model.eval()

        # Do the initial reset        
        observations = self.env.reset()

        completed = False

        while not completed:

            # Get the action to take from the model
            observation = observations['STEREO_CAMERAS']
            state = torch.FloatTensor(observation).to(model.device)

            with torch.no_grad():
                pi, action, value = model(state)

            # action = torch.argmax(action).item()

            # Pass the action to the env step and get the observations and done
            observations, reward, done, info = self.env.step(action.cpu().numpy())

            steps_completed += 1 
            total_reward += reward

            if done:
                completed = True

        avg_reward = total_reward / steps_completed

        model.log("eval_steps", steps_completed, prog_bar=True, on_step=False, on_epoch=True)
        model.log("eval_total_reward", total_reward, prog_bar=True, on_step=False, on_epoch=True)
        model.log("eval_avg_reward", avg_reward, prog_bar=True, on_step=False, on_epoch=True)

        # enable grads + batchnorm + dropout
        torch.set_grad_enabled(True)
        model.train()
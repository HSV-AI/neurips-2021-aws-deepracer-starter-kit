import zmq
import time
import msgpack

import msgpack_numpy as m
m.patch()


class DeepracerZMQClient:
    def __init__(self, host="127.0.0.1", port=8888):
        print(f"Connecting to deepracer at {host}:{port}")
        self.host = host
        self.port = port
        self.socket = zmq.Context().socket(zmq.REQ)
        # Large timout for first connection
        self.socket.set(zmq.SNDTIMEO, 600000)  # 10m
        self.socket.set(zmq.RCVTIMEO, 600000)
        self.socket.connect(f"tcp://{self.host}:{self.port}")
    
    def set_agent_ready(self):
        packed_msg = msgpack.packb({"ready": 1})
        self.socket.send(packed_msg)

    def recieve_response(self):
        packed_response = self.socket.recv()
        response = msgpack.unpackb(packed_response)
        return response

    def send_msg(self, msg: dict):
        packed_msg = msgpack.packb(msg)
        self.socket.send(packed_msg)

        response = self.recieve_response()
        return response

class DeepracerEnvHelper:
    def __init__(self, port=8888):
        self.zmq_client = DeepracerZMQClient(port=port)
        self.zmq_client.set_agent_ready()
        self.obs = None

    def send_act_rcv_obs(self, action):
        action_dict = {"action": action}
        self.obs = self.zmq_client.send_msg(action_dict)
        return self.obs
    
    def env_reset(self):
        if self.obs is None: # First communication to zmq server
            self.obs = self.zmq_client.recieve_response()
            # Smaller timeout after first connection
            self.zmq_client.socket.set(zmq.SNDTIMEO, 20000)  # 20s
            self.zmq_client.socket.set(zmq.RCVTIMEO, 20000)

        else: # If prev_episode done and reset called, fast forward one step for new episode
            self.obs = self.send_act_rcv_obs(4) # Action ignored due to reset()

        return self.obs
    
    def unpack_rl_coach_obs(self, rl_coach_obs):
        observation = rl_coach_obs['_next_state']
        reward = rl_coach_obs['_reward']
        done = rl_coach_obs['_game_over']
        info = rl_coach_obs['info']
        if type(info) is not dict:
            info = {}
        info['goal'] = rl_coach_obs['_goal']
        return observation, reward, done, info

if __name__ == "__main__":
    client = DeepracerZMQClient()
    packed_msg = msgpack.packb({"ready": 1})
    client.socket.send(packed_msg)
    episodes_completed = 0
    steps_completed = 0
    while True:
        packed_response = client.socket.recv()
        env_response = msgpack.unpackb(packed_response)
        steps_completed += 1 
        if env_response['_game_over']:
            episodes_completed += 1
            print("Episodes Completed:", episodes_completed, "Steps:", steps_completed)
            steps_completed = 0
        packed_action = msgpack.packb({"action": 1})
        client.socket.send(packed_action)



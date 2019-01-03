import numpy as np
import matplotlib.pyplot as plt
from Box import Game

plt.ion()
game = Game(10,10,2,5)
fig = plt.figure()
img = plt.imshow(np.squeeze(game.get_state()*255.0),origin="lower")
plt.show()
command = input()
command = command.strip()
while command != "exit":
    command = int(command)
    reward,is_terminal = game.execute_action(command)
    print("action: {}  reward: {}  terminal: {}".format(Game.action_names[command],reward,is_terminal))
    state = game.get_state()*255.0
    img.set_data(np.squeeze(state))
    plt.draw()
    command = input()
    command = command.strip()

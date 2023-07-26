import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/tests_snn_pos/2023-04-03+15:12:52+kalman+none+cyberzoo+optitrackstate+hover.csv", skipinitialspace=True)
data = pd.read_csv("data/tests_pid_pos/2023-04-03+15:37:10+kalman+none+cyberzoo+optitrackstate+hover.csv", skipinitialspace=True)

# plt.figure()
# data.roll.plot()
# data.pitch.plot()
# data.yaw.plot()
# plt.legend()

# data.cmd_roll = data.cmd_roll / 32768
# data.cmd_pitch = data.cmd_pitch / 32768
# data.cmd_yaw = data.cmd_yaw / 32768

# data.cmd_roll.plot()
# data.cmd_pitch.plot()
# data.cmd_yaw.plot()
# plt.legend()

plt.figure()
plt.plot(data.timeTick, data.otX, '.')
plt.plot(data.timeTick, data.otY, '.')
# plt.plot(data.timeTick, data.otZ, '.')

plt.plot(data.timeTick, data.target_x)
plt.plot(data.timeTick, data.target_y)
# plt.plot(data.timeTick, data.target_z)
plt.legend()

# plt.figure()
# data.stateX.plot(x=data.timeTick)
# data.stateY.plot(x=data.timeTick)
# data.stateZ.plot(x=data.timeTick)
# plt.legend()



plt.show()
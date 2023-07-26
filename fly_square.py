"""
Execute a Crazyflie flight and log it.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

from scipy.spatial.transform import Rotation

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie import Console
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.log import LogConfig

import flight.utils as util
from flight.FileLogger import FileLogger
from flight.NatNetClient import NatNetClient

# TODO: merge these? (prepared trajectories and trajectories)
from flight.trajectories import takeoff, landing
from flight.prepared_trajectories import *


def create_filename(fileroot, keywords, estimator, uwb, optitrack, trajectory):
    # Date
    date = datetime.today().strftime(r"%Y-%m-%d+%H:%M:%S")

    # Additional keywords
    if keywords is not None:
        keywords = "+" + "+".join(keywords)
    else:
        keywords = ""

    # Options
    if optitrack == "logging":
        options = f"{estimator}+{uwb}{keywords}+optitracklog+{'_'.join(trajectory)}"
    elif optitrack == "state":
        options = f"{estimator}+{uwb}{keywords}+optitrackstate+{'_'.join(trajectory)}"
    else:
        options = f"{estimator}+{uwb}{keywords}+{'_'.join(trajectory)}"

    # Join
    if fileroot[-1] == "/":
        return f"{fileroot}{date}+{options}.csv"
    else:
        return f"{fileroot}/{date}+{options}.csv"


def setup_logger(
    cf, uri, fileroot, keywords, logconfig, estimator, uwb, flow, optitrack, trajectory
):
    # Create filename from options and date
    file = create_filename(fileroot, keywords, estimator, uwb, optitrack, trajectory)
    print(f"Log location: {file}")

    # Logger setup
    flogger = FileLogger(cf, uri, logconfig, file)

    # Enable log configurations based on system setup:
    # Defaults
    flogger.enableConfig("attitude")
    flogger.enableConfig("gyros")
    flogger.enableConfig("acc")
    flogger.enableConfig("state")
    flogger.enableConfig("ctrltarget")
    # flogger.enableConfig("controller")


    # UWB
    if uwb == "twr":
        flogger.enableConfig("twr")
    elif uwb == "tdoa":
        print("Needs custom TDoA logging in firmware!")
        # For instance, see here: https://github.com/Huizerd/crazyflie-firmware/blob/master/src/utils/src/tdoa/tdoaEngine.c
        # flogger.enableConfig("tdoa")
    # Flow
    if flow:
        flogger.enableConfig("laser")
        flogger.enableConfig("flow")
    # OptiTrack
    if optitrack != "none":
        flogger.enableConfig("otpos")
        flogger.enableConfig("otatt")
    # Estimator
    if estimator == "kalman":
        flogger.enableConfig("kalman")

    # Start
    flogger.start()
    print("Logging started")

    return flogger, file


def setup_optitrack(optitrack):
    # If we don't use OptiTrack
    if optitrack == "none":
        ot_position = None
        ot_attitude = None
    # If we do use OptiTrack
    else:
        # Global placeholders
        ot_position = np.zeros(3)
        ot_attitude = np.zeros(3)

        # Streaming client in separate thread
        streaming_client = NatNetClient()
        streaming_client.newFrameListener = receive_new_frame
        streaming_client.rigidBodyListener = receive_rigidbody_frame
        streaming_client.run()
        print("OptiTrack streaming client started")

    # TODO: do we need to return StreamingClient?
    return ot_position, ot_attitude

def wait_for_position_estimator(scf):
    print('Waiting for estimator to find position...')

    log_config = LogConfig(name='Kalman Variance', period_in_ms=500)
    log_config.add_variable('kalman.varX', 'float')
    log_config.add_variable('kalman.varY', 'float')
    log_config.add_variable('kalman.varZ', 'float')

    var_y_history = [1000] * 10
    var_x_history = [1000] * 10
    var_z_history = [1000] * 10

    threshold = 0.0001

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            # ot_position = [1, 1, 0]
            cf.extpos.send_extpos(
                ot_position[0], ot_position[1], ot_position[2]
            )
        
            data = log_entry[1]

            var_x_history.append(data['kalman.varX'])
            var_x_history.pop(0)
            var_y_history.append(data['kalman.varY'])
            var_y_history.pop(0)
            var_z_history.append(data['kalman.varZ'])
            var_z_history.pop(0)

            min_x = min(var_x_history)
            max_x = max(var_x_history)
            min_y = min(var_y_history)
            max_y = max(var_y_history)
            min_z = min(var_z_history)
            max_z = max(var_z_history)

            print("{} {} {}".
                  format(max_x - min_x, max_y - min_y, max_z - min_z))

            # print(f"{ot_position[0] - var_x_history[-1]}, {ot_position[1] - var_y_history[-1]}, {ot_position[2] - var_z_history[-1]}")
            # print(f"{var_x_history[-1]}, {var_y_history[-1]}, {var_z_history[-1]}, {np.abs(ot_position[0] - var_x_history[-1])}")

            # if (np.abs(ot_position[0] - var_x_history[-1]) < threshold) and \
            #     (np.abs(ot_position[1] - var_y_history[-1]) < threshold) and \
            #     (np.abs(ot_position[2] - var_z_history[-1]) < threshold):
            #      break
            if (max_x - min_x) < threshold and (
                    max_y - min_y) < threshold and (
                    max_z - min_z) < threshold:
                break
                 
def start_onboard_logging(cf):
    cf.param.set_value("usd.logging", "1")

def stop_onboard_logging(cf):
    cf.param.set_value("usd.logging", "0")

def reset_estimator(cf, estimator):
    # Kalman
    if estimator == "kalman":
        cf.param.set_value("kalman.resetEstimation", "1")
        time.sleep(1)
        cf.param.set_value("kalman.resetEstimation", "0")


def receive_new_frame(*args, **kwargs):
    pass

def activate_kalman_estimator(cf):
    cf.param.set_value('stabilizer.estimator', '2')
    print(cf.param.get_value('stabilizer.estimator'))


def receive_rigidbody_frame(id, position, rotation):
    # Modify globals
    # TODO: is flogger needed?
    global flogger, ot_position, ot_attitude

    # Check ID
    if id == ot_id:
        # Register position
        ot_position = util.ot2control(position)
        ot_pos_dict = {
            "otX": ot_position[0],
            "otY": ot_position[1],
            "otZ": ot_position[2],
        }
        flogger.registerData("otpos", ot_pos_dict)

        # Register attitude
        rotation_euler = util.quat2euler(rotation)
        ot_attitude = util.ot2control(rotation_euler)
        ot_att_dict = {
            "otRoll": ot_attitude[0],
            "otPitch": ot_attitude[1],
            "otYaw": ot_attitude[2],
        }
        flogger.registerData("otatt", ot_att_dict)


def console_cb(text):
    global console_log
    console_log.append(text)


def do_taskdump(cf):
    cf.param.set_value("system.taskDump", "1")


def process_taskdump(file, console_log):
    # Dataframe placeholders
    label_data, load_data, stack_data = [], [], []

    # Get headers
    headers = []
    for i, line in enumerate(console_log):
        if "Task dump" in line:
            headers.append(i)
    # None indicates the end of the list
    headers.append(None)

    # Get one task dump
    
    for i in range(len(headers) - 1):
        dump = console_log[headers[i] + 2 : headers[i + 1]]

        # Process strings: strip \n, \t, spaces, SYSLOAD:
        loads, stacks, labels = [], [], []
        
        for line in dump:
            if line[0] != 'W':
                entries = line.strip("SYSLOAD: ").split("\t")
                if entries[0].split()[0] != "TEENSY:":
                    if len(entries) < 3:
                        print(entries)
                    else:
                        loads.append(entries[0].strip())  # no sep means strip all space, \n, \t
                        stacks.append(entries[1].strip())
                        labels.append(entries[2].strip())
                else:
                    print(entries)

        # Store labels
        if not label_data:
            label_data = labels

        # Append to placeholders    
        load_data.append(loads)
        stack_data.append(stacks)

    # Check if we have data at all
    if headers[0] is not None and label_data:
        # Put in dataframe
        load_data = pd.DataFrame(load_data, columns=label_data)
        stack_data = pd.DataFrame(stack_data, columns=label_data)

        # Save dataframes
        load_data.to_csv(file.strip(".csv") + "+load.csv", sep=",", index=False)
        stack_data.to_csv(file.strip(".csv") + "+stackleft.csv", sep=",", index=False)
    else:
        print("No task dump data found")

def activate_high_level_commander(cf):
    cf.param.set_value('commander.enHighLevel', '1')

def send_extpose_rot_matrix(cf, x, y, z, rot):
    """
    Send the current Crazyflie X, Y, Z position and attitude as a (3x3)
    rotaton matrix. This is going to be forwarded to the Crazyflie's
    position estimator.
    """
    quat = Rotation.from_euler("xyz", rot, degrees=True).as_quat()

    # cf.extpos.send_extpose(x, y, z, quat[0], quat[1], quat[2], quat[3])
    cf.extpos.send_extpos(x, y, z)

def build_trajectory(trajectories, space):
    # Load yaml file with space specification
    with open(space, "r") as f:
        space = yaml.full_load(f)
        home = space["home"]
        ranges = space["range"]

    # Account for height offset
    altitude = home["z"] + ranges["z"]
    side_length = min([ranges["x"], ranges["y"]]) * 2
    radius = min([ranges["x"], ranges["y"]])
    x_bound = [home["x"] - ranges["x"], home["x"] + ranges["x"]]
    y_bound = [home["y"] - ranges["y"], home["y"] + ranges["y"]]

    # Build trajectory
    # Takeoff
    setpoints = takeoff(home["x"], home["y"], altitude, 0.0)
    for trajectory in trajectories:
        # If nothing, only nothing
        if trajectory == "nothing":
            setpoints = None
            return setpoints
        elif trajectory == "hover":
            setpoints += hover(home["x"], home["y"], altitude)
        elif trajectory == "square":
            setpoints += square(home["x"], home["y"], side_length, altitude)
        elif trajectory == "octagon":
            setpoints += octagon(home["x"], home["y"], radius, altitude)
        elif trajectory == "triangle":
            setpoints += triangle(home["x"], home["y"], radius, altitude)
        elif trajectory == "hourglass":
            setpoints += hourglass(home["x"], home["y"], side_length, altitude)
        elif trajectory == "random":
            setpoints += randoms(home["x"], home["y"], x_bound, y_bound, altitude)
        elif trajectory == "scan":
            setpoints += scan(home["x"], home["y"], x_bound, y_bound, altitude)
        else:
            raise ValueError(f"{trajectory} is an unknown trajectory")

    # Add landing
    setpoints += landing(home["x"], home["y"], altitude, 0.0)

    return setpoints


def follow_setpoints(cf, setpoints, optitrack):
    # Counter for task dump logging
    time_since_dump = 0.0
    commander = cf.high_level_commander

    # Start
    try:
        print("Flight started")
        # Do nothing, just sit on the ground
        if setpoints is None:
            while True:
                time.sleep(0.05)
                time_since_dump += 0.05

                # Task dump
                if time_since_dump > 2:
                    print("Do task dump")
                    do_taskdump(cf)
                    time_since_dump = 0.0

        # Do actual flight
        else:
            wait_for_position_estimator(cf)
            # commanding take-off
            send_extpose_rot_matrix(cf, ot_position[0], ot_position[1], ot_position[2], ot_attitude)

            print("Commanding take-off")
            commander.takeoff(1.0, 2.0)

            # Send position and wait
            time_passed = 0.0
            while time_passed < 5.0:
                send_extpose_rot_matrix(cf, ot_position[0], ot_position[1], ot_position[2], ot_attitude)
                # cf.commander.send_position_setpoint(*point)
                time.sleep(0.05)
                time_passed += 0.05
                time_since_dump += 0.05

                # Task dump
                if time_since_dump > 2:
                    print("Do task dump")
                    do_taskdump(cf)
                    time_since_dump = 0.0
            
            # x, y, z, yaw, duration
            print("Commanding position")
            commander.go_to(0.5, 0.0, 1.0, 0.0, 2.0, relative=False)
            time_passed = 0.0
            while time_passed < 40.0:
                send_extpose_rot_matrix(cf, ot_position[0], ot_position[1], ot_position[2], ot_attitude)
                # cf.commander.send_position_setpoint(*point)
                time.sleep(0.05)
                time_passed += 0.05
                time_since_dump += 0.05

                # Task dump
                if time_since_dump > 2:
                    print("Do task dump")
                    do_taskdump(cf)
                    time_since_dump = 0.0
                        # x, y, z, yaw, duration
            print("Commanding position back")
            commander.go_to(0.0, 0.0, 1.0, 0.0, 2.0, relative=False)
            time_passed = 0.0
            while time_passed < 5.0:
                send_extpose_rot_matrix(cf, ot_position[0], ot_position[1], ot_position[2], ot_attitude)
                # cf.commander.send_position_setpoint(*point)
                time.sleep(0.05)
                time_passed += 0.05
                time_since_dump += 0.05

                # Task dump
                if time_since_dump > 2:
                    print("Do task dump")
                    do_taskdump(cf)
                    time_since_dump = 0.0
            # Finished
            print("Commanding land")
            commander.land(0.0, 2.0)
            time.sleep(2)
            commander.stop()


    # Prematurely break off flight / quit doing nothing
    except KeyboardInterrupt:
        if setpoints is None:
            print("Quit doing nothing!")
        else:
            print("Emergency landing!")
            commander.land(0.0, 2.0)
            time.sleep(2)
            commander.stop()


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--fileroot", type=str, required=True)
    parser.add_argument("--keywords", nargs="+", type=str.lower, default=None)
    parser.add_argument("--logconfig", type=str, required=True)
    parser.add_argument("--space", type=str, required=True)
    parser.add_argument(
        "--estimator",
        choices=["complementary", "kalman"],
        type=str.lower,
        required=True,
    )
    parser.add_argument(
        "--uwb", choices=["none", "twr", "tdoa"], type=str.lower, required=True
    )
    parser.add_argument("--flow", action="store_true")
    parser.add_argument("--trajectory", nargs="+", type=str.lower, required=True)
    parser.add_argument(
        "--optitrack",
        choices=["none", "logging", "state"],
        type=str.lower,
        default="none",
    )
    parser.add_argument("--optitrack_id", type=int, default=None)
    args = vars(parser.parse_args())

    # If no UWB, then OptiTrack
    # If no UWB and Flowdeck, then complementary
    if args["uwb"] == "none":
        assert args["optitrack"] == "state", "OptiTrack state needed in absence of UWB"
        # if not args["flow"]:
        #     assert (
        #         args["estimator"] == "complementary"
        #     ), "Absence of UWB and Flowdeck will lead Crazyflie to set estimator to 'complementary'"

    # Set up Crazyflie
    uri = "radio://0/80/2M/E7E7E7E7E7"
    cflib.crtp.init_drivers(enable_debug_driver=False)
    cf = Crazyflie(rw_cache="./cache")

    # Set up print connection to console
    # TODO: synchronize this with FileLogger: is this possible?
    console_log = []
    console = Console(cf)
    console.receivedChar.add_callback(console_cb)

    # Create directory if not there
    Path(args["fileroot"]).mkdir(exist_ok=True)

    # Set up logging
    flogger, file = setup_logger(
        cf,
        uri,
        args["fileroot"],
        args["keywords"],
        args["logconfig"],
        args["estimator"],
        args["uwb"],
        args["flow"],
        args["optitrack"],
        args["trajectory"],
    )

    # Check OptiTrack if it's there
    ot_id = args["optitrack_id"]
    ot_position, ot_attitude = setup_optitrack(args["optitrack"])
    # Wait for fix
    if ot_position is not None:
        while (ot_position == 0).any():

            print("Waiting for OptiTrack fix...")
            time.sleep(1)
        print(f'crazyflie position: {ot_position}')
        print("OptiTrack fix acquired")

    # Reset estimator
    activate_kalman_estimator(cf)
    activate_high_level_commander(cf)
    reset_estimator(cf, args["estimator"])
    # time.sleep(4)

    # Build trajectory
    setpoints = build_trajectory(args["trajectory"], args["space"])

    time.sleep(3)
    
    # Start onboard logging
    start_onboard_logging(cf)
    
    # Do flight
    follow_setpoints(cf, setpoints, args["optitrack"])


    # Stop the onboard logging
    stop_onboard_logging(cf)
    # End flight
    print("Done")
    time.sleep(2)
    cf.close_link()

    # Process task dumps
    # TODO: add timestamps / ticks (like logging) to this
    process_taskdump(file, console_log)

#!/usr/bin/env python
#coding:utf-8

# Version:         0.2.0
# Update data:     2021-10-08
# Updata contents: could plot pressure acc torque info
# Modified by:     Xiaosheng
# https://github.com/Tianxiaosheng/parse_ucdf

import tkinter as tk
from tkinter import *
import argparse
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
class Planner_Data(object):
    def __init__(self):
        self.frequency = 10
        self.offsets = []
        self.state = []
        self.state_full = []
        self.cmd = []
        self.cmd_planner = []
        self.steer_ang_vel = []
        self.size = 0
        return

    def parse_logfile(self, log_file):
        pattern = re.compile(r'.*offset to map')
        with open(log_file, 'r') as log:
                self.offsets = []
                line = log.readline()
                while line:
                    match = pattern.match(line)
                    if match:
                        line_split = line.split('>')
                        list_split = line_split[1].split(',')
                        data_split = list_split[0].split(':')
                        offset_to_map = float(data_split[1])
                        data_split = list_split[1].split(':')
                        offset_to_traj = float(data_split[1])
                        self.offsets.append([offset_to_map, offset_to_traj])
                    line = log.readline()
        self.offsets = np.array(self.offsets)

    def parse_canstate(self, log_file, filter_acc, filter_type = 1):
        pattern = re.compile(r'^can_state +')
        with open(log_file, 'r') as log:
                self.state = []
                line = log.readline()
                while line:
                    match = pattern.match(line)
                    if match:
                        line_split = line.split(': ')
                        data_split = line_split[1].split(' ')
                        vel = float(data_split[0])
                        steer = float(data_split[1])
                        self.state.append([vel, steer, 0.0])
                    line = log.readline()
        self.state = np.array(self.state)
        WINDOW = 5
        if len(self.state) < WINDOW + 1: return
        for i in range(len(self.state) - 1):
            if i >= len(self.state) - WINDOW:
                cur = self.state[i - 1]
                nex = self.state[i]
                nex[2] = cur[2]
            else:
                cur = self.state[i]
                nex = self.state[i + WINDOW]
                cur[2] = (nex[0] - cur[0]) * self.frequency / WINDOW
                nex[2] = cur[2]
        if not filter_acc: return
        if 2 == filter_type:
            b, a = signal.butter(3, 0.2)
            zi = signal.lfilter_zi(b, a)
            self.state[:, 2], _ = signal.lfilter(b, a, self.state[:, 2],
                    zi=zi*self.state[:, 2][0])
        elif 1 == filter_type:
            kernel_size = 5
            kernel = np.zeros(kernel_size)
            for i in range(len(kernel)):
                kernel[i] = 1.0 / kernel_size
            self.state[:, 2] = np.convolve(self.state[:, 2], kernel, 'same')

    def parse_canstate_full(self, log_file, filter_acc, filter_type = 1):
        pattern = re.compile(r'^can_state_full +')
        with open(log_file, 'r') as log:
                self.can_state_full = []
                line = log.readline()
                while line:
                    match = pattern.match(line)
                    if match:
                        line_split = line.split('vel: ')
                        line_split = line_split[1].split(')')
                        vel = float(line_split[0])

                        line_split = line.split('steer: ')
                        line_split = line_split[1].split(')')
                        steer = float(line_split[0])

                        line_split = line.split('brake_enabled: ')
                        line_split = line_split[1].split(')')
                        brake = float(line_split[0])

                        line_split = line.split('shift: ')
                        line_split = line_split[1].split(')')
                        shift = float(line_split[0])

                        line_split = line.split('control_source: ')
                        line_split = line_split[1].split(')')
                        control_source = float(line_split[0])

                        if line.count('pressure'):
                            line_split = line.split('pressure: ')
                            line_split = line_split[1].split(')')
                            pressure = float(line_split[0])
                        else:
                            pressure = float(0.0)

                        if line.count('torque') == 1:
                            line_split = line.split('torque: ')
                            line_split = line_split[1].split(')')
                            torque = float(line_split[0])
                        else:
                            torque = float(0.0)

                        acc = 0.0
                        self.can_state_full.append(
                                [vel, steer, acc, brake, shift, control_source, pressure, torque])
                    line = log.readline()
        self.can_state_full = np.array(self.can_state_full)
        WINDOW = 5
        if len(self.can_state_full) < WINDOW + 1: return
        for i in range(len(self.can_state_full) - 1):
            if i >= len(self.can_state_full) - WINDOW:
                cur = self.can_state_full[i - 1]
                nex = self.can_state_full[i]
                nex[2] = cur[2]
            else:
                cur = self.can_state_full[i]
                nex = self.can_state_full[i + WINDOW]
                cur[2] = (nex[0] - cur[0]) * self.frequency / WINDOW
                nex[2] = cur[2]
        if not filter_acc: return
        if 2 == filter_type:
            b, a = signal.butter(3, 0.2)
            zi = signal.lfilter_zi(b, a)
            self.can_state_full[:, 2], _ = signal.lfilter(b, a, self.can_state_full[:, 2],
                    zi=zi*self.can_state_full[:, 2][0])
        elif 1 == filter_type:
            kernel_size = 5
            kernel = np.zeros(kernel_size)
            for i in range(len(kernel)):
                kernel[i] = 1.0 / kernel_size
            self.can_state_full[:, 2] = np.convolve(self.can_state_full[:, 2], kernel, 'same')


    def parse_cancmd(self, log_file):
        pattern = re.compile(r'^can_cmd +')
        with open(log_file, 'r') as log:
                self.cmd = []
                line = log.readline()
                while line:
                    match = pattern.match(line)
                    if match:
                        line_split = line.split(' : ')
                        line_split = line_split[1].split(',')
                        data_split = line_split[0].split('vel: ')
                        data_split = data_split[1].split(')')
                        vel = float(data_split[0])
                        data_split = line_split[1].split('acc: ')
                        data_split = data_split[1].split(')')
                        acc = float(data_split[0])
                        data_split = line_split[2].split('steer: ')
                        data_split = data_split[1].split(')')
                        steer = float(data_split[0])
                        data_split = line_split[3].split('shift: ')
                        data_split = data_split[1].split(')')
                        shift = float(data_split[0])
                        data_split = line_split[4].split('throttle: ')
                        data_split = data_split[1].split(')')
                        throttle = float(data_split[0])
                        data_split = line_split[5].split('brake: ')
                        data_split = data_split[1].split(')')
                        brake = float(data_split[0])
                        data_split = line_split[8].split('estop: ')
                        data_split = data_split[1].split(')')
                        estop = float(data_split[0])
 
                        self.cmd.append(
                                [vel, steer, acc, brake, shift, estop, throttle])
                    line = log.readline()
        self.cmd = np.array(self.cmd)

    def parse_cancmd_planner(self, log_file):
        pattern = re.compile(r'^can_cmd_planner +')
        with open(log_file, 'r') as log:
                self.cmd_planner = []
                line = log.readline()
                while line:
                    match = pattern.match(line)
                    if match:
                        line_split = line.split(' : ')
                        line_split = line_split[1].split(',')
                        data_split = line_split[0].split('vel: ')
                        data_split = data_split[1].split(')')
                        vel = float(data_split[0])
                        data_split = line_split[1].split('acc: ')
                        data_split = data_split[1].split(')')
                        acc = float(data_split[0])
                        data_split = line_split[2].split('steer: ')
                        data_split = data_split[1].split(')')
                        steer = float(data_split[0])
                        data_split = line_split[5].split('brake: ')
                        data_split = data_split[1].split(')')
                        brake = float(data_split[0])
                        data_split = line_split[3].split('shift: ')
                        data_split = data_split[1].split(')')
                        shift = float(data_split[0])

                        self.cmd_planner.append([vel, steer, acc, brake, shift])
                    line = log.readline()
        self.cmd_planner = np.array(self.cmd_planner)

    def calc_steer_ang_vel(self):
        self.size = min(len(self.cmd),
                len(self.can_state_full))

        self.steer_ang_vel = np.zeros((self.size, 2))
        for i in range(self.size - 3):
            self.steer_ang_vel[i, 0] = (self.cmd[i + 3, 1] - self.cmd[i, 1]) * self.frequency / 3.0
            self.steer_ang_vel[i, 1] = (self.can_state_full[i + 3, 1] - self.can_state_full[i, 1]) * self.frequency / 3.0

class Stg(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("parse_ucdf")
        self.planner_data = Planner_Data()
        self.parser_options()
        self.planner_data.parse_canstate_full(self.planner_opts.log_file, True ,2)
        self.planner_data.parse_cancmd(self.planner_opts.log_file)
        self.planner_data.parse_cancmd_planner(self.planner_opts.log_file)
        self.planner_data.calc_steer_ang_vel()

        self.createWdidgets()

    def createWdidgets(self):
        ''' 界面 '''
        footframe = tk.Frame(self)
        footframe['borderwidth'] = 12
        footframe['relief'] = 'raised'
        footframe.grid(column=0, row=0, sticky=(N, W, E, S))

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        footframe.columnconfigure(0, weight=5)
        footframe.rowconfigure(0, weight=5)

        fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = fig.add_subplot(111)
        self.twinx = self.ax.twinx()
        self.canvas = FigureCanvasTkAgg(fig, footframe)
        self.canvas.get_tk_widget().pack(fill="both", expand=1)

        toolbar = NavigationToolbar2Tk(self.canvas, footframe)
        toolbar.update()

        footframe2 = tk.Frame(self)
        footframe2.grid(row=0,column=1, sticky=(N, W, E, S))
        footframe2.columnconfigure(1, weight=1)

        labelAcc = tk.LabelFrame(footframe2, text='Acc')
        labelVel = tk.LabelFrame(footframe2, text='Vel')
        labelShift = tk.LabelFrame(footframe2, text='Shift')
        labelSteer = tk.LabelFrame(footframe2, text='Steer')
        labelState = tk.LabelFrame(footframe2, text='State')

        labelAcc.grid(column=0, row=0, padx=8, pady=4, sticky='EW')
        labelVel.grid(column=0, row=1, padx=8, pady=4, sticky='EW')
        labelShift.grid(column=0, row=2, padx=8, pady=4, sticky='EW')
        labelSteer.grid(column=0, row=3, padx=8, pady=4, sticky='EW')
        labelState.grid(column=0, row=4, padx=8, pady=4, sticky='EW')

        if len(self.planner_data.cmd) != len(self.planner_data.can_state_full):
            print("Error, Some data lost")
        self.size = min(len(self.planner_data.cmd),
                len(self.planner_data.can_state_full))
        self.frame = np.arange(0, self.size)

        # 1. set check_value
        self.checkVar_expt_acc_planner = IntVar()  # acc
        self.checkVar_expt_acc = IntVar()
        self.checkVar_veh_acc = IntVar()
        self.checkVar_expt_throttle = IntVar()
        self.checkVar_veh_torque = IntVar()
        self.checkVar_veh_pressure = IntVar()

        self.checkVar_expt_vel_planner = IntVar()  # vel
        self.checkVar_expt_vel = IntVar()
        self.checkVar_veh_vel = IntVar()

        self.checkVar_expt_shift_planner = IntVar()  # shift
        self.checkVar_expt_shift = IntVar()
        self.checkVar_veh_shift = IntVar()

        self.checkVar_expt_brake = IntVar()  # brake

        self.checkVar_expt_steer_ang = IntVar()  # steer_ang
        self.checkVar_veh_steer_ang = IntVar()

        self.checkVar_expt_steer_ang_vel = IntVar()  # steer_ang_vel
        self.checkVar_veh_steer_ang_vel = IntVar()

        self.checkVar_expt_estop = IntVar()  # Status
        self.checkVar_veh_control_source = IntVar()

        # 2. place checkbutton
        checkbutton_expt_acc_planner = tk.Checkbutton(labelAcc,  # acc
                text='Expt_Acc_Planner',
                variable = self.checkVar_expt_acc_planner,
                onvalue = 1, offvalue = 0,
                command=self.printf_info)
        checkbutton_expt_acc_planner.grid(row=0, column=0, sticky=(W))

        checkbutton_expt_acc = tk.Checkbutton(labelAcc, text='Expt_Acc',
                variable = self.checkVar_expt_acc, onvalue = 1, offvalue = 0,
                command=self.printf_info)
        checkbutton_expt_acc.grid(row=1, column=0, sticky=(W))

        checkbutton_veh_acc = tk.Checkbutton(labelAcc, text='Veh_Acc',
                variable = self.checkVar_veh_acc, onvalue = 1, offvalue = 0,
                command=self.printf_info)
        checkbutton_veh_acc.grid(row=2, column=0, sticky=(W))

        checkbutton_expt_throttle = tk.Checkbutton(labelAcc, text='Expt_Throttle',
                variable = self.checkVar_expt_throttle, onvalue = 1, offvalue = 0,
                command=self.printf_info)
        checkbutton_expt_throttle.grid(row=3, column=0, sticky=(W))

        checkbutton_expt_brake = tk.Checkbutton(labelAcc, text='Expt_Brake',  # brake
                variable = self.checkVar_expt_brake, onvalue = 1, offvalue = 0,
                command=self.printf_info)
        checkbutton_expt_brake.grid(row=4, column=0, sticky=(W))

        checkbutton_veh_torque = tk.Checkbutton(labelAcc, text='Veh_Torque',
                variable = self.checkVar_veh_torque, onvalue = 1, offvalue = 0,
                command=self.printf_info)
        checkbutton_veh_torque.grid(row=5, column=0, sticky=(W))

        checkbutton_veh_pressure = tk.Checkbutton(labelAcc, text='Veh_Pressure',
                variable = self.checkVar_veh_pressure, onvalue = 1, offvalue = 0,
                command=self.printf_info)
        checkbutton_veh_pressure.grid(row=6, column=0, sticky=(W))

        checkbutton_expt_vel_planner = tk.Checkbutton(labelVel, text='Expt_Vel_Planner',  # vel
                variable = self.checkVar_expt_vel_planner, onvalue = 1, offvalue = 0,
                command=self.printf_info)
        checkbutton_expt_vel_planner.grid(row=0, column=0, sticky=(W))

        checkbutton_expt_vel = tk.Checkbutton(labelVel, text='Expt_Vel',
                variable = self.checkVar_expt_vel, onvalue = 1, offvalue = 0,
                command=self.printf_info)
        checkbutton_expt_vel.grid(row=1, column=0, sticky=(W))

        checkbutton_veh_vel = tk.Checkbutton(labelVel, text='Veh_Vel',
                variable = self.checkVar_veh_vel, onvalue = 1, offvalue = 0,
                command=self.printf_info)
        checkbutton_veh_vel.grid(row=2, column=0, sticky=(W))

        checkbutton_expt_shift_planner = tk.Checkbutton(labelShift, text='Expt_Shift_Planner',  # shift
                variable = self.checkVar_expt_shift_planner, onvalue = 1, offvalue = 0,
                command=self.printf_info)
        checkbutton_expt_shift_planner.grid(row=0, column=0, sticky=(W))

        checkbutton_expt_shift = tk.Checkbutton(labelShift, text='Expt_Shift',
                variable = self.checkVar_expt_shift, onvalue = 1, offvalue = 0,
                command=self.printf_info)
        checkbutton_expt_shift.grid(row=1, column=0, sticky=(W))

        checkbutton_veh_shift = tk.Checkbutton(labelShift, text='Veh_Shift',
                variable = self.checkVar_veh_shift, onvalue = 1, offvalue = 0,
                command=self.printf_info)
        checkbutton_veh_shift.grid(row=2, column=0, sticky=(W))

        checkbutton_expt_steer_ang = tk.Checkbutton(labelSteer, text='Expt_Steer_Ang',  #steer_ang
                variable = self.checkVar_expt_steer_ang, onvalue = 1, offvalue = 0,
                command=self.printf_info)
        checkbutton_expt_steer_ang.grid(row=0, column=0, sticky=(W))

        checkbutton_veh_steer_ang = tk.Checkbutton(labelSteer, text='Veh_Steer_Ang',
                variable = self.checkVar_veh_steer_ang, onvalue = 1, offvalue = 0,
                command=self.printf_info)
        checkbutton_veh_steer_ang.grid(row=1, column=0, sticky=(W))

        checkbutton_expt_steer_ang_vel = tk.Checkbutton(labelSteer, text='Expt_Steer_Ang_Vel',
                variable = self.checkVar_expt_steer_ang_vel, onvalue = 1, offvalue = 0,
                command=self.printf_info)
        checkbutton_expt_steer_ang_vel.grid(row=2, column=0, sticky=(W))

        checkbutton_veh_steer_ang_vel = tk.Checkbutton(labelSteer, text='Veh_Steer_Ang_Vel',
                variable = self.checkVar_veh_steer_ang_vel, onvalue = 1, offvalue = 0,
                command=self.printf_info)
        checkbutton_veh_steer_ang_vel.grid(row=3, column=0, sticky=(W))

        checkbutton_expt_estop = tk.Checkbutton(labelState, text='Expt_Estop',  # status
                variable = self.checkVar_expt_estop, onvalue = 1, offvalue = 0,
                command=self.printf_info)
        checkbutton_expt_estop.grid(row=0, column=0, sticky=(W))

        checkbutton_veh_control_source = tk.Checkbutton(labelState, text='Veh_Control_Source',
                variable = self.checkVar_veh_control_source, onvalue = 1, offvalue = 0,
                command=self.printf_info)
        checkbutton_veh_control_source.grid(row=1, column=0, sticky=(W))

        # 3. set checkVar's init value
        self.checkVar_expt_acc_planner.set(1)
        self.checkVar_expt_acc.set(1)
        self.checkVar_veh_acc.set(1)
        self.checkVar_expt_throttle.set(1)
        self.checkVar_expt_brake.set(1)
        self.checkVar_veh_torque.set(1)
        self.checkVar_veh_pressure.set(1)

        self.checkVar_expt_vel_planner.set(0)
        self.checkVar_expt_vel.set(0)
        self.checkVar_veh_vel.set(0)

        self.checkVar_expt_shift_planner.set(0)
        self.checkVar_expt_shift.set(0)
        self.checkVar_veh_shift.set(0)

        self.checkVar_expt_steer_ang.set(0)
        self.checkVar_veh_steer_ang.set(0)

        self.checkVar_expt_steer_ang_vel.set(0)
        self.checkVar_veh_steer_ang_vel.set(0)

        self.checkVar_expt_estop.set(0)
        self.checkVar_veh_control_source.set(0)

        self.printf_info();

    def parser_options(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--log', action='store', dest='log_file' \
                            , default="None", help='plot planner profile gui v1.0')

        self.planner_opts = parser.parse_args()

    def printf_info(self):

        #clear prev plot result
        self.ax.clear()
        self.twinx.clear()

        # 4. plot data
        if self.checkVar_expt_acc_planner.get() == 1:  # acc
           self.ax.plot(self.frame, self.planner_data.cmd_planner[0:self.size, 2], 'r',
                   label="expt_acc_planner")

        if self.checkVar_expt_acc.get() == 1:
            self.ax.plot(self.frame, self.planner_data.cmd[0:self.size, 2], 'r--',
                   label="expt_acc")

        if self.checkVar_veh_acc.get() == 1:
           self.ax.plot(self.frame, self.planner_data.can_state_full[0:self.size, 2],
                   'g', label="veh_acc")

        if self.checkVar_expt_throttle.get() == 1:  # expt throttle
           self.twinx.plot(self.frame, self.planner_data.cmd[0:self.size, 3],
                   'darkviolet', label="expt_throttle")

        if self.checkVar_expt_brake.get() == 1:  # expt brake
           self.twinx.plot(self.frame, self.planner_data.cmd[0:self.size, 3],
                   'k', label="expt_brake")

        if self.checkVar_veh_torque.get() == 1:  # brake pressure
           self.twinx.plot(self.frame, self.planner_data.can_state_full[0:self.size, 7],
                   'greenyellow', label="veh_torque")

        if self.checkVar_veh_pressure.get() == 1:  # brake pressure
           self.twinx.plot(self.frame, self.planner_data.can_state_full[0:self.size, 6],
                   'k--', label="veh_brake_pressure")

        if self.checkVar_expt_vel_planner.get() == 1:  # vel
            self.ax.plot(self.frame, self.planner_data.cmd_planner[0:self.size, 0], 'y',
                   label="expt_vel_planner")

        if self.checkVar_expt_vel.get() == 1:
            self.ax.plot(self.frame, self.planner_data.cmd[0:self.size, 0], 'y--',
                   label="expt_vel")

        if self.checkVar_veh_vel.get() == 1:
           self.ax.plot(self.frame, self.planner_data.can_state_full[0:self.size, 0],
                   'b', label="veh_vel")

        if self.checkVar_expt_shift_planner.get() == 1:  # shift
           self.ax.plot(self.frame, self.planner_data.cmd_planner[0:self.size, 4],
                   'tab:pink', label="expt_shift_planner 1:P,2:R,3:N,4:D")

        if self.checkVar_expt_shift.get() == 1:
           self.ax.plot(self.frame, self.planner_data.cmd[0:self.size, 4],
                   'tab:purple', label="expt_shift 1:P,2:R,3:N,4:D")

        if self.checkVar_veh_shift.get() == 1:
           self.ax.plot(self.frame, self.planner_data.can_state_full[0:self.size, 4],
                   'tab:blue', label="veh_shift 1:P,2:R,3:N,4:D")

        if self.checkVar_expt_steer_ang.get() == 1:  # steer
           self.ax.plot(self.frame, self.planner_data.cmd[0:self.size, 1],
                   'm', label="expt_steer_ang")

        if self.checkVar_veh_steer_ang.get() == 1:
           self.ax.plot(self.frame, self.planner_data.can_state_full[0:self.size, 1],
                   'c', label="veh_steer_ang")

        if self.checkVar_expt_steer_ang_vel.get() == 1:
           self.ax.plot(self.frame, self.planner_data.steer_ang_vel[0:self.size, 0],
                   'm--', label="expt_steer_ang_vel")

        if self.checkVar_veh_steer_ang_vel.get() == 1:
           self.ax.plot(self.frame, self.planner_data.steer_ang_vel[0:self.size, 1],
                   'c--', label="veh_steer_ang_vel")

        if self.checkVar_expt_estop.get() == 1:  # estop
           self.twinx.plot(self.frame, self.planner_data.cmd[0:self.size, 5],
                   'tab:green', label="expt_estop")

        if self.checkVar_veh_control_source.get() == 1:  # control
           self.twinx.plot(self.frame, self.planner_data.can_state_full[0:self.size, 5],
                   'tab:gray', label="veh_control_source 0:Auto, 3:Manual")

        self.ax.legend(loc="best")
        self.twinx.legend(loc="best")
        self.ax.grid(True)
        self.ax.set_xlabel("frame(100ms)")
        self.ax.set_ylabel("Vel(m/s) Acc(m/ss) Shift Steer(.)")
        self.twinx.set_ylabel("Throttle Brake Torque Pressure Estop Control_source")
        self.ax.set_title("ucdf_plot")
        self.canvas.draw()
    def _quit(self):
        ''' 退出 '''
        self.quit()
        self.destroy()

app = Stg()
app.mainloop()


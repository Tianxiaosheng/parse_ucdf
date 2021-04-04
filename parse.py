#!/usr/bin/env python3
#coding:utf-8
import tkinter as tk
from tkinter import *
import argparse
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
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
                        line_split = line.split(' : ')
                        line_split = line_split[1].split(',')
                        data_split = line_split[0].split('vel: ')
                        data_split = data_split[1].split(')')
                        vel = float(data_split[0])
                        data_split = line_split[1].split('steer: ')
                        data_split = data_split[1].split(')')
                        steer = float(data_split[0])
                        data_split = line_split[5].split('brake_enabled: ')
                        data_split = data_split[1].split(')')
                        brake = float(data_split[0])
                        acc = 0.0
                        self.can_state_full.append([vel, steer, acc, brake])
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
                        data_split = line_split[5].split('brake: ')
                        data_split = data_split[1].split(')')
                        brake = float(data_split[0])
                        self.cmd.append([vel, steer, acc, brake])
                    line = log.readline()
        self.cmd = np.array(self.cmd)



class Stg(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("St_graph")
        self.planner_data = Planner_Data()
        self.parser_options()
        self.planner_data.parse_canstate_full(self.planner_opts.log_file, True ,2)
        self.planner_data.parse_cancmd(self.planner_opts.log_file)
        self.createWdidgets()

    def createWdidgets(self):
        ''' 界面 '''
        footframe = tk.Frame(self)
        footframe['borderwidth'] = 12
        footframe['relief'] = 'raised'
        footframe.grid(column=0, row=0, sticky=(N, W, E, S))

        fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, footframe)
        self.canvas.get_tk_widget().grid(column=0,row=0,sticky=(E, W))

        if len(self.planner_data.cmd) != len(self.planner_data.can_state_full):
            print("Error, Some data lost")
        size = min(len(self.planner_data.cmd), len(self.planner_data.can_state_full))
        frame = np.arange(0, size)

        self.ax.plot(frame, self.planner_data.cmd[0:size, 0], 'r')
        self.ax.plot(frame, self.planner_data.can_state_full[0:size, 0], 'g')
        self.canvas.draw()

        button1 = tk.Button(footframe, text='重画', command=self.printf_info)
        button1.grid(column=1, row=1, sticky=(E, W))
        button2 = tk.Button(footframe, text='退出', command=self._quit)
        button2.grid(column=2, row=2, sticky=(E, W))

        self.checkVar = IntVar()
        checkbutton = tk.Checkbutton(footframe, text='checkbutton', variable = self.checkVar, onvalue = 1, offvalue = 0, command=self.printf_info)
        checkbutton.grid(column=3, row=3, sticky=(E, W))


    def parser_options(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--log', action='store', dest='log_file' \
                            , default="None", help='planner log file')

        self.planner_opts = parser.parse_args()

    def printf_info(self):
        print(self.checkVar.get())
    def _quit(self):
        ''' 退出 '''
        self.quit()
        self.destroy()

app = Stg()
app.mainloop()


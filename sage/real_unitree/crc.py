# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import ctypes
import os
import platform
import struct

import cyclonedds.idl as idl
from unitree_hg.msg import LowCmd as HGLowCmd_
from unitree_hg.msg import LowState as HGLowState_


class Singleton:
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super(Singleton, cls).__new__(cls)
        return cls.__instance

    def __init__(self):
        pass


class CRC(Singleton):
    def __init__(self):
        # 4 bytes aligned, little-endian format.
        # size 812
        self.__packFmtLowCmd = "<4B4IH2x" + "B3x5f3I" * 20 + "4B" + "55Bx2I"
        # size 1180
        self.__packFmtLowState = "<4B4IH2x" + "13fb3x" + "B3x7fb3x3I" * 20 + "4BiH4b15H" + "8hI41B3xf2b2x2f4h2I"
        # size 1004
        self.__packFmtHGLowCmd = "<2B2x" + "B3x5fI" * 35 + "5I"
        # size 2092
        self.__packFmtHGLowState = "<2I2B2xI" + "13fh2x" + "B3x4f2hf7I" * 35 + "40B5I"

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.platform = platform.system()
        if self.platform == "Linux":
            if platform.machine() == "x86_64":
                self.crc_lib = ctypes.CDLL(script_dir + "/lib/crc_amd64.so")
            elif platform.machine() == "aarch64":
                self.crc_lib = ctypes.CDLL(script_dir + "/lib/crc_aarch64.so")

            self.crc_lib.crc32_core.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32)
            self.crc_lib.crc32_core.restype = ctypes.c_uint32

    def Crc(self, msg: idl.IdlStruct):
        return self.__Crc32(self.__PackHGLowCmd(msg))

    def __PackHGLowCmd(self, cmd: HGLowCmd_):
        origData = []
        origData.append(cmd.mode_pr)
        origData.append(cmd.mode_machine)

        for i in range(35):
            origData.append(cmd.motor_cmd[i].mode)
            origData.append(cmd.motor_cmd[i].q)
            origData.append(cmd.motor_cmd[i].dq)
            origData.append(cmd.motor_cmd[i].tau)
            origData.append(cmd.motor_cmd[i].kp)
            origData.append(cmd.motor_cmd[i].kd)
            origData.append(cmd.motor_cmd[i].reserve)

        origData.extend(cmd.reserve)
        origData.append(cmd.crc)

        return self.__Trans(struct.pack(self.__packFmtHGLowCmd, *origData))

    def __PackHGLowState(self, state: HGLowState_):
        origData = []
        origData.extend(state.version)
        origData.append(state.mode_pr)
        origData.append(state.mode_machine)
        origData.append(state.tick)

        origData.extend(state.imu_state.quaternion)
        origData.extend(state.imu_state.gyroscope)
        origData.extend(state.imu_state.accelerometer)
        origData.extend(state.imu_state.rpy)
        origData.append(state.imu_state.temperature)

        for i in range(35):
            origData.append(state.motor_state[i].mode)
            origData.append(state.motor_state[i].q)
            origData.append(state.motor_state[i].dq)
            origData.append(state.motor_state[i].ddq)
            origData.append(state.motor_state[i].tau_est)
            origData.extend(state.motor_state[i].temperature)
            origData.append(state.motor_state[i].vol)
            origData.extend(state.motor_state[i].sensor)
            origData.append(state.motor_state[i].motorstate)
            origData.extend(state.motor_state[i].reserve)

        origData.extend(state.wireless_remote)
        origData.extend(state.reserve)
        origData.append(state.crc)

        return self.__Trans(struct.pack(self.__packFmtHGLowState, *origData))

    def __Trans(self, packData):
        calcData = []
        calcLen = (len(packData) >> 2) - 1

        for i in range(calcLen):
            d = (
                (packData[i * 4 + 3] << 24)
                | (packData[i * 4 + 2] << 16)
                | (packData[i * 4 + 1] << 8)
                | (packData[i * 4])
            )
            calcData.append(d)

        return calcData

    def _crc_py(self, data):
        bit = 0
        crc = 0xFFFFFFFF
        polynomial = 0x04C11DB7

        for i in range(len(data)):
            bit = 1 << 31
            current = data[i]

            for b in range(32):
                if crc & 0x80000000:
                    crc = (crc << 1) & 0xFFFFFFFF
                    crc ^= polynomial
                else:
                    crc = (crc << 1) & 0xFFFFFFFF

                if current & bit:
                    crc ^= polynomial

                bit >>= 1

        return crc

    def _crc_ctypes(self, data):
        uint32_array = (ctypes.c_uint32 * len(data))(*data)
        length = len(data)
        crc = self.crc_lib.crc32_core(uint32_array, length)
        return crc

    def __Crc32(self, data):
        if self.platform == "Linux":
            return self._crc_ctypes(data)
        else:
            return self._crc_py(data)

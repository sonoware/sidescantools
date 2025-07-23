import numpy as np
from xtf_wrapper import XTFWrapper
import pyxtf
from jsf import JSFFile, JSFSystemInformation, JSFSonarDataPacket
import os
from pathlib import Path
from datetime import datetime


class SidescanFile:
    """Wrapper to read sidescan files
    """
    filepath: Path
    """Path to read file"""
    choose_subsys: int
    """Use this argument if u are reading a JSF File that contains multiple subsystems 
        to choose the sub system which shall be evaluated:

        - 0: First sub system (usually LF if present, otherwise HF)
        - 1: Second sub system (usually HF if LF and HF are present)
        - 2: Custom/additional sub systems
    """
    ping_len: int
    """Length of a ping in sample number"""
    num_ping: int
    """Length of a ping in sample number"""
    num_ch: int
    """Number of channels"""
    subsys_num: int
    """Number of Subsystems"""
    data: np.ndarray
    """raw sidescan data - [channel index, ping index, sample index]"""
    subsys_names: list
    """Names of subsystem (in case of JSF)"""
    sound_velocity: float
    """SOS/2 in water"""
    ping_x_axis: np.ndarray
    """X-Axis in m for a single ping (slant direction)"""
    timestamp: list
    """Time of each ping"""
    starting_depth: np.ndarray
    """Starting Depth (window offset) in Samples"""
    longitude: np.ndarray
    """Longitude in degree
    """
    latitude: np.ndarray
    """Latitude in degree
    """
    coord_units: int
    gain_adc: np.ndarray
    """Gain Factor of ADC"""
    depth: np.ndarray
    """Depth in Meters (Below Water Surface)"""
    packet_no: np.ndarray
    """Packet Number
    """
    # TODO: validate in XTF doc
    sensor_heading: np.ndarray
    """ in degree (?)"""
    sensor_pitch: np.ndarray
    """ in degree (?)"""
    sensor_primary_altitude: np.ndarray
    """ in m (?)"""
    sensor_roll: np.ndarray
    """ in degree (?)"""
    sensor_speed: np.ndarray
    """ in m/s (?)"""
    sensor_aux_altitude: np.ndarray
    """ in degree (?)"""

    seconds_per_ping: np.ndarray
    """in s per channel"""
    slant_range: np.ndarray
    """in m per channel"""

    def __init__(self, filepath: str | os.PathLike, choose_subsys=0):
        self.filepath = Path(filepath)
        self.choose_subsys = choose_subsys

        # check whether this is XTF/JSF/not supported
        if self.filepath.suffix.casefold() == ".xtf":
            xtf = XTFWrapper(file_path=self.filepath)

            # set data fields
            self.ping_len = xtf.num_sample_per_ping
            self.num_ch = xtf.num_ch
            self.data = xtf.sonar_data
            self.num_ping = np.shape(xtf.sonar_data)[-2]

            self.sound_velocity = np.empty(self.num_ping)
            self.timestamp = [datetime] * self.num_ping
            self.sensor_heading = np.empty(self.num_ping)
            self.sensor_pitch = np.empty(self.num_ping)
            self.sensor_aux_altitude = np.empty(self.num_ping)
            self.sensor_primary_altitude = np.empty(self.num_ping)
            self.sensor_roll = np.empty(self.num_ping)
            self.sensor_speed = np.empty(self.num_ping)
            self.longitude = np.empty(self.num_ping)
            self.latitude = np.empty(self.num_ping)
            self.depth = np.empty(self.num_ping)
            self.packet_no = np.empty(self.num_ping)
            self.seconds_per_ping = np.empty((self.num_ch, self.num_ping))
            self.slant_range = np.empty((self.num_ch, self.num_ping))

            # not included
            self.starting_depth = np.zeros(self.num_ping)
            self.gain_adc = np.zeros(self.num_ping)
            
            # read XTF info
            for p_idx in range(self.num_ping):
                self.sound_velocity[p_idx] = xtf.packets[pyxtf.XTFHeaderType.sonar][p_idx].SoundVelocity * 2 # TODO: this is half SOS in water, right?
                self.timestamp[p_idx] = datetime(
                    xtf.packets[pyxtf.XTFHeaderType.sonar][p_idx].Year,
                    xtf.packets[pyxtf.XTFHeaderType.sonar][p_idx].Month,
                    xtf.packets[pyxtf.XTFHeaderType.sonar][p_idx].Day,
                    xtf.packets[pyxtf.XTFHeaderType.sonar][p_idx].Hour,
                    xtf.packets[pyxtf.XTFHeaderType.sonar][p_idx].Minute,
                    xtf.packets[pyxtf.XTFHeaderType.sonar][p_idx].Second)
                self.sensor_heading[p_idx] = xtf.packets[pyxtf.XTFHeaderType.sonar][p_idx].SensorHeading
                self.sensor_pitch[p_idx] = xtf.packets[pyxtf.XTFHeaderType.sonar][p_idx].SensorPitch
                self.sensor_primary_altitude[p_idx] = xtf.packets[pyxtf.XTFHeaderType.sonar][p_idx].SensorPrimaryAltitude
                self.sensor_roll[p_idx] = xtf.packets[pyxtf.XTFHeaderType.sonar][p_idx].SensorRoll
                self.sensor_speed[p_idx] = xtf.packets[pyxtf.XTFHeaderType.sonar][p_idx].SensorSpeed
                self.sensor_aux_altitude[p_idx] = xtf.packets[pyxtf.XTFHeaderType.sonar][p_idx].SensorAuxAltitude
                self.longitude[p_idx] = xtf.packets[pyxtf.XTFHeaderType.sonar][p_idx].SensorXcoordinate
                self.latitude[p_idx] = xtf.packets[pyxtf.XTFHeaderType.sonar][p_idx].SensorYcoordinate
                self.depth[p_idx] = xtf.packets[pyxtf.XTFHeaderType.sonar][p_idx].SensorDepth
                self.packet_no[p_idx]  = xtf.packets[pyxtf.XTFHeaderType.sonar][p_idx].PingNumber
                for ch_idx in range(self.num_ch):
                    if len(xtf.packets[pyxtf.XTFHeaderType.sonar][p_idx].ping_chan_headers) == self.num_ch:
                        self.seconds_per_ping[ch_idx, p_idx] = xtf.packets[pyxtf.XTFHeaderType.sonar][p_idx].ping_chan_headers[ch_idx].SecondsPerPing
                        self.slant_range[ch_idx, p_idx] = xtf.packets[pyxtf.XTFHeaderType.sonar][p_idx].ping_chan_headers[ch_idx].SlantRange
                    else:
                        self.seconds_per_ping[ch_idx, p_idx] = xtf.packets[pyxtf.XTFHeaderType.sonar][p_idx].ping_chan_headers[0].SecondsPerPing
                        self.slant_range[ch_idx, p_idx] = xtf.packets[pyxtf.XTFHeaderType.sonar][p_idx].ping_chan_headers[0].SlantRange

            self.ping_x_axis = xtf.x_axis_m


        elif self.filepath.suffix.casefold() == ".jsf":
            jsf_file = JSFFile(self.filepath)

            # find number and names of subsystems in the present data
            self.subsys_names = []
            self.subsys_num = 0
            self.num_ch = 0
            for packet in jsf_file.packets:
                if type(packet) == JSFSystemInformation:
                    self.subsys_num = packet.num_subsystems
                if type(packet) == JSFSonarDataPacket:
                    if packet.header.subsys_no not in self.subsys_names:
                        self.subsys_names.append(packet.header.subsys_no)

            if len(self.subsys_names) != self.subsys_num:
                print(f"Mismatch on subsys names: {self.subsys_names} and expected num: {self.subsys_num}")
                self.subsys_num = len(self.subsys_names)
            
            data_port = []
            data_star = []
            expected_idx = np.zeros(2, dtype=int)

            for packet in jsf_file.packets:
                if type(packet) == JSFSonarDataPacket:

                    # dual channel data of desired sub system
                    if packet.header.subsys_no == self.subsys_names[self.choose_subsys]:
                        if packet.header.channel >= self.num_ch:
                            self.num_ch = packet.header.channel + 1

                        # check ping number
                        if expected_idx[packet.header.channel] != packet.message.ping_no:
                            print(f"Expected ping mismatch: Subsys: {packet.header.subsys_no}, Channel: {packet.header.channel}, Ping number: {packet.message.ping_no} - expected ping: {expected_idx[packet.header.channel]}")
                            expected_idx[packet.header.channel] = packet.message.ping_no + 1
                        else:
                            expected_idx[packet.header.channel] += 1

                        # portside
                        if packet.header.channel == 0:
                            data_port.append(packet.message.data)
                        
                        # starboard
                        elif packet.header.channel == 1:
                            data_star.append(packet.message.data)

            # TODO: Desired behavior or implement other strategy?
            if self.num_ch != 2:
                print(f"Expected 2 channels, but found {self.num_ch} channels")
                raise NotImplementedError

            # Merge the data
            self.data = np.array((np.fliplr(data_port), data_star))

            # set data fields
            self.ping_len = self.data.shape[-1]
            self.num_ping = self.data.shape[-2]

            self.sound_velocity = np.empty(self.num_ping)
            self.timestamp = [datetime] * self.num_ping
            self.sensor_heading = np.empty(self.num_ping)
            self.sensor_pitch = np.empty(self.num_ping)
            self.sensor_primary_altitude = np.empty(self.num_ping)
            self.sensor_roll = np.empty(self.num_ping)
            self.sensor_speed = np.empty(self.num_ping)
            self.longitude = np.empty(self.num_ping)
            self.latitude = np.empty(self.num_ping)
            self.depth = np.empty(self.num_ping)
            self.packet_no = np.empty(self.num_ping)
            self.seconds_per_ping = np.empty((self.num_ch, self.num_ping))
            self.slant_range = np.empty((self.num_ch, self.num_ping))

            # not included
            self.sensor_aux_altitude = np.zeros(self.num_ping)

            self.starting_depth = np.zeros(self.num_ping)
            self.gain_adc = np.zeros(self.num_ping)

            # walk through all packets and take meta info from channel 0
            p_idx = 0

            # TODO: check if all values are interpreted correctly
            for packet in jsf_file.packets:
                if type(packet) == JSFSonarDataPacket:
                    if packet.header.subsys_no == self.subsys_names[self.choose_subsys] and packet.header.channel == 0:
                        
                        self.sound_velocity[p_idx] = packet.message.SOS
                        self.timestamp[p_idx] = datetime.fromtimestamp(packet.message.time)
                        self.sensor_heading[p_idx] = packet.message.compass_heading / 100
                        self.sensor_pitch[p_idx] = packet.message.pitch / 32768 * 180
                        self.sensor_primary_altitude[p_idx] = packet.message.altitude / 1e3
                        self.sensor_roll[p_idx] = packet.message.roll / 32768 * 180
                        self.sensor_speed[p_idx] = packet.message.speed / 10 / 1.944
                        
                        self.depth[p_idx] = packet.message.depth / 1e3
                        self.packet_no[p_idx] = packet.message.packet_no
                        self.seconds_per_ping[:, p_idx] = packet.message.sampling_interval_ns / 1e9
                        # check whether SOS is valid
                        sos = packet.message.SOS
                        if sos == 0:
                            sos = 1500
                        # round to 2 decimals here to avoid potential resampling of data in slant range correction
                        self.slant_range[:, p_idx] = np.round(self.seconds_per_ping[0, p_idx] * self.ping_len * sos / 2, decimals=2)
                        self.starting_depth[p_idx] = packet.message.starting_depth
                        self.gain_adc[p_idx] = packet.message.gain_adc

                        if packet.message.coord_units == 2:
                            # convert Latitude, longitude in minutes of arc times 10000 to degree
                            self.longitude[p_idx] = packet.message.longitude / 10000 / 60
                            self.latitude[p_idx] = packet.message.latitude / 10000 / 60
                        else:
                            # TODO: implement other cases:
                            # 1 = X, Y in millimeters
                            # 2 = Latitude, longitude in minutes of arc times 10000
                            # 3 = X, Y in decimeters
                            # 4 = X, Y in centimeters
                            raise NotImplementedError
                        
                        p_idx += 1

            self.ping_x_axis = np.linspace(0, self.slant_range[0][0], self.ping_len)

        else:
            print(f"File type {self.filepath.suffix.casefold()} isn't supported.")
            # TODO: Byte-weise nach nächstem passenden sync suchen -> Schauen ob header korrekt interpretiert werden kann -> Ja/Nein -> Bisherige Bytes speichern/Weitersuchen
            raise NotImplementedError
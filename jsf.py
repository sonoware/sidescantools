from datetime import datetime
import struct
import numpy as np
from pathlib import Path


## header definition from JSF DATA FILE DESCRIPTION 0023492_REV_M, August 20, 2024
class JSFHeader:
    """JSF File Header"""

    HEADERSIZE: int = 16
    fmt: str = "="
    format_def = {
        "sync": "H",
        "ver": "B",
        "session": "B",
        "msg_type": "H",
        "cmd_type": "B",
        "subsys_no": "B",
        "channel": "B",
        "seq_no": "B",
        "reserved": "H",
        "msg_size": "l",
    }
    sync: int
    """Marker for the Sync/Start of Header (always 0x1601)"""
    ver: int
    """Protocol Version"""
    session: int
    """Session Identifier"""
    msg_type: int
    """Message Type (e.g. 80 = Acoustic Return Data)"""
    cmd_type: int
    """Command Type"""
    subsys_no: int
    """Subsystem Number
        - Sub-Bottom (SB) = 0
        - Low frequency data of a dual-frequency side scan = 20
        - High frequency data of a dual-frequency side scan = 21
        - Very High frequency data of a tri-frequency side scan = 22
        - Bathymetric low frequency data of a dual side scan = 40
        - Bathymetric high frequency data of a dual side scan = 41
        - Bathymetric very high frequency of a tri-frequency = 42
        - Bathymetric motion tolerant, low frequency dual side scan = 70
        - Bathymetric motion tolerant high frequency dual side scan = 71
        - Bathymetric motion tolerant very high frequency tri-frequency = 72
        - Raw Serial/UDP/TCP data =100
        - Parsed Serial/UDP/TCP data =101
        - Gap Filler data =120
    """
    channel: int
    """Channel for a Multi-Channel Subsystem

        For Side Scan Subsystems:
        - 0 = Port
        - 1 = Starboard
    """
    seq_no: int
    """Sequence Number """
    reserved: int
    msg_size: int
    """Size of the following message in bytes"""

    def __init__(self, data_bytes):
        self.fmt += "".join(self.format_def.values())
        fields = struct.unpack(self.fmt, data_bytes)
        for (
            var_name,
            value,
        ) in zip(self.format_def.keys(), fields):
            setattr(self, var_name, value)


class JSFSonarDataMessage:
    """JSF Message Type 80: Sonar Data Message"""

    HEADERSIZE: int = 240
    fmt: str = "="

    format_def = {
        "time": "l",
        "starting_depth": "L",
        "ping_no": "L",
        "reserved1": "4s",
        "MSB": "H",
        "LSB": "H",
        "LSB2": "H",
        "reserved2": "6s",
        "ID_code": "h",
        "validity": "H",
        "reserved3": "H",
        "data_format": "h",
        "dist_antenna_tow_aft": "h",
        "dist_antenna_tow_star": "h",
        "reserved4": "4s",
        "pipe_km": "f",
        "heave": "f",
        "reserved5": "24s",
        "gap_filler_lateral": "f",
        "longitude": "l",
        "latitude": "l",
        "coord_units": "h",
        "anno_string": "24s",
        "samples": "H",
        "sampling_interval_ns": "L",
        "gain_adc": "H",
        "user_transmit_level": "h",
        "reserved6": "h",
        "transmit_pulse_start_f": "H",
        "transmit_pulse_end_f": "H",
        "sweep_len_ms": "H",
        "pressure": "l",
        "depth": "l",
        "sample_frequency": "H",
        "outgoing_pulse_identifier": "H",
        "altitude": "l",
        "SOS": "f",
        "mixer_f": "f",
        "year": "h",
        "day": "h",
        "hour": "h",
        "minute": "h",
        "second": "h",
        "time_basis": "h",
        "weighting": "h",
        "num_pulse": "h",
        "compass_heading": "H",
        "pitch": "h",
        "roll": "h",
        "reserved7": "h",
        "reserved8": "h",
        "trigger_source": "h",
        "mark_number": "H",
        "position_fix_hour": "h",
        "position_fix_minutes": "h",
        "position_fix_seconds": "h",
        "course": "h",
        "speed": "h",
        "position_fix_day": "h",
        "position_fix_year": "h",
        "milliseconds_today": "L",
        "max_abs_adc": "H",
        "reserved9": "h",
        "reserved10": "h",
        "sonar_software_ver": "6s",
        "spherical_correction": "l",
        "packet_no": "H",
        "adc_dec": "h",
        "reserved11": "h",
        "water_temp": "h",
        "layback": "f",
        "reserved12": "l",
        "cable_out": "H",
        "reserved13": "H",
    }

    time: int
    """TimeSince1970

        - Ping Time in seconds since the start of time-based time function (midnight 1/1/1970)
    """
    starting_depth: int
    """Starting Depth (window offset) in Samples"""
    ping_no: int
    """Ping Number (increases with each ping) """
    reserved1: bytes
    MSB: int
    """MSB (Most Significant Bits) High order bits to extend 16 bits unsigned
        short values to 20 bits. The 4MSB bits become the most significant
        portion of the new 20-bit value.

        - Bits 0 - 3: Start Frequency
        - Bits 4 - 7: End Frequency
        - Bits 8 - 11: Samples in this Packet
        - Bits 12 - 15: Mark Number (added in protocol version 0xA)

        The Most Significant Bits fields extend 16-bit integers to 20 bits. These
        are added as needed when the range of possible values exceeds what
        can be stored in a 16-bit integer. The simplest way to use these
        additional bits is to treat the value as a 32-bit integer; the existing value
        becomes the least significant 16 bits, and the MSB field becomes the
        next most significant 4 bits with the most significant 12 bits set to
        zeros.
    """
    LSB: int
    """ LSB - Extended precision

        Low-order bits for fields requiring greater precision.

        - Bits 0-7: Sample Interval- - Sample interval fractional component
        - Bits 8-15: Course- - fractional portion of the course
        (Added in protocol version 0xB)
    """
    LSB2: int
    """LBS2 - Extended precision

        Low-order bits for fields requiring greater precision.

        - Bits 0 - 3: Speed - sub-fractional speed component (added in protocol version 0xC).
        - Bits 4 - 13: Sweep Length in Microsecond, from 0 - 999 (added in protocol version 0xD).
        - Bits 14 - 15: Reserved
    """
    reserved2: bytes
    ID_code: int
    """ID Code (always 1)

        1 = Seismic Data
    """
    validity: int
    """Validity Flag

        - Bit 0: Lat Lon or XY valid
        - Bit 1: Course valid
        - Bit 2: Speed valid
        - Bit 3: Heading valid
        - Bit 4: Pressure valid
        - Bit 5: Pitch roll valid
        - Bit 6: Altitude valid
        - Bit 7: Heave
        - Bit 8: Water temperature valid
        - Bit 9: Depth valid
        - Bit 10: Annotation valid
        - Bit 11: Cable counter valid
        - Bit 12: KP valid
        - Bit 13: Position interpolated
        - Bit 14: Water sound speed valid
    """
    reserved3: int
    data_format: int
    """Data Format

        - 0 = one short per sample - envelope data. The total number of bytes
        of data to follow is 2 * samples.
        - 1 = two shorts per sample - stored as real (one short), imaginary (one
        short). The total number of bytes of data to follow is 4 * samples.
        - 2 = one short per sample - before the matched filter. The total
        number of bytes of data to follow is 2 * samples.
        - 9 = two shorts per sample - stored as real (one short), imaginary (one
        short) - before matched filtering. This is the code for unmatched
        filtered analytic data, whereas value 1 is intended for match filtered
        analytic data. The total number of bytes of data to follow is 4 * samples.

        NOTE: Values greater than 255 indicate that the data to follow is in an
        EdgeTech proprietary format.
    """
    dist_antenna_tow_aft: int
    """Distance from Antenna to Tow point in Centimeters.
        Sonar Aft is Positive 
    """
    dist_antenna_tow_star: int
    """Distance from the antenna to the tow point in centimeters.
        Sonar to Starboard is Positive.
    """
    reserved4: bytes
    pipe_km: float
    """Kilometers of Pipe"""
    heave: float
    """Heave in meters
        Positive value: Direction is down
    """
    reserved5: bytes
    gap_filler_lateral: float
    """Gap filler lateral position offset relative to sonar in meters. Offset is
        negative when the gap filler is port of sonar and positive when the gap
        filler is starboard of sonar. The longitude and latitude reported below
        already have this offset applied.
    """
    longitude: int
    """Longitude in 10000 * (Minutes of Arc) or X in Millimeters or
        Decimeters. See Validity Flag (bytes 30 - 31) and Coordinate Units
        (bytes 88 - 89).
    """
    latitude: int
    """Latitude in 10000 * (Minutes of Arc) or Y in Millimeters or Decimeters.
        See Validity Flag (bytes 30 - 31) and Coordinate Units (bytes 88 - 89).
    """
    coord_units: int
    """Coordinate Units

        1 = X, Y in millimeters
        2 = Latitude, longitude in minutes of arc times 10000
        3 = X, Y in decimeters
        4 = X, Y in centimeters
    """
    anno_string: str = ""
    """Annotation String (ASCII Data)"""
    samples: int
    """Samples

        NOTE: For protocol versions 0xA and above, the MSB1 field
        should include the MSBs (Most Significant Bits) needed to
        determine the number of samples.

        See bits 8-11 in bytes 16-17. Field MSB1 for MSBs for
        large sample sizes.
    """
    sampling_interval_ns: int
    """Sampling Interval in Nanoseconds

        NOTE: For protocol versions 0xB and above, see the LSBs field,
        which should include the fractional component needed to
        determine the sample interval.

        See bits 0-7 in bytes 18-19. Field LSB1 for LSBs for increased
        precision.
    """
    gain_adc: int
    """Gain Factor of ADC"""
    user_transmit_level: int
    """User Transmit Level Setting (0 - 100%). """
    reserved6: int
    transmit_pulse_start_f: int
    """Transmit Pulse Starting Frequency in daHz (decaHertz, units of 10Hz).

        NOTE: For protocol versions 0xA and above, the MSB1 field
        should include the MSBs (Most Significant Bits) needed to
        determine the starting frequency of the transmit pulse.

        See Bits 0-3 in bytes 16-17. Field MSB1 for MSBs for large transmit pulse.
    """
    transmit_pulse_end_f: int
    """Transmit Pulse Ending Frequency in daHz (decaHertz, units of 10Hz).

        NOTE: For protocol versions 0xA and above, the MSB1 field
        should include the MSBs (Most Significant Bits) needed to
        determine the starting frequency of the transmit pulse.

        See bits 4-7 in bytes 16-17. Field MSB1 for MSBs for large transmit pulse.
    """
    sweep_len_ms: int
    """Sweep Length in Milliseconds.

        See bytes 18-19 for LSBs (Least Significant Bits). LSB2 bits 4 - 13
        contain the microsecond portion (0 - 999). LSB2 part was added
        in protocol version 0xD and was previously 0.
    """
    pressure: int
    """Pressure in Milli PSI (1 unit = 1/1000 PSI)

        See VALIDITY FLAG (bytes 30-31) 
    """
    depth: int
    """Depth in Millimeters (if not = 0)

        See VALIDITY FLAG (bytes 30-31).
    """
    sample_frequency: int
    """Sample Frequency of the Data in Hertz

        NOTE: For all data types, EXCEPT RAW (Data Format = 2), this is
        the sampling frequency of the data. For RAW data, this is one-
        half the sample frequency of the data (FS /2). All values are
        modulo 65536. Use this in conjunction with the Sample Interval
        (Bytes 114-115) to calculate the correct sample rate.
    """
    outgoing_pulse_identifier: int
    """Outgoing Pulse Identifier"""
    altitude: int
    """Altitude in Millimeters

        A zero implies not filled. See VALIDITY FLAG (Bytes 30-31)
    """
    SOS: float
    """Sound Speed in Meters per Second.

        See VALIDITY FLAG (Byte 30-31). 
    """
    mixer_f: float
    """Mixer Frequency in Hertz

        For single pulse systems, this should be close to the
        center frequency.
        For multi-pulse systems, this should be the approximate
        center frequency of the span of all the pulses.
    """
    year: int
    """Year Data Recorded (CPU time), e.g., 2009.

        - The Ping Time can also be determined from the Year, Day,
        Hour, Minute, and Seconds as per bytes 156 to 165. Provides 1
        second-level accuracy and resolution.
        - See Bytes 0-3. These 2-time stamps are equivalent and
        identical. For most purposes, this should not be used.
        - For higher resolution (milliseconds), use the Year and Day
        values of bytes 156 to 159, use the milliSecondsToday value of
        bytes 200-203 to complete the timestamp.
        - System time is set to UTC, regardless of the time zone. This
        time format is backward compatible with all older protocol
        revisions.
    """
    day: int
    """Day (1 - 366) (should not be used)"""
    hour: int
    """Hour (see Bytes 200-203) (should not be used)"""
    minute: int
    """Minute (should not be used)"""
    second: int
    """Second (should not be used)"""
    time_basis: int
    """Time Basis (always 3)"""
    weighting: int
    """Weighting Factor for Block Floating Point Expansion -- defined as 2 to N Volts for LSB.
        All data MUST be scaled by 2^(-N), where N is the Weighting Factor.
    """
    num_pulse: int
    """Number of Pulses in the Water"""
    compass_heading: int
    """Compass Heading (0 to 359.99) in units of 1/100 Degree.

    See VALIDITY FLAG (bytes 30-31).

    The Compass heading is the magnetic heading of the towfish. If a Gyro
    sensor is properly interfaced with the Discover Topside Acquisition
    Unit with a valid NMEA HDT message, this field will contain the Gyro
    heading relative to True North.
    """
    pitch: int
    """Pitch [(degrees / 180.0) * 32768.0] maximum resolution.

        Positive values indicate bow up.

        See VALIDITY FLAG (bytes 30-31).
    """
    roll: int
    """Roll [(degrees / 180.0) * 32768.0] maximum resolution.

        Positive values indicate port up.

        See VALIDITY FLAG (bytes 30-31).
    """
    reserved7: int
    reserved8: int
    trigger_source: int
    """Trigger Source

        - 0 = Internal
        - 1 = External
        - 2 = Coupled
    """
    mark_number: int
    """Mark Number

        0 = No Mark

        See bytes 16 -17 fields MSB1 for MSBs (Most Significant Bits) for large
        values (> 655350).
        """
    position_fix_hour: int
    """Position Fix Hour (0 - 23)

        NOTE: The NAV time is the time of the latitude and longitude fix.
    """
    position_fix_minutes: int
    """Position Fix Minutes (0 - 59)

        NOTE: The NAV time is the time of the latitude and longitude fix. 
    """
    position_fix_seconds: int
    """Position Fix Seconds (0 - 59)

        NOTE: The NAV time is the time of the latitude and longitude fix. 
    """
    course: int
    """Course in Degrees (0 to 359.9)

        Starting with the protocol version, 0x0C, two digits of fractional
        degrees are stored in LSB1. The fractional portion is in LSBs (Least
        Significant Bits). See bytes 18 - 19.
    """
    speed: int
    """Speed - in Tenths of a Knot

        Starting with protocol version 0x0C, one additional digit of fractional
        knot (1/100) is stored in LSB2. See LSB2 (bytes 20 -21) for an additional
        fractional digit.
    """
    position_fix_day: int
    """Position Fix Day (1 - 366)"""
    position_fix_year: int
    """Position Fix Year"""
    milliseconds_today: int
    """Milliseconds Today (Since Midnight)

        Use with seconds since 1970 to get time to the milliseconds (time of Ping).
    """
    max_abs_adc: int
    """Maximum Absolute Value of ADC Samples in this Packet"""
    reserved9: int
    reserved10: int
    sonar_software_ver: str = ""
    """Sonar Software Version Number - ASCII"""
    spherical_correction: int
    """Initial Spherical Correction Factor in Samples times 100.

        A value of -1 indicates that the spherical spreading is disabled.
    """
    packet_no: int
    """Packet Number

        Each ping starts with packet 1
    """
    adc_dec: int
    """ADC Decimation * 100 times"""
    reserved11: int
    water_temp: int
    """Water Temperature in Units of 1/10 Degree C.

        See VALIDITY FLAG (bytes 30-31).
    """
    layback: float
    """Layback

        Distance to the sonar in meters.
    """
    reserved12: int
    cable_out: int
    """Cable Out in Decimeters
        
        See VALIDITY FLAG (bytes 30-31). 
    """
    reserved13: int

    def __init__(self, data_bytes):
        self.fmt += "".join(self.format_def.values())
        fields = struct.unpack(self.fmt, data_bytes)
        for (
            var_name,
            value,
        ) in zip(self.format_def.keys(), fields):
            if hasattr(self, var_name):
                if getattr(self, var_name) is str():
                    value = value.decode("ascii")
            setattr(self, var_name, value)

        self.time_converted = datetime.fromtimestamp(self.time / 1e3)
        size_factor = 2
        if self.data_format == 1 or self.data_format == 9:
            size_factor = 4
        self.byte_len_sonar_data = (
            self.samples * size_factor
        )  # num of sonar values * size

    def load_data(self, data):

        if self.data_format == 0:
            # one short per sample - envelope data
            fields = struct.unpack(f"{self.samples}h", data)

            self.data = np.array(fields, dtype=np.int16)
            self.data = np.array(self.data, dtype=np.float32)
            self.data *= 2 ** (-1 * self.weighting)

        elif self.data_format == 1:
            # two shorts per sample - stored as real (one short), imaginary (one short)
            fields = struct.unpack(f"{self.samples * 2}h", data)

            self.data = np.array(fields, dtype=np.int16)
            # TODO: interpretation

        elif self.data_format == 2:
            # one short per sample - before the matched filter
            fields = struct.unpack(f"{self.samples}h", data)

            self.data = np.array(fields, dtype=np.int16)
            # TODO: interpretation

        elif self.data_format == 9:
            # two shorts per sample - stored as real (one short), imaginary (one short) - before matched filtering
            fields = struct.unpack(f"{self.samples * 2}h", data)

            self.data = np.array(fields, dtype=np.int16)
            # TODO: interpretation

        else:
            print("Case not implemented - the seems to be a proprietry format")
            NotImplementedError()


class JSFSonarDataPacket:
    """A JSF Package containing a header and the corresponding message"""

    header: JSFHeader
    message: JSFSonarDataMessage

    def __init__(self, header: JSFHeader, message: JSFSonarDataMessage):
        self.header = header
        self.message = message


class JSFSonarMessageStatus:
    """JSF Message Type 40: Sonar Message Status"""

    MESSAGESIZE: int = 100
    fmt: str = "="

    format_def = {
        "overflowCount": "L",
        "errorCount": "L",
        "lastError10": "10h",
        "freeDiskSpace": "L",
        "dataAcquisitionActivity": "L",
        "serviceNeeded": "B",
        "reserved1": "2B",
        "storageFlags": "B",
        "temperatureStatus": "B",
        "timeStatus": "B",
        "reserved2": "H",
        "bottleTemperature": "l",
        "ambientTemperature": "l",
        "power48Volts": "l",
        "reserved3": "l",
        "lowRateIO4": "4h",
        "serialPortState": "8B",
        "runtimeAlerts": "l",
        "reserved4": "5l",
    }

    overfowCount: int
    errorCount: int
    astError10: np.array
    """lastError[10]. Error IDs of the last ten errors"""
    freeDiskSpace: int
    """freeDiskSpace. Disk space available for sonar data storage in kb."""
    dataAcquisitionActivity: int
    """dataAcquisitionActivityIndicator. Indicates that data is being received (increments over time)."""
    serviceNeeded: int
    """serviceNeeded. General service warning message. This is a bit map
        with one bit for each power on self-test status.

        - A non-zero bit indicates a POST test failure.
        - Bit 0: SB Power Amp feedback on channel 0 test failed.
        - Bit 1: SB Power Amp feedback on secondary channel (1) test failed.
        - Bit 2: Interface Card diagnostic failed.
        - Bit 3: Health Monitor Sensors (formerly the 48 Volt Power Check)
        - Bit 4: Reserved
        - Bit 5: Internal Configuration Error (Missing Pulse File, etc.)
        - Bit 6: Reserved
        - Bit 7: Reserved
    """
    reserved1: bytes
    storageFags: int
    """storageFlags. Storage subsystem flags

        - Bit 0: Disk storage is enabled (should be set when on).
        - Bit 1: Disk primary drive error detected. The operator needs to reset.
        - Bit 2: Disk write error on the drive - all storage disabled.
        - Bit 7: Disk playback is enabled.
    """
    temperatureStatus: int
    """temperatureStatus. Ambient Temperature Status

        - Bit 0: Temperature is OK
        - Bit 1: Temperature is in error - below the minimum possible value.
        - Bit 2: Temperature is below the recommended value.
        - Bit 3: Temperature is above the recommended value.
        - Bit 4: Temperature is too high - PINGING IS DISABLED.
        - Bit 5: Temperature is in error - above the maximum possible value
    """
    timeStatus: int
    """timeStatus. Status of time synchronization

        - Bits 0-3: Source for time sync. See TimeSyncSentenceType above for values.
        - Bits 4-6: Status of time sync. See TimeSyncStatusEnumType above.
        - Bit 7: If a PPS is enabled but either not active or not at the
          expected pulse per second rate
    """
    reserved2: bytes
    bottelTemperature: int
    """bottleTemperature. Bottle temperature in Degrees C * 1000."""
    amientTemperature: int
    """ambientTemperature. Ambient temperatures in Degrees C*1000"""
    power48Vots: int
    """power48Volts. Check of 48 Volt Power in millivolts."""
    reserved3: int
    owRateIO4: np.array
    """lowRateIO[4]. Low rate misc IO analog values."""
    seriaPortState: np.array
    """serialPortState[8]. 
    
        Serial Port Summary Status. There are 4 bits for
        each of up to 16 ports. Each 4-bit value indicates the state of the
        port, which can be any of the values listed above in
        SerialPortStateEnumType
    """
    runtimeAerts: int
    """Runtime alert bits. See SonarAlertEnumType for symbolic bit defines. """
    reserved4: bytes

    def __init__(self, data_bytes):
        self.fmt += "".join(self.format_def.values())
        fields = struct.unpack(self.fmt, data_bytes)
        for (
            var_name,
            value,
        ) in zip(self.format_def.keys(), fields):
            if hasattr(self, var_name):
                if getattr(self, var_name) is np.array:
                    value = np.array(value, dtype=int)
            setattr(self, var_name, value)


class JSFNavigationOffsets:
    """JSF Message Type 181: Navigation Offsets"""

    MESSAGESIZE: int = 64
    fmt: str = "="

    format_def = {
        "x_offset": "f",
        "y_offset": "f",
        "latitude_offset": "f",
        "longitude_offset": "f",
        "aft_offset": "f",
        "starboard_offset": "f",
        "depth_offset": "f",
        "altitude_offset": "f",
        "heading_offset": "f",
        "pitch_offset": "f",
        "roll_offset": "f",
        "yaw_offset": "f",
        "tow_point_offset": "f",
        "reserved": "3f",
    }

    x_offset: float
    """X offset in meters"""
    y_offset: float
    """Y offset in meters"""
    latitude_offset: float
    """Latitude Offset in degrees"""
    longitude_offset: float
    """Longitude Offset in degrees"""
    aft_offset: float
    """Aft Offset in meters: Forward is negative"""
    starboard_offset: float
    """Starboard Offset in meters: Port is negative"""
    depth_offset: float
    """Depth Offset in meters: Up is negative"""
    altitude_offset: float
    """Altitude Offset in meters: Down is negative"""
    heading_offset: float
    """Heading Offset in degrees"""
    pitch_offset: float
    """Pitch Offset in degrees: Nose up is positive"""
    roll_offset: float
    """Roll Offset in degrees: Port side up is positive"""
    yaw_offset: float
    """Yaw Offset in degrees: Toward Port is negative"""
    tow_point_offset: float
    """Tow point elevation offset (Up is positive)"""
    reserved: bytes

    def __init__(self, data_bytes):
        self.fmt += "".join(self.format_def.values())
        fields = struct.unpack(self.fmt, data_bytes)
        for (
            var_name,
            value,
        ) in zip(self.format_def.keys(), fields):
            setattr(self, var_name, value)


class JSFSystemInformation:
    """Message Type 182: System Information"""

    MESSAGESIZE = 24
    fmt: str = "="

    format_def = {
        "system_type": "l",
        "low_rate_IO_enabled": "l",
        "vers_sonar_software": "l",
        "num_subsystems": "l",
        "num_serial_devices": "l",
        "serial_num_tow_vehicle": "l",
    }

    system_type: int
    """System Type"""
    low_rate_IO_enabled: int
    """Low Rate I/O Enabled Option (0 = disabled)"""
    vers_sonar_software: int
    """Version Number of Sonar software used to generate data """
    num_subsystems: int
    """Number of Subsystems Present in this Message"""
    num_serial_devices: int
    """Number of Serial Port Devices Present in this Message"""
    serial_num_tow_vehicle: int
    """Serial Number of Tow Vehicle used to Collect Data"""
    reserved: bytes

    def __init__(self, bytes_def, bytes_undef):
        self.fmt += "".join(self.format_def.values())
        fields = struct.unpack(self.fmt, bytes_def)
        for (
            var_name,
            value,
        ) in zip(self.format_def.keys(), fields):
            setattr(self, var_name, value)

        self.reserved = bytes_undef


class JSFTargetFileData:
    """Message Type 1260: Target File Data"""

    HEADERSIZE = 824
    fmt: str = "="

    format_def = {
        "time": "l",
        "millisec": "l",
        "center_ping_no": "L",
        "subsys_ID": "B",
        "channel_no": "B",
        "coord_units": "h",
        "longitude": "f",
        "latitude": "f",
        "altitude": "f",
        "course": "f",
        "heading": "f",
        "slant_range": "f",
        "target_length": "f",
        "target_width": "f",
        "target_height": "f",
        "target_vers": "H",
        "reserved": "2b",
        "x_offset": "f",
        "y_offset": "f",
        "latitude_offset": "f",
        "longitude_offset": "f",
        "depth_offset": "f",
        "heading_offset": "f",
        "pitch_offset": "f",
        "roll_offset": "f",
        "aft_offset": "f",
        "starboard_offset": "f",
        "target_name": "40s",
        "tag_label": "40s",
        "target_description": "128s",
        "full_path": "512s",
        "target_image_size": "L",
        "reserved": "4s",
    }

    time: int
    """Time in Seconds since 1/1/1970"""
    millisec: int
    """Milliseconds in the Current Second"""
    center_ping_no: int
    """Center Ping number"""
    subsys_ID: int
    """Subsystem ID"""
    channel_no: int
    """Channel Number"""
    coord_units: int
    """Coordinate Units

        - 1 = X, Y in millimeters
        - 2 = Latitude, longitude in minutes of arc times 10000
        - 3 = X, Y in decimeters
        - 4 = X, Y in centimeters

        NOTE: Option 4 (X, Y in centimeters) is only available in protocol
        versions 0x11 and above.
    """
    longitude: float
    """Longitude (X value)"""
    latitude: float
    """Latitude (Y values)"""
    altitude: float
    """Altitude in meters"""
    course: float
    """Course in degrees"""
    heading: float
    """Heading in degrees"""
    slant_range: float
    """Slant range in meters"""
    target_length: float
    """Length of target in meters"""
    target_width: float
    """Width of the target in meters"""
    target_height: float
    """Height of target in meters """
    target_vers: int
    """Target File version number"""
    reserved: bytes
    x_offset: float
    """X offset in meters"""
    y_offset: float
    """Y offset in meters"""
    latitude_offset: float
    """Latitude offset in degrees"""
    longitude_offset: float
    """Longitude offset in degrees"""
    depth_offset: float
    """Depth offset in meters (Down is positive)"""
    heading_offset: float
    """Heading offset in degrees"""
    pitch_offset: float
    """Pitch offset in degrees (Nose up is positive)"""
    roll_offset: float
    """Roll offset in degrees (Port side up is positive)"""
    aft_offset: float
    """Aft offset in meters"""
    starboard_offset: float
    """Starboard offset in meters"""
    target_name: str
    """Target Name"""
    tag_label: str
    """Tag label"""
    target_description: str
    """Description of a target from operator input"""
    full_path: str
    """Full path of target file name"""
    target_image_size: int
    """Target image size in bytes (size of the JPEG target data)"""
    reserved: bytes

    img_jpg: bytes

    def __init__(self, data_bytes):
        self.fmt += "".join(self.format_def.values())
        fields = struct.unpack(self.fmt, data_bytes)
        for (
            var_name,
            value,
        ) in zip(self.format_def.keys(), fields):
            if hasattr(self, var_name):
                if getattr(self, var_name) is str():
                    value = value.decode("ascii")
            setattr(self, var_name, value)


class JSFNMEAString:
    """Message Type 2002: NMEA String"""

    HEADERSIZE = 12
    fmt: str = "="

    format_def = {
        "time": "l",
        "milliseconds": "l",
        "source": "B",
        "reserved": "3s",
    }

    time: int
    """Time in Seconds since 1/1/1970"""
    milliseconds: int
    """Milliseconds in the Current Second"""
    source: int
    """Source

        - 1 = Sonar
        - 2 = Discover
        - 3 = ETSI
    """
    reserved: bytes

    nmea_string: str
    """NMEA String Data"""

    def __init__(self, data_bytes):
        self.fmt += "".join(self.format_def.values())
        fields = struct.unpack(self.fmt, data_bytes)
        for (
            var_name,
            value,
        ) in zip(self.format_def.keys(), fields):
            setattr(self, var_name, value)


class JSFPitchRollData:
    """Message Type 2020: Pitch Roll Data"""

    MESSAGESIZE = 44
    fmt: str = "="

    format_def = {
        "time": "l",
        "milliseconds": "l",
        "reserved1": "4s",
        "acc_x": "h",
        "acc_y": "h",
        "acc_z": "h",
        "rate_gyro_x": "h",
        "rate_gyro_y": "h",
        "rate_gyro_z": "h",
        "pitch": "h",
        "roll": "h",
        "temperature": "h",
        "device_info": "H",
        "heave": "h",
        "heading": "H",
        "validity": "l",
        "yaw": "h",
        "reserved2": "h",
    }

    time: int
    """Time in Seconds since 1/1/1970"""
    milliseconds: int
    """Milliseconds in the Current Second"""
    reserved1: bytes
    acc_x: int
    """Acceleration in X

        Multiply by (20 * 1.5) / (32768) to get Gs 
    """
    acc_y: int
    """Acceleration in Y
    
        Multiply by (20 * 1.5) / (32768) to get Gs 
    """
    acc_z: int
    """Acceleration in Y
    
        Multiply by (20 * 1.5) / (32768) to get Gs 
    """
    rate_gyro_x: int
    """Rate Gyro in X

        Multiply by (500 * 1.5) / (32768) to get Degrees/Sec 
    """
    rate_gyro_y: int
    """Rate Gyro in Y
    
        Multiply by (500 * 1.5) / (32768) to get Degrees/Sec 
    """
    rate_gyro_z: int
    """Rate Gyro in Z
    
        Multiply by (500 * 1.5) / (32768) to get Degrees/Sec 
    """
    pitch: int
    """Pitch

        Multiply by (180.0 / 32768.0) to get Degrees
        Bow up is positive
    """
    roll: int
    """Roll:

        Multiply by (180.0 / 32768.0) to get Degrees.
        Port up is positive
    """
    temperature: int
    """Temperature in Units of 1/10 of a Degree Celsius"""
    device_info: int
    """Device-specific info.
        This is device-specific info provided for diagnostic purposes.
    """
    heave: int
    """Estimated Heave in Millimeters. Positive is down."""
    heading: int
    """Heading in units of 0.01 Degrees (0…360)"""
    validity: int
    """Data Validity Flags

        - Bit 0: Ax
        - Bit 1: Ay
        - Bit 2: Az
        - Bit 3: Rx
        - Bit 4: Ry
        - Bit 5: Rz
        - Bit 6: Pitch
        - Bit 7: Roll
        - Bit 8: Heave
        - Bit 9: Heading
        - Bit 10: Temperature
        - Bit 11: Device Info
        - Bit 12: Yaw
    """
    yaw: int
    """Yaw in units of 0.01 Degrees (0…360)"""
    reserved2: int

    def __init__(self, data_bytes):
        self.fmt += "".join(self.format_def.values())
        fields = struct.unpack(self.fmt, data_bytes)
        for (
            var_name,
            value,
        ) in zip(self.format_def.keys(), fields):
            setattr(self, var_name, value)


class JSFPressureSensorReading:
    """Message Type 2060: Pressure Sensor Reading"""

    MESSAGESIZE = 76
    fmt: str = "="

    format_def = {
        "time": "l",
        "milliseconds": "l",
        "reserved1": "4s",
        "pressure": "l",
        "temperature": "l",
        "salinity": "l",
        "validity": "l",
        "conductivity": "l",
        "sos": "l",
        "depth": "l",
        "reserved2": "36s",
    }

    time: int
    """Time in Seconds since 1/1/1970"""
    milliseconds: int
    """Milliseconds in the Current Second"""
    reserved1: bytes
    pressure: int
    """Pressure in Units of 1/1000th of a PSI"""
    temperature: int
    """Temperature in Units of 1/1000th of a Degree Celsius."""
    salinity: int
    """Salinity in Parts Per Million"""
    validity: int
    """Validity Data Flag:

        - Bit 0: Pressure
        - Bit 1: Temperature
        - Bit 2: Salt PPM
        - Bit 3: Conductivity
        - Bit 4: Sound velocity
        - Bit 5: Depth
    """
    conductivity: int
    """Conductivity in Micro-Siemens per Centimeter"""
    sos: int
    """Velocity of Sound in Millimeters per Second"""
    depth: int
    """Depth in Meters"""
    reserved2: bytes

    def __init__(self, data_bytes):
        self.fmt += "".join(self.format_def.values())
        fields = struct.unpack(self.fmt, data_bytes)
        for (
            var_name,
            value,
        ) in zip(self.format_def.keys(), fields):
            setattr(self, var_name, value)


class JSFReflectionCoeff:
    """Message Type 2071: Reflection Coefficient"""

    HEADERSIZE = 32
    fmt: str = "="

    format_def = {
        "time": "l",
        "milliseconds": "l",
        "reserved1": "4s",
        "pingNumber": "L",
        "reflectionCoefficientDecibels": "f",
        "altitudeMilliseconds": "f",
        "calibrationGainDecibels": "f",
        "calibrationReferenceDecibels": "f",
    }

    time: int
    """Time in Seconds since 1/1/1970"""
    milliseconds: int
    """Milliseconds in the Current Second"""
    reserved1: bytes
    pingNumber: int
    """pingNumber. The ping number for which the reflection coefficient was
        computed. Use with the subsystem, channel, and timestamp
        unambiguously associated with a ping.
    """
    reflectionCoefficientDecibels: float
    """reflectionCoefficientDecibels. The reflection coefficient in decibels.
        Good values are between 0 and -40.
    """
    altitudeMilliseconds: float
    """altitudeMilliseconds. Altitude in milliseconds where the reflection
        coefficient was computed.
    """
    calibrationGainDecibels: float
    """calibrationGainDecibels. Pulse calibration gain in decibels applied to
        sonar samples before reflection coefficient calculation.
    """
    calibrationReferenceDecibels: float
    """calibrationReferenceDecibels. The calibration “zero-point’ in decibels.
        This value was subtracted from the computer reflection coefficient
        value to get the value reported in the reflection coefficient (bytes 4-7).
    """
    coeff_data: bytes

    def __init__(self, data_bytes):
        self.fmt += "".join(self.format_def.values())
        fields = struct.unpack(self.fmt, data_bytes)
        for (
            var_name,
            value,
        ) in zip(self.format_def.keys(), fields):
            setattr(self, var_name, value)


class JSFDopplerVelocityLog:
    """Message Type 2080: Doppler Velocity Log Data (DVL)"""

    MESSAGESIZE = 72
    fmt: str = "="

    format_def = {
        "time": "l",
        "milliseconds": "l",
        "reserved1": "4s",
        "validity": "L",
        "dist_to_bottom1": "l",
        "dist_to_bottom2": "l",
        "dist_to_bottom3": "l",
        "dist_to_bottom4": "l",
        "x_velo_bottom": "h",
        "y_velo_bottom": "h",
        "z_velo_bottom": "h",
        "x_velo_water": "h",
        "y_velo_water": "h",
        "z_velo_water": "h",
        "depth": "H",
        "pitch": "h",
        "roll": "h",
        "heading": "H",
        "salinity": "H",
        "temperature": "h",
        "sos": "h",
        "reserved2": "14s",
    }

    time: int
    """Time in Seconds since 1/1/1970"""
    milliseconds: int
    """Milliseconds in the Current Second"""
    reserved1: bytes
    validity: int
    """Validity Data Flags:

        - Bit 0: X, Y Velocity Present
        - Bit 1: 0 = Earth Coordinates, 1= Ship coordinates
        - Bit 2: Z (Vertical Velocity) Present
        - Bit 3: X, Y Water Velocity Present
        - Bit 4: Z (Vertical Water Velocity) Present
        - Bit 5: Distance to Bottom Present
        - Bit 6: Heading Present
        - Bit 7: Pitch Present
        - Bit 8: Roll Present
        - Bit 9: Temperature Present
        - Bit 10: Depth Present
        - Bit 11: Salinity Present
        - Bit 12: Sound Velocity Present
        - Bit 31: Error Detected
        - Rest: Reserved, Presently 0
    """
    dist_to_bottom1: int
    """Distance to bottom in centimeters for beam 1.
        A value of 0 indicates an invalid or non-existing reading.
    """
    dist_to_bottom2: int
    """Distance to bottom in centimeters for beam 2.
        A value of 0 indicates an invalid or non-existing reading.
    """
    dist_to_bottom3: int
    """Distance to bottom in centimeters for beam 3.
        A value of 0 indicates an invalid or non-existing reading.
    """
    dist_to_bottom4: int
    """Distance to bottom in centimeters for beam 4.
        A value of 0 indicates an invalid or non-existing reading.
    """
    x_velo_bottom: int
    """X Velocity with Respect to the Bottom in millimeters per second.
        A positive value indicates Starboard or East. 
    """
    y_velo_bottom: int
    """Z Vertical Velocity with Respect to the Bottom in millimeters per second.
        A positive value indicates Forward or North.
    """
    z_velo_bottom: int
    """Z Velocity with Respect to the Bottom in millimeters per second.
        A positive value indicates Upward.
    """
    x_velo_water: int
    """X Velocity with respect to a water layer in millimeters per second.
        A positive value indicates Starboard or East.
    """
    y_velo_water: int
    """Y Velocity with respect to a water layer in millimeters per second.
        A positive value indicates Forward or North.
    """
    z_velo_water: int
    """Z Vertical Velocity with respect to a water layer in millimeters per
        second. A positive value indicates Upward.
    """
    depth: int
    """Depth from Depth Sensor in Decimeters"""
    pitch: int
    """Pitch in units of 0.01 of a Degree (-180 to +180).
        A positive value is Bow Up.
    """
    roll: int
    """Roll in units of 0.01 of a Degree (-180 to +180).
        A positive value is Port Up. 
    """
    heading: int
    """Heading in units of 0.01 of a Degree (0 to 360)"""
    salinity: int
    """Salinity in 1 Part Per Thousand"""
    temperature: int
    """Temperature in units of 1/100 of a degree Celsius"""
    sos: int
    """Sound Velocity in Meters per Second """
    reserved2: bytes

    def __init__(self, data_bytes):
        self.fmt += "".join(self.format_def.values())
        fields = struct.unpack(self.fmt, data_bytes)
        for (
            var_name,
            value,
        ) in zip(self.format_def.keys(), fields):
            setattr(self, var_name, value)


class JSFSituationComprehensiveMessage:
    """Message Type 2091: Situation Comprehensive Message (Version 2)"""

    MESSAGESIZE = 100
    fmt: str = "="

    format_def = {
        "time": "l",
        "milliseconds": "l",
        "reserved1": "4s",
        "validity": "L",
        "velocity12": "B",
        "reserved2": "3s",
        "timestamp": "Q",
        "latitude": "d",
        "longitude": "d",
        "depth": "f",
        "altitude": "f",
        "heave": "f",
        "velocity1": "f",
        "velocity2": "f",
        "velocity_down": "f",
        "pitch": "f",
        "roll": "f",
        "heading": "f",
        "sos": "f",
        "water_temperature": "f",
        "reserved3": "12s",
    }

    time: int
    """Time in Seconds since 1/1/1970"""
    milliseconds: int
    """Milliseconds in the Current Second"""
    reserved1: bytes
    validity: int
    """Validity Flag:

        - Bit 0: Timestamp Provided by the Source Valid
        - Bit 1: Longitude Valid
        - Bit 2: Latitude Valid
        - Bit 3: Depth Valid
        - Bit 4: Altitude Valid
        - Bit 5: Heave Valid
        - Bit 6: Velocity 1 & 2 Valid
        - Bit 7: Velocity down Valid
        - Bit 8: Pitch Valid
        - Bit 9: Roll Valid
        - Bit 10: Heading Valid
        - Bit 11: Sound Speed Valid
        - Bit 12: Water Temperature Valid
        - Others: Reserved, Presently 0.
    """
    velocity12: int
    """Velocity12 Directions (Velocity1 and Velocity2 Types):

        - 0 = North and East,
        - 1 = Forward and Starboard,
        - 2 = +45 Degrees Rotated from Forward.
    """
    reserved2: bytes
    timestamp: int
    """Timestamp (0.01 of a microsecond)

    Microsecond since 12:00:00 AM GST, January 1, 1970. To get seconds since 1970, divide by 1e7)
    """
    latitude: float
    """Latitude in Degrees (North is Positive)"""
    longitude: float
    """Longitude in Degrees (East is Positive)"""
    depth: float
    """Depth in Meters (Below Water Surface)"""
    altitude: float
    """Altitude in Meters (Above Seafloor)"""
    heave: float
    """Heave in Meter (Positive is Down)"""
    velocity1: float
    """Velocity1 in Meters per Second (North Velocity or Forward)"""
    velocity2: float
    """Velocity2 in Meters per Second (East Velocity or Starboard)"""
    velocity_down: float
    """Velocity Down in Meters per Second (Down Velocity)"""
    pitch: float
    """Pitch in Degrees (Bow up is Positive)"""
    roll: float
    """Roll in Degrees (Port is Positive)"""
    heading: float
    """Heading in Degrees (0 to 359.9)"""
    sos: float
    """Sound Speed in Meters per Second"""
    water_temperature: float
    """Water Temperature (in Degrees Celsius)"""
    reserved3: bytes

    def __init__(self, data_bytes):
        self.fmt += "".join(self.format_def.values())
        fields = struct.unpack(self.fmt, data_bytes)
        for (
            var_name,
            value,
        ) in zip(self.format_def.keys(), fields):
            setattr(self, var_name, value)


class JSFCableCounterDataMessage:
    """Message Type 2100: Cable Counter Data Message"""

    MESSAGESIZE = 32
    fmt: str = "="

    format_def = {
        "time": "l",
        "milliseconds": "l",
        "reserved1": "4s",
        "cable_length": "f",
        "cable_speed": "f",
        "cable_length_valid": "h",
        "cable_speed_valid": "h",
        "cable_counter_error": "h",
        "cable_tension_valid": "h",
        "cable_tension_kg": "f",
    }

    time: int
    """Time in Seconds since 1/1/1970"""
    milliseconds: int
    """Milliseconds in the Current Second"""
    reserved1: bytes
    cable_length: float
    """Cable Length in Meters"""
    cable_speed: float
    """Cable Speed in Meters per Second"""
    cable_length_valid: int
    """Cable Length Valid Flag (0 - Invalid)"""
    cable_speed_valid: int
    """Cable Speed Valid Flag (0 - Invalid)"""
    cable_counter_error: int
    """Cable Counter Error (0 - No Error)"""
    cable_tension_valid: int
    """Cable Tension Valid Flag (0 - Invalid)"""
    cable_tension_kg: float
    """Cable Tension in Kilograms"""

    def __init__(self, data_bytes):
        self.fmt += "".join(self.format_def.values())
        fields = struct.unpack(self.fmt, data_bytes)
        for (
            var_name,
            value,
        ) in zip(self.format_def.keys(), fields):
            setattr(self, var_name, value)


class JSFKilometerOfPipe:
    """Message Type 2101: Kilometer of Pipe Data"""

    MESSAGESIZE = 20
    fmt: str = "="

    format_def = {
        "time": "l",
        "milliseconds": "l",
        "source": "B",
        "reserved1": "3s",
        "km_of_pipe": "f",
        "km_pipe_valid": "h",
        "km_pipe_report_error": "h",
    }

    time: int
    """Time in Seconds since 1/1/1970"""
    milliseconds: int
    """Milliseconds in the Current Second"""
    source: int
    """Source

        - 1 = Sonar
        - 2 = DISCOVER
        - 3 = ETSI
    """
    reserved1: bytes
    km_of_pipe: float
    """Kilometer of Pipe (KP)"""
    km_pipe_valid: int
    """Flag (Valid KP Value)"""
    km_pipe_report_error: int
    """Flag (KP Report Error)"""

    def __init__(self, data_bytes):
        self.fmt += "".join(self.format_def.values())
        fields = struct.unpack(self.fmt, data_bytes)
        for (
            var_name,
            value,
        ) in zip(self.format_def.keys(), fields):
            setattr(self, var_name, value)


class JSFContainerTimestampMsg:
    """Message Type 2111: Container Timestamp Message"""

    MESSAGESIZE = 12
    fmt: str = "="

    format_def = {
        "time": "l",
        "milliseconds": "l",
        "reserved1": "4s",
    }

    time: int
    """Time in Seconds since 1/1/1970"""
    milliseconds: int
    """Milliseconds in the Current Second"""
    reserved1: bytes

    def __init__(self, data_bytes):
        self.fmt += "".join(self.format_def.values())
        fields = struct.unpack(self.fmt, data_bytes)
        for (
            var_name,
            value,
        ) in zip(self.format_def.keys(), fields):
            setattr(self, var_name, value)


# THIS MESSAGE TYPE IS UNDOCUMENTED IN JSF FORMAT DEFINITION - Reverse engineered
class JSFStorageMsgTimestamp:
    """Message Type 426: Storage Message Timestamp - interpretation unclear, reverse engineered using the JSF File Viewer"""

    MESSAGESIZE = 8
    fmt: str = "="

    format_def = {
        "time": "l",
        "reserved1": "4s",
    }

    time: int
    """Time in Seconds since 1/1/1970"""
    reserved1: bytes

    def __init__(self, data_bytes):
        self.fmt += "".join(self.format_def.values())
        fields = struct.unpack(self.fmt, data_bytes)
        for (
            var_name,
            value,
        ) in zip(self.format_def.keys(), fields):
            setattr(self, var_name, value)

# Class to store unknown data
class JSFUnknownMessage:
    """Wrapper class to store data that couldn't be interpreted"""
    size: int
    data: bytes
    type: int

    def __init__(self, size, data, type):
        self.size = size
        self.data = data
        self.type = type

class JSFFile:
    """Class for reading JSF files. All header and messages are stored in the packets list."""

    packets: list = []
    file_path: Path
    num_channel: int = 0
    sync_number: int = 0
    expected_msg_no = np.zeros(2)  # assuming 2 channel systems, otherwise ignore for now

    def __init__(self, path: Path | str):
        if path is str:
            self.file_path = Path(path)
        else:
            self.file_path = path

        self.packets = []
        with open(self.file_path, "rb") as f:

            # read headers till EOF is reached (b'')
            while header_bytes := f.read(JSFHeader.HEADERSIZE):
                cur_header = JSFHeader(header_bytes)
                if self.sync_number == 0:
                    self.sync_number = cur_header.sync

                # check whether sync number stays sane
                if self.sync_number == cur_header.sync:

                    match cur_header.msg_type:

                        # Sonar message status
                        case 40:

                            if JSFSonarMessageStatus.MESSAGESIZE == cur_header.msg_size:
                                message_bytes = f.read(
                                    JSFSonarMessageStatus.MESSAGESIZE
                                )
                                data_message = JSFSonarMessageStatus(message_bytes)

                                self.packets.append(cur_header)
                                self.packets.append(data_message)

                            else:
                                print("JSFSonarMessageStatus: Error in msg size")

                                # only read without interpreting the data
                                message_bytes = f.read(cur_header.msg_size)

                        # Sonar data message
                        case 80:
                            # read the message of the current header
                            message_bytes = f.read(JSFSonarDataMessage.HEADERSIZE)
                            data_message = JSFSonarDataMessage(message_bytes)

                            # read the data attached to the message
                            data_bytes = f.read(data_message.byte_len_sonar_data)
                            data_message.load_data(data_bytes)

                            packet = JSFSonarDataPacket(cur_header, data_message)
                            self.num_channel = max(
                                self.num_channel, cur_header.channel + 1
                            )
                            self.packets.append(packet)

                        # Navigation Offsets
                        case 181:
                            if JSFNavigationOffsets.MESSAGESIZE == cur_header.msg_size:
                                message_bytes = f.read(JSFNavigationOffsets.MESSAGESIZE)
                                data_message = JSFNavigationOffsets(message_bytes)

                                self.packets.append(cur_header)
                                self.packets.append(data_message)
                            else:
                                print("JSFNavigationOffsets: Error in msg size")

                                # only read without interpreting the data
                                message_bytes = f.read(cur_header.msg_size)

                        # System Information
                        case 182:
                            message_bytes_def = f.read(JSFSystemInformation.MESSAGESIZE)

                            if JSFSystemInformation.MESSAGESIZE < cur_header.msg_size:
                                message_bytes_undef = f.read(
                                    cur_header.msg_size
                                    - JSFSystemInformation.MESSAGESIZE
                                )
                            else:
                                message_bytes_undef = b""

                            data_message = JSFSystemInformation(
                                message_bytes_def, message_bytes_undef
                            )

                            self.packets.append(cur_header)
                            self.packets.append(data_message)

                            # gather information for data processing
                            self.num_subsys = data_message.num_subsystems

                        # Storage Message Timestamp
                        case 426:
                            # read the message of the current header
                            message_bytes = f.read(JSFStorageMsgTimestamp.MESSAGESIZE)
                            time_msg = JSFStorageMsgTimestamp(message_bytes)

                            self.packets.append(cur_header)
                            self.packets.append(time_msg)

                        # File padding message
                        case 428:
                            message_bytes = f.read(cur_header.msg_size)

                            # ignore header and message, info from Rev 1.20 about this message type:
                            #  In some implementations files are padded to optimize the write process.
                            #  These messages should be ignored.

                        # Target File Data
                        case 1260:
                            # read the message of the current header
                            message_bytes = f.read(JSFTargetFileData.HEADERSIZE)
                            file_message = JSFTargetFileData(message_bytes)

                            # read the image data attached to the message
                            img_bytes = f.read(file_message.target_image_size)
                            file_message.img_jpg = img_bytes

                            self.packets.append(cur_header)
                            self.packets.append(file_message)

                        # NMEA String
                        case 2002:
                            # read the message of the current header
                            message_bytes = f.read(JSFNMEAString.HEADERSIZE)
                            nmea_message = JSFNMEAString(message_bytes)

                            # read the image data attached to the message
                            str_data = f.read(
                                cur_header.msg_size - JSFNMEAString.HEADERSIZE
                            )
                            nmea_message.nmea_string = str_data.decode("ascii")

                            self.packets.append(cur_header)
                            self.packets.append(nmea_message)

                        # Pitch Roll Data
                        case 2020:
                            # read the message of the current header
                            message_bytes = f.read(JSFPitchRollData.MESSAGESIZE)
                            pitch_roll_msg = JSFPitchRollData(message_bytes)

                            self.packets.append(cur_header)
                            self.packets.append(pitch_roll_msg)

                        # Pressure Sensor Reading
                        case 2060:
                            # read the message of the current header
                            message_bytes = f.read(JSFPressureSensorReading.MESSAGESIZE)
                            pressure_msg = JSFPressureSensorReading(message_bytes)

                            self.packets.append(cur_header)
                            self.packets.append(pressure_msg)

                        # Reflection Coefficient
                        case 2071:
                            # read the message of the current header
                            message_bytes = f.read(JSFReflectionCoeff.HEADERSIZE)
                            coeff_msg = JSFReflectionCoeff(message_bytes)
                            coeff_msg.coeff_data = f.read(
                                cur_header.msg_size - JSFReflectionCoeff.HEADERSIZE
                            )

                            self.packets.append(cur_header)
                            self.packets.append(coeff_msg)

                        # Doppler Velocity Log Data
                        case 2080:
                            # read the message of the current header
                            message_bytes = f.read(JSFDopplerVelocityLog.MESSAGESIZE)
                            doppler_velo = JSFDopplerVelocityLog(message_bytes)

                            self.packets.append(cur_header)
                            self.packets.append(doppler_velo)

                        # Situation Comprehensive Message (Version 2)
                        case 2091:
                            # read the message of the current header
                            message_bytes = f.read(
                                JSFSituationComprehensiveMessage.MESSAGESIZE
                            )
                            comprehensive_msg = JSFSituationComprehensiveMessage(
                                message_bytes
                            )

                            self.packets.append(cur_header)
                            self.packets.append(comprehensive_msg)

                        # Cable Counter Data Message
                        case 2100:
                            # read the message of the current header
                            message_bytes = f.read(
                                JSFCableCounterDataMessage.MESSAGESIZE
                            )
                            cable_counter_msg = JSFCableCounterDataMessage(
                                message_bytes
                            )

                            self.packets.append(cur_header)
                            self.packets.append(cable_counter_msg)

                        # Kilometer of Pipe Data
                        case 2101:
                            # read the message of the current header
                            message_bytes = f.read(JSFKilometerOfPipe.MESSAGESIZE)
                            km_pipe = JSFKilometerOfPipe(message_bytes)

                            self.packets.append(cur_header)
                            self.packets.append(km_pipe)

                        # Container Timestamp Message
                        case 2111:
                            # read the message of the current header
                            message_bytes = f.read(JSFContainerTimestampMsg.MESSAGESIZE)
                            time_msg = JSFContainerTimestampMsg(message_bytes)

                            self.packets.append(cur_header)
                            self.packets.append(time_msg)

                        case _:
                            # Skip message bytes and report the error
                            print(
                                f"Message of type: {cur_header.msg_type} is unknown - {cur_header.msg_size} bytes are read without interpretation"
                            )
                            message_bytes = f.read(cur_header.msg_size)

                            unknown_msg = JSFUnknownMessage(cur_header.msg_size, message_bytes, cur_header.msg_type)
                            self.packets.append(cur_header)
                            self.packets.append(unknown_msg)

                else:
                    print(f"File out of sync at {f.tell()}")
                    # TODO: It would be possible to implement an algorithm to work with files where the header is out of sync:
                    # - Search per byte for the next fitting sync
                    # - Check if header can be interpreted, if no search for next sync
                    # - If yes -> use normal case loop
                    raise NotImplementedError

            f.close()

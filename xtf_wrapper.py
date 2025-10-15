from dataclasses import dataclass, field
import pyxtf
from pathlib import Path
import numpy as np


@dataclass
class XTFWrapper:
    file_path: Path
    file_header: pyxtf.XTFFileHeader = pyxtf.XTFFileHeader()
    packets: dict = field(default_factory=dict)
    num_ch: int = 2
    sonar_data: list = field(default_factory=list)

    def __post_init__(self):
        self.file_header, self.packets = pyxtf.xtf_read(str(self.file_path))
        self.num_ch = self.file_header.NumberOfSonarChannels

        # read pings and construct starboard and portside picture
        for ch in range(self.num_ch):
            try:
                self.sonar_data.append(
                    np.flipud(
                        pyxtf.concatenate_channel(
                            self.packets[pyxtf.XTFHeaderType.sonar],
                            file_header=self.file_header,
                            channel=ch,
                            weighted=False,
                        )
                    )
                )
            except:
                ch_data = []
                ch_idx = 0
                for xtf_ping in self.packets[0]:
                    try:
                        ch_data.append(xtf_ping.data[ch])
                    except:
                        ch_data.append(ch_data[-1])
                        print(f"missing ping at position {ch_idx}")
                    ch_idx += 1
                self.sonar_data.append(np.array(ch_data))

        # calculate x axis in m for later plotting
        sec_per_ping = (
            self.packets[pyxtf.XTFHeaderType.sonar][1]
            .ping_chan_headers[0]
            .SecondsPerPing
        )
        self.num_sample_per_ping = (
            self.packets[pyxtf.XTFHeaderType.sonar][1].ping_chan_headers[0].NumSamples
        )
        sos_2 = self.packets[pyxtf.XTFHeaderType.sonar][1].SoundVelocity
        self.x_axis_m = np.linspace(0, sec_per_ping * sos_2, self.num_sample_per_ping)

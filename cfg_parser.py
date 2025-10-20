from enum import Enum
import os
import yaml

GAINSTRAT = Enum("GAINSTRAT", [("BAC", 0), ("EGN", 1)])


class CFGParser:
    cfg: dict
    categories: dict

    def __init__(self):
        self.cfg = {
            "Working dir": "./sidescan_out",
            "Georef dir": "./georef_out",
            "EGN table path": "./sidescan_out/EGN_table.npz",
            "Project filename": "project_info.yml",
            "EGN table name": "EGN_table.npz",
            "Btm chunk size": 1000,
            "Btm def thresh": 0.6,
            "Btm downsampling": 1,
            "Active convert dB": True,
            "Btm equal hist": True,
            "Active pie slice filter": True,
            "Active sharpening filter": False,
            "Active gain norm": True,
            "Active hist equal": True,
            "Slant gain norm strategy": GAINSTRAT.BAC.value,
            "Slant vertical beam angle": 60,
            "Slant nadir angle": 0,
            "Slant active intern depth": False,
            "Slant chunk size": 1000,
            "Slant active use downsampling": True,
            "Slant active multiprocessing": True,
            "Slant num worker": 8,
            "Slant active export proc data": True,
            "Slant active export slant data": True,
            "View reprocess file": False,
            "Img chunk size": 1000,
            "Img include raw data": False,
            "Georef active proc data": True,
            "Georef UTM": True,
            "Georef Navigation": True,
            "Resolution": 0.2,
            "Search Radius": 0.4,
            "Georef active custom colormap": False,
            "Path": [],
            "Meta info": dict(),
            "BAC resolution": 360,
            "EGN table resolution parameters": [360, 2],
            "Bottom line refinement search range": 0.06,
            "Active bottom line refinement": True,
            "Active btm refinement shift by offset": True,
            "Active bottom line smoothing": True,
            "Additional bottom line inset": 0,
        }
        self.categorized_dict = {
            k: dict()
            for k in [
                "0: Main Processing Parameters",
                "1: Bottom Detection",
                "2: Slant and Gain Correction",
                "3: Georeferencing and Viewing",
                "4: Meta info",
            ]
        }

    def save_cfg(self, filepath: os.PathLike, cfg: dict):
        """Save CFG in a structure that makes working with the CLI version more easy."""
        self.categorized_dict["0: Main Processing Parameters"].update(
            [
                CFGParser.get_kv_pair(cfg, "Georef dir"),
                CFGParser.get_kv_pair(cfg, "Active convert dB"),
                CFGParser.get_kv_pair(cfg, "EGN table path"),
                CFGParser.get_kv_pair(cfg, "Active bottom line refinement"),
                CFGParser.get_kv_pair(cfg, "Active btm refinement shift by offset"),
                CFGParser.get_kv_pair(cfg, "Bottom line refinement search range"),
                CFGParser.get_kv_pair(cfg, "Active bottom line smoothing"),
                CFGParser.get_kv_pair(cfg, "Additional bottom line inset"),
                CFGParser.get_kv_pair(cfg, "Active pie slice filter"),
                CFGParser.get_kv_pair(cfg, "Active sharpening filter"),
                CFGParser.get_kv_pair(cfg, "Active gain norm"),
                CFGParser.get_kv_pair(cfg, "Active hist equal"),
                CFGParser.get_kv_pair(cfg, "Slant gain norm strategy"),
                CFGParser.get_kv_pair(cfg, "Slant vertical beam angle"),
                CFGParser.get_kv_pair(cfg, "Slant nadir angle"),
            ]
        )
        self.categorized_dict["1: Bottom Detection"].update(
            [
                CFGParser.get_kv_pair(cfg, "Btm chunk size"),
                CFGParser.get_kv_pair(cfg, "Btm def thresh"),
                CFGParser.get_kv_pair(cfg, "Btm downsampling"),
                CFGParser.get_kv_pair(cfg, "Btm equal hist"),
            ]
        )
        self.categorized_dict["2: Slant and Gain Correction"].update(
            [
                CFGParser.get_kv_pair(cfg, "Slant active intern depth"),
                CFGParser.get_kv_pair(cfg, "Slant chunk size"),
                CFGParser.get_kv_pair(cfg, "Slant active use downsampling"),
                CFGParser.get_kv_pair(cfg, "Slant active multiprocessing"),
                CFGParser.get_kv_pair(cfg, "Slant num worker"),
                CFGParser.get_kv_pair(cfg, "Slant active export proc data"),
                CFGParser.get_kv_pair(cfg, "Slant active export slant data"),
                CFGParser.get_kv_pair(cfg, "BAC resolution"),
                CFGParser.get_kv_pair(cfg, "EGN table resolution parameters"),
            ]
        )
        self.categorized_dict["3: Georeferencing and Viewing"].update(
            [
                CFGParser.get_kv_pair(cfg, "View reprocess file"),
                CFGParser.get_kv_pair(cfg, "Img chunk size"),
                CFGParser.get_kv_pair(cfg, "Img include raw data"),
                CFGParser.get_kv_pair(cfg, "Georef active proc data"),
                CFGParser.get_kv_pair(cfg, "Georef UTM"),
                CFGParser.get_kv_pair(cfg, "Georef Navigation"),
                CFGParser.get_kv_pair(cfg, "Resolution"),
                CFGParser.get_kv_pair(cfg, "Search Radius"),
                CFGParser.get_kv_pair(cfg, "Georef active custom colormap"),
            ]
        )
        self.categorized_dict["4: Meta info"].update(
            [
                CFGParser.get_kv_pair(cfg, "Working dir"),
                CFGParser.get_kv_pair(cfg, "Project filename"),
                CFGParser.get_kv_pair(cfg, "EGN table name"),
                CFGParser.get_kv_pair(cfg, "Path"),
                CFGParser.get_kv_pair(cfg, "Meta info"),
            ]
        )

        try:
            f = open(
                filepath,
                "w",
            )
            yaml.dump(self.categorized_dict, f)
            f.close()
        except:
            print(f"Can't write to {filepath}")

    def load_cfg(self, path: os.PathLike):
        f = open(
            path,
            "r",
        )
        loaded_dict = yaml.safe_load(f)
        f.close()
        if "0: Main Processing Parameters" in dict(loaded_dict).keys():
            # load categorized cfg
            for key in dict(loaded_dict).keys():
                try:
                    for intern_key in loaded_dict[key]:
                        self.cfg[intern_key] = loaded_dict[key][intern_key]
                except:
                    print(f"Couldn't load setting with key: [{key}][{intern_key}]")
        else:
            # load legacy cfg
            for key in dict(loaded_dict).keys():
                try:
                    self.cfg[key] = loaded_dict[key]
                except:
                    print(f"Couldn't load setting with key: {key}")

        return self.cfg

    @staticmethod
    def get_kv_pair(cfg, key):
        return (key, cfg[key])

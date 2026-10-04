"""mechdesign — four-bar linkage and gear-train design tools."""
from .fourbar import FourBar, animate
from .gears import GearTrain, design_train, min_pinion_teeth, parse_meshes, plot_train

__all__ = ["FourBar", "animate", "GearTrain", "design_train", "min_pinion_teeth",
           "parse_meshes", "plot_train"]

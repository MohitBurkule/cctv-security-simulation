"""SurvSim: photorealistic multi-camera, multi-person surveillance simulator."""

from .core.loader import load_scenario
from .core.models import Camera, Environment, Person, Scenario
from .simulator import Simulator

__all__ = ["Camera", "Environment", "Person", "Scenario", "Simulator", "load_scenario"]
__version__ = "0.1.0"

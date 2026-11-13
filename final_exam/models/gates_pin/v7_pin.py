"""V5's independent PE2+ signed gates with appended NMOS/PMOS one-hot inputs."""
from .type_input_pin import MOSDeviceTypeInputPin
from .v5_pin import OPAttentionGateV5Pin


class OPAttentionGateV7Pin(MOSDeviceTypeInputPin, OPAttentionGateV5Pin):
    version = 'v7'

"""V4's shared PE2+ signed gate with an appended NMOS/PMOS one-hot input."""
from .type_input_pin import MOSDeviceTypeInputPin
from .v4_pin import OPAttentionGateV4Pin


class OPAttentionGateV6Pin(MOSDeviceTypeInputPin, OPAttentionGateV4Pin):
    version = 'v6'

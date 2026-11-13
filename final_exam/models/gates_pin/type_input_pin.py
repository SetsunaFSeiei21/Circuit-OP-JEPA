"""Append physical MOS polarity without changing OP or terminal preprocessing."""
import torch
from torch.nn import functional as F


class MOSDeviceTypeInputPin:
    """Input-only extension shared by V6 (V4) and V7 (V5).

    The owner's canonical device type, never its position in a vocabulary,
    selects NMOS=[1,0] or PMOS=[0,1]. Non-MOS/padded owners contribute zeros;
    the inherited layout still protects all their attention edges.
    """

    @property
    def input_dim(self):
        return super().input_dim + 2

    @property
    def architecture_contract(self):
        return dict(super().architecture_contract, device_type_encoding=dict(
            kind='one_hot', categories=['nmos', 'pmos'], width=2,
            placement='after_terminal_roles', normalization='none'))

    def device_type_one_hot(self, batch, *, dtype):
        polarity = self.mos_polarity[batch['owner_device_types']]
        valid = (polarity >= 0) & batch['owner_mask']
        one_hot = F.one_hot(polarity.clamp_min(0), num_classes=2).to(dtype=dtype)
        return torch.where(valid.unsqueeze(-1), one_hot, 0.)

    def pair_inputs(self, batch, values, supported):
        original = super().pair_inputs(batch, values, supported)
        types = self.device_type_one_hot(batch, dtype=values.dtype)
        types = types[:, :, None, None, :].expand(*original.shape[:-1], 2)
        return torch.cat((original, types), dim=-1)

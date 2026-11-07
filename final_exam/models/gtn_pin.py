"""Pin-level GTN with topology-loop discovery separated from pin cliques.

Learned nodes are terminals and external ports. Loops are discovered in the
physical owner-device/net bipartite graph, then projected to participating
terminal endpoints. A same-net pin clique and an intra-device pin clique are
not physical circuit loops. External ports are not OP owners or loop devices.
"""
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from .common import graph_memory, model_config
from .common_pin import PinEmbedding, require_pin_batch
from .gtn import GTNLayer, GTNDecoder


def build_loop_membership_pin(pin_to_device,pin_net_index,pin_mask):
    """CPU bool [B,L,P], using topology only; empty loop slots stay false."""
    import networkx as nx
    if pin_mask.ndim != 2 or pin_to_device.shape != pin_mask.shape or pin_net_index.shape != pin_mask.shape:
        raise ValueError('Pin loop inputs must all have shape [batch, pins]')
    owners=pin_to_device.detach().to('cpu',dtype=torch.long)
    nets=pin_net_index.detach().to('cpu',dtype=torch.long)
    valid=pin_mask.detach().to('cpu',dtype=torch.bool)
    rows=[]
    for owner_row,net_row,live in zip(owners.tolist(),nets.tolist(),valid.tolist()):
        graph=nx.Graph()
        for owner,net,active in zip(owner_row,net_row,live):
            if active and owner>=0:
                if net<0:
                    raise ValueError('An observed physical pin requires a named net identity')
                graph.add_edge(('device',owner),('net',net))
        memberships=set()
        for cycle in nx.minimum_cycle_basis(graph):
            devices={index for kind,index in cycle if kind=='device'}
            cycle_nets={index for kind,index in cycle if kind=='net'}
            if len(devices)<2:
                continue
            members=tuple(i for i,(owner,net,active) in enumerate(zip(owner_row,net_row,live))
                          if active and owner in devices and net in cycle_nets)
            if members:
                memberships.add(members)
        rows.append(sorted(memberships))
    output=torch.zeros(len(rows),max(1,max(map(len,rows),default=0)),pin_mask.shape[1],dtype=torch.bool)
    for b,cycles in enumerate(rows):
        for cycle,members in enumerate(cycles):
            output[b,cycle,list(members)]=True
    return output


class GTNEncoderPin(nn.Module):
    def __init__(self,config,prep):
        super().__init__()
        self.cfg=model_config(config)
        self.embedding=PinEmbedding(self.cfg,prep)
        self.layers=nn.ModuleList(GTNLayer(self.cfg,len(prep['feature_spec']['terminal_names'])**2)
                                  for _ in range(self.cfg.encoder_layers))
        self.register_buffer('loops_enabled',torch.tensor(True,dtype=torch.bool))
        self._loop_cache=OrderedDict()

    def prepare_resident_pin(self,store):
        if 'gtn_loop_membership_by_pin' not in store.static:
            masks=build_loop_membership_pin(store.static['pin_to_device'],
                store.static['pin_net_index'],store.static['device_mask']).to(store.device)
            store.static['gtn_loop_membership_by_pin']=masks.transpose(1,2)
        return store.static['gtn_loop_membership_by_pin'].transpose(1,2)

    def _memberships(self,batch):
        if 'gtn_loop_membership_by_pin' in batch:
            value=batch['gtn_loop_membership_by_pin']
            if value.ndim!=3 or value.shape[:2]!=batch['device_mask'].shape:
                raise ValueError('GTN pin membership must have shape [batch,pins,loops]')
            return value.transpose(1,2).bool()
        # Independent small-batch checkpoint inference; production runner
        # prepares resident tensors before fitting/evaluating each split.
        rows=[]
        for owner,net,valid in zip(batch['pin_to_device'].detach().cpu(),
                                   batch['pin_net_index'].detach().cpu(),
                                   batch['device_mask'].detach().cpu()):
            key=(tuple(owner.tolist()),tuple(net.tolist()),tuple(valid.tolist()))
            if key not in self._loop_cache:
                self._loop_cache[key]=build_loop_membership_pin(owner[None],net[None],valid[None])[0]
                if len(self._loop_cache)>64:
                    self._loop_cache.popitem(last=False)
            else:
                self._loop_cache.move_to_end(key)
            rows.append(self._loop_cache[key])
        count=max(row.shape[0] for row in rows)
        return torch.stack([F.pad(row,(0,0,0,count-row.shape[0])) for row in rows]).to(batch['device_mask'].device)

    def forward(self,batch):
        require_pin_batch(batch)
        nodes=self.embedding(batch)
        memberships=self._memberships(batch)
        for layer in self.layers:
            nodes=layer(nodes,batch,memberships,bool(self.loops_enabled))
        return graph_memory(nodes,batch['device_mask'])


class GTNDecoderPin(GTNDecoder):
    """GTN native mean/64/16/metric MLP, pooling pin memory."""

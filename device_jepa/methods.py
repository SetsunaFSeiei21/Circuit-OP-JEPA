"""One registry and selection contract for experiment entrypoints.

Registration makes a method available, not automatically part of old experiments.
New architectures still need their training/checkpoint implementation and tests.
"""
import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Method:
    workflow: str
    node_granularity: str
    requires_op_at_inference: bool
    description: str
    paired_with: str | None = None
    uses_design_parameters: bool = True
    backbone_reference: str | None = None

    def best_checkpoint(self, name):
        return 'decoder_best.pt' if self.workflow == 'jepa' else f'{name}_best.pt'


METHODS = {
    'zerosim_pin': Method('baseline', 'pin', False, 'Supervised pin-node model'),
    'zerosim_device': Method('baseline', 'device', False, 'Supervised device-node model'),
    'jepa_zerosim': Method('jepa', 'device', False, 'OP alignment pretraining, then frozen-encoder decoder'),
    'zerosim_pin_op': Method('baseline', 'pin', True, 'Pin-node model with continuous DC OP inputs', 'zerosim_pin'),
    'zerosim_device_op': Method('baseline', 'device', True, 'Device-node model with continuous DC OP inputs', 'zerosim_device'),
    'zerosim_pin_op_only': Method('baseline', 'pin', True, 'Topology + DC OP only; no design-parameter inputs',
                                 uses_design_parameters=False, backbone_reference='zerosim_pin_op'),
    'zerosim_device_op_only': Method('baseline', 'device', True, 'Topology + DC OP only; no design-parameter inputs',
                                    uses_design_parameters=False, backbone_reference='zerosim_device_op'),
}

# This compatibility default is deliberately explicit, never tuple(METHODS).
# Registering a new ablation must not silently expand an existing experiment.
DEFAULT_MODELS = ('zerosim_pin', 'zerosim_device', 'jepa_zerosim',
                  'zerosim_pin_op', 'zerosim_device_op')
LEGACY_MODES = ('zerosim_pin', 'zerosim_device', 'jepa_zerosim')
MODES = tuple(METHODS)
BASELINE_MODES = tuple(name for name, method in METHODS.items() if method.workflow == 'baseline')
PIN_MODES = tuple(name for name, method in METHODS.items() if method.node_granularity == 'pin')
OP_MODES = tuple(name for name, method in METHODS.items() if method.requires_op_at_inference)
OP_ONLY_MODES = tuple(name for name, method in METHODS.items() if not method.uses_design_parameters)
OP_PAIRS = tuple((method.node_granularity, method.paired_with, name)
                 for name, method in METHODS.items() if method.paired_with is not None)
MATCHED_BACKBONES = tuple((method.backbone_reference, name)
                         for name, method in METHODS.items() if method.backbone_reference is not None)


def method_spec(name):
    try:
        return METHODS[name]
    except (KeyError, TypeError):
        raise ValueError(f'Unknown model {name!r}; available: {", ".join(METHODS)}') from None


def validate_models(names, *, allow_all=False):
    if not isinstance(names, (list, tuple)) or not names or any(not isinstance(n, str) for n in names):
        raise ValueError('Models must be a nonempty list of model names')
    if allow_all and list(names) == ['all']:
        return list(METHODS)
    if len(names) != len(set(names)):
        raise ValueError('Models must be unique; do not repeat a model')
    for name in names:
        method_spec(name)
    return list(names)


def select_models(options, requested=None, *, default=DEFAULT_MODELS):
    """CLI overrides config; persist the exact resolved selection in run configs."""
    experiment = options.setdefault('experiment', {})
    if not isinstance(experiment, dict):
        raise ValueError('experiment must be an object with an optional models list')
    names = requested if requested is not None else experiment.get('models', list(default))
    selected = validate_models(names)
    experiment['models'] = selected
    return selected


def model_catalog():
    rows = ['Available models (choose with --models NAME [NAME ...]):']
    for name, method in METHODS.items():
        op = 'yes' if method.requires_op_at_inference else 'no'
        rows.append(f'  {name:<24} OP at inference: {op:<3} | {method.description}')
    rows.append('  all                      Explicitly select every registered model')
    return '\n'.join(rows)


class _ListModels(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        print(model_catalog())
        parser.exit()


class _SelectModels(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is not None:
            parser.error('Specify --models (or its --modes alias) only once')
        try:
            selected = validate_models(values, allow_all=True)
        except ValueError as exc:
            parser.error(str(exc))
        setattr(namespace, self.dest, selected)


def add_model_arguments(parser, *, selection=True, comparison=False):
    parser.add_argument('--list-models', action=_ListModels, nargs=0,
                        help='List supported models and exit without reading data or starting jobs')
    if selection:
        parser.add_argument('--models', '--modes', dest='models', nargs='+', action=_SelectModels,
                            metavar='NAME', default=None,
                            help=('Expected comparison cohort; unselected reports are rejected. '
                                  if comparison else
                                  'Only train/evaluate these models; overrides experiment.models. ')
                                 + 'Use all explicitly for every registered model; --modes is a legacy alias.')

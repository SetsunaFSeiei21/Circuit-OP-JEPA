"""Explicit, independent stage budgets for matched training runs."""
from .early_stopping import early_stopping_config
from .methods import method_spec, validate_models


def positive_epochs(training, key):
    value = training.get(key)
    if type(value) is not int or value < 1:
        raise ValueError(f"training.{key} must be an explicit positive integer")
    return value


def smoke_epochs(options):
    options['training'].update(pretrain_epochs=1, decoder_epochs=1, baseline_epochs=2)


def comparison_epoch_budget(options, modes):
    """Fail before running any stage; do not silently rewrite a user's budget."""
    tr = options['training']
    early_stopping_config(tr)
    modes = validate_models(modes)
    pretrain = positive_epochs(tr, 'pretrain_epochs')
    decoder = positive_epochs(tr, 'decoder_epochs')
    baseline = positive_epochs(tr, 'baseline_epochs')
    budgets = {mode: {'pretrain_epochs': pretrain if method_spec(mode).workflow == 'jepa' else 0,
                      'supervised_epochs': decoder if method_spec(mode).workflow == 'jepa' else baseline,
                      'total_epochs': pretrain + decoder if method_spec(mode).workflow == 'jepa' else baseline}
               for mode in modes}
    if len({item['total_epochs'] for item in budgets.values()}) > 1:
        raise ValueError('Incompatible total training budget: '
                         f'JEPA pretrain_epochs ({pretrain}) + decoder_epochs ({decoder}) '
                         f'must equal baseline_epochs ({baseline}); set all three explicitly')
    return budgets

"""Validation-only stopping, independent of exact best-checkpoint selection."""
import math


DEFAULTS = dict(enabled=False, patience=10, min_delta=0.0, min_epochs=1)
SUPERVISED_OBJECTIVE = 'normalized_performance_mse'
PRETRAIN_OBJECTIVE = 'op_alignment_regularized'


def early_stopping_config(training):
    """An omitted setting preserves fixed-epoch legacy configurations."""
    raw = training.get('early_stopping', {})
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError('training.early_stopping must contain only enabled, patience, min_delta, min_epochs')
    config = {**DEFAULTS, **raw}
    if type(config['enabled']) is not bool:
        raise ValueError('training.early_stopping.enabled must be a boolean')
    for name in ('patience', 'min_epochs'):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f'training.early_stopping.{name} must be a positive integer')
    delta = config['min_delta']
    if type(delta) not in (int, float) or not math.isfinite(delta) or delta < 0:
        raise ValueError('training.early_stopping.min_delta must be finite and nonnegative')
    config['min_delta'] = float(delta)
    return config


class EarlyStopping:
    def __init__(self, training, max_epochs, objective):
        self.config = early_stopping_config(training)
        if type(max_epochs) is not int or max_epochs < 1:
            raise ValueError('max_epochs must be a positive integer')
        self.max_epochs, self.objective = max_epochs, objective
        self.epochs_ran = self.best_epoch = self.patience_best_epoch = self.bad_epochs = 0
        self.best_value = self.patience_best_value = math.inf
        self.should_stop = False

    def update(self, value, epoch):
        """Return exact improvement for saving; min_delta affects patience only."""
        if (type(epoch) is not int or epoch != self.epochs_ran + 1
                or epoch > self.max_epochs or self.should_stop):
            raise ValueError('Early-stopping epochs must be sequential and within the budget')
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Early stopping requires a finite validation objective')
        self.epochs_ran = epoch
        improved = value < self.best_value
        if improved:
            self.best_value, self.best_epoch = float(value), epoch
        if self.patience_best_epoch == 0 or value < self.patience_best_value - self.config['min_delta']:
            self.patience_best_value, self.patience_best_epoch = float(value), epoch
        self.bad_epochs = epoch - self.patience_best_epoch
        self.should_stop = (self.config['enabled'] and epoch >= self.config['min_epochs']
                            and self.bad_epochs >= self.config['patience'] and epoch < self.max_epochs)
        return improved

    def announce(self, label):
        if self.should_stop:
            print(f'{label}: early stopping at {self.epochs_ran}/{self.max_epochs} epochs '
                  f'| val {self.objective} best={self.best_value:.6g} at epoch {self.best_epoch} '
                  f'| {self.bad_epochs} epochs without improvement > min_delta', flush=True)

    def summary(self):
        if not self.epochs_ran or (not self.should_stop and self.epochs_ran != self.max_epochs):
            raise ValueError('Cannot finalize an unfinished training stage')
        return dict(version=1, config=dict(self.config), split='val', mode='min', objective=self.objective,
                    max_epochs=self.max_epochs, epochs_ran=self.epochs_ran,
                    best_epoch=self.best_epoch, best_value=self.best_value,
                    patience_best_epoch=self.patience_best_epoch, patience_best_value=self.patience_best_value,
                    bad_epochs=self.bad_epochs, stopped_early=self.should_stop,
                    stop_reason='patience_exhausted' if self.should_stop else 'max_epochs')


def executed_epochs(summary, training, maximum, objective):
    """Verify recorded execution; fixed-epoch old reports remain readable."""
    config = early_stopping_config(training)
    if summary is None:
        if config['enabled']:
            raise ValueError('Missing early-stopping execution provenance')
        return maximum
    if not isinstance(summary, dict):
        raise ValueError('Invalid early-stopping execution provenance')
    if (type(summary.get('version')) is not int or type(summary.get('max_epochs')) is not int
            or not isinstance(summary.get('config'), dict) or set(summary['config']) != set(DEFAULTS)):
        raise ValueError('Invalid early-stopping execution protocol')
    early_stopping_config({'early_stopping': summary['config']})
    expected = dict(version=1, config=config, split='val', mode='min', objective=objective, max_epochs=maximum)
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError('Early-stopping execution differs from the configured validation protocol')
    count, best, reference, bad = (summary.get(key) for key in
        ('epochs_ran', 'best_epoch', 'patience_best_epoch', 'bad_epochs'))
    if (any(type(value) is not int for value in (count, best, reference, bad))
            or not 1 <= reference <= best <= count <= maximum or bad != count - reference):
        raise ValueError('Invalid early-stopping epoch counts')
    for key in ('best_value', 'patience_best_value'):
        if type(summary.get(key)) not in (int, float) or not math.isfinite(summary[key]):
            raise ValueError('Invalid early-stopping validation objective')
    if summary['best_value'] > summary['patience_best_value']:
        raise ValueError('Early-stopping best objective contradicts the patience reference')
    stopped = summary.get('stopped_early')
    if type(stopped) is not bool or stopped != (count < maximum):
        raise ValueError('Early-stopping status contradicts the executed epoch count')
    trigger = max(reference + config['patience'], config['min_epochs'])
    if stopped:
        if (not config['enabled'] or count != trigger
                or summary.get('stop_reason') != 'patience_exhausted'):
            raise ValueError('Invalid early-stopping termination reason or patience')
    elif (summary.get('stop_reason') != 'max_epochs'
          or (config['enabled'] and trigger < maximum)):
        raise ValueError('Training continued beyond its early-stopping rule')
    return count

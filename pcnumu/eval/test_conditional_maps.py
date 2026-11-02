"""Small overflow/edge regression check; run with the transformer_ee Python environment."""
import tempfile
from pathlib import Path
import numpy as np
from eval_by_sample import conditional_maps

with tempfile.TemporaryDirectory() as out:
    x = np.tile([1., 1., 10., 11.], 3)
    y = np.tile([1., 20., 10., 11.], 3)
    conditional_maps(x, y, np.repeat(['1', '2', '3'], 4), out)
    with np.load(Path(out) / 'sample3_pred_vs_true_trueE_counts.npz') as data:
        assert data['slice_counts'].sum() == 3
        assert data['counts'].sum() == 2
        assert data['n_x_outside'] == 1 and data['n_y_outside'] == 1
        assert np.isclose(data['conditional_probability'].sum(), 1.5)
print('PASS: inclusive final edge, overflow accounting and conditional normalization')

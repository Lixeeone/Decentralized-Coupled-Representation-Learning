import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from dcrl.runner import run_experiment, resolve_config
from dcrl.io import load_checkpoint
from dcrl.data import load_data


class ReproductionTests(unittest.TestCase):
    def config(self, output):
        return {"name":"test", "output":str(output), "steps":12, "log_every":3,
                "checkpoint_every":4,"plot":False,"seed":11,"target":"final",
                "data":{"kind":"hd_gmm","n_samples":30,"dim":6,"intrinsic_dim":2,"preprocess":"standardize"},
                "domain":{"kind":"ball"},"dynamics":{"rank":2}}

    def test_resume_is_exact_for_both_solvers(self):
        for method in ["dcrl","adcrl"]:
            with tempfile.TemporaryDirectory() as t:
                t=Path(t)
                full=self.config(t/'full'); full['method']=method
                run_experiment(full)
                split=self.config(t/'split'); split['method']=method; split['steps']=6
                run_experiment(split)
                split['steps']=12
                run_experiment(split,resume=t/'split/checkpoint.npz')
                a,ma=load_checkpoint(t/'full/checkpoint.npz')
                b,mb=load_checkpoint(t/'split/checkpoint.npz')
                for key in a: np.testing.assert_array_equal(a[key],b[key])
                self.assertEqual(ma['rng_state'],mb['rng_state'])

    def test_changed_resume_protocol_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            cfg=self.config(Path(t)/'run'); run_experiment(cfg)
            cfg['dynamics']['eta_w']=.2
            with self.assertRaises(ValueError): run_experiment(cfg,resume=Path(t)/'run/checkpoint.npz')

    def test_outputs_are_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as t:
            cfg=self.config(Path(t)/'run'); run_experiment(cfg)
            with self.assertRaises(FileExistsError): run_experiment(cfg)

    def test_endpoint_reference_matches_final_configuration(self):
        with tempfile.TemporaryDirectory() as t:
            cfg=self.config(Path(t)/'run'); run_experiment(cfg)
            with np.load(Path(t)/'run/final.npz',allow_pickle=False) as f:
                x,q=f['x'],f['q_target']
                cov=x.T@x/len(x)
                self.assertLess(np.linalg.norm(cov@q-q@(q.T@cov@q)),1e-10)

    def test_feature_checksum_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'features.npy'; np.save(path,np.ones((10,3)))
            with self.assertRaises(ValueError):
                load_data({'kind':'features','path':str(path),'sha256':'bad'},0)

    def test_unknown_config_and_distorted_manifold_fail(self):
        with self.assertRaises(ValueError): resolve_config({'eta_w':.2})
        with self.assertRaises(ValueError):
            resolve_config({'data':{'kind':'sphere','preprocess':'standardize'},'domain':{'kind':'sphere'}})

    def test_external_feature_route_matches_synthetic_input(self):
        with tempfile.TemporaryDirectory() as t:
            x=np.arange(30,dtype=float).reshape(10,3)
            path=Path(t)/'features.npy'; np.save(path,x)
            got,info=load_data({'kind':'features','path':str(path),'preprocess':'none'},0)
            np.testing.assert_array_equal(got,x)
            self.assertEqual(info['shape'],[10,3])


if __name__ == '__main__': unittest.main()

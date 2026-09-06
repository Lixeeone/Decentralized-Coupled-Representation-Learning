"""Scientific regression tests: update identities, geometry, and diagnostics."""
import unittest
import numpy as np
from dcrl.dynamics import DCRL, DynamicsConfig, oja_delta
from dcrl.gossip import ADCRL, pairwise_mix
from dcrl.geometry import Domain, make_domain
from dcrl.data import generate, standardize
from dcrl.metrics import (initialize_w, lyapunov, principal_subspace,
    row_orthogonality_error, subspace_alignment_error, canonical_offdiag_mass)
from dcrl.graph import graph_generator, circle_functions, sphere_functions, evaluation_grid


class ScientificTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(19)
        self.x = self.rng.normal(size=(23, 7))
        self.w = initialize_w(3, 7, 5, .8)

    def test_rank_one_accumulation_matches_covariance_equation(self):
        cov = self.x.T @ self.x / len(self.x)
        expected = self.w @ cov - self.w @ cov @ self.w.T @ self.w
        np.testing.assert_allclose(oja_delta(self.x, self.w, block_size=4), expected, atol=1e-14)

    def test_single_sample_update_has_linear_payload_form(self):
        x, w = self.x[:1], self.w
        y = w @ x[0]
        expected = np.outer(y, x[0]) - np.outer(y, y) @ w
        np.testing.assert_allclose(oja_delta(x, w), expected, atol=1e-14)

    def test_scalar_ablation_is_not_matrix_lateral_inhibition(self):
        y = self.x @ self.w.T
        expected = y.T @ self.x / len(self.x) - np.mean(np.sum(y*y, axis=1)) * self.w
        np.testing.assert_allclose(oja_delta(self.x, self.w, "scalar"), expected)
        self.assertGreater(np.linalg.norm(expected - oja_delta(self.x, self.w)), .1)

    def test_synchronous_order_uses_updated_spatial_state(self):
        c = DynamicsConfig(rank=3, eta_x=.01, eta_w=.001)
        domain = Domain("ball", radius=100)
        a = DCRL(self.x, c, domain, seed=2, w0=self.w)
        rng = np.random.default_rng(2)
        xn = self.x + c.eta_x*c.diffusion*c.beta*(self.x @ self.w.T @ self.w) + np.sqrt(2*c.diffusion*c.eta_x)*rng.normal(size=self.x.shape)
        wn = self.w + c.eta_w*oja_delta(xn, self.w)
        a.advance()
        np.testing.assert_allclose(a.x, xn, atol=1e-14)
        np.testing.assert_allclose(a.w, wn, atol=1e-14)

    def test_no_hidden_qr(self):
        a = DCRL(self.x, DynamicsConfig(rank=3), Domain("ball", radius=100), w0=self.w)
        a.advance()
        self.assertGreater(row_orthogonality_error(a.w), .1)

    def test_fixed_stream_ablation_keeps_x_exact(self):
        a = DCRL(self.x, DynamicsConfig(rank=3, spatial=False), Domain("identity"))
        a.advance()
        np.testing.assert_array_equal(a.x, self.x)

    def test_frozen_w_ablation(self):
        a = DCRL(self.x, DynamicsConfig(rank=3, plasticity=False), Domain("ball"), w0=self.w)
        a.advance()
        np.testing.assert_array_equal(a.w, self.w)

    def test_active_identity_domain_is_rejected(self):
        with self.assertRaises(ValueError):
            DCRL(self.x, DynamicsConfig(rank=3), Domain("identity"))

    def test_ball_uses_empirical_center(self):
        x = self.x + 100
        d = make_domain({"kind":"ball"}, x)
        np.testing.assert_allclose(d.center, x.mean(0))
        z = d.retract(x*2)
        self.assertLessEqual(np.linalg.norm(z-d.center,axis=1).max(),d.radius+1e-10)

    def test_sphere_tangent_and_retraction(self):
        x = generate("sphere", n_samples=30)
        d = Domain("sphere", radius=1)
        v = d.tangent(x, self.rng.normal(size=x.shape))
        np.testing.assert_allclose(np.sum(x*v,axis=1),0,atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(d.retract(x+v),axis=1),1,atol=1e-14)

    def test_torus_tangent_and_retraction(self):
        x = generate("torus",n_samples=30)
        d = Domain("torus")
        v = d.tangent(x, self.rng.normal(size=x.shape))
        np.testing.assert_allclose(np.sum(d._normal(x)*v,axis=1),0,atol=1e-14)
        z = d.retract(x+v)
        np.testing.assert_allclose((np.linalg.norm(z[:,:2],axis=1)-2)**2+z[:,2]**2,.7**2,atol=1e-12)

    def test_strip_retraction_preserves_points(self):
        for kind in ["swiss_roll","s_curve","cylinder","hyperboloid"]:
            x = generate(kind,n_samples=20)
            d = Domain(kind,radius=1)
            np.testing.assert_allclose(d.retract(x),x,atol=1e-8,err_msg=kind)

    def test_lyapunov_directional_derivative(self):
        cov = self.x.T @ self.x / len(self.x)
        a = self.w @ self.w.T - np.eye(3)
        exact = -np.trace(a @ a @ self.w @ cov @ self.w.T)
        drift = oja_delta(self.x,self.w)
        h = 1e-6
        numerical = (lyapunov(self.w+h*drift)-lyapunov(self.w-h*drift))/(2*h)
        self.assertLessEqual(exact,0)
        self.assertAlmostEqual(numerical,exact,places=8)

    def test_euler_lyapunov_remainder_is_second_order(self):
        f = oja_delta(self.x,self.w)
        grad = (self.w @ self.w.T-np.eye(3)) @ self.w
        first = np.sum(grad*f)
        residual=[]
        for h in [1e-3,5e-4]:
            residual.append(abs(lyapunov(self.w+h*f)-lyapunov(self.w)-h*first))
        self.assertAlmostEqual(residual[0]/residual[1],4,delta=.02)

    def test_principal_basis_residual(self):
        q, vals, info = principal_subspace(self.x,3,solver="dense")
        np.testing.assert_allclose(self.x.T@self.x@q/len(self.x),q*vals,atol=1e-12)
        self.assertLess(info['eigen_residual'],1e-12)

    def test_iterative_and_dense_targets_agree(self):
        q,_,_=principal_subspace(self.x,3,solver="iterative")
        q2,_,_=principal_subspace(self.x,3,solver="dense")
        self.assertLess(subspace_alignment_error(q.T,q2),1e-10)

    def test_gauge_is_rotation_invariant(self):
        q,_,_=principal_subspace(self.x,3)
        r=initialize_w(3,3,13)
        w=r@q.T
        self.assertLess(subspace_alignment_error(w,q),1e-12)
        self.assertLess(canonical_offdiag_mass(self.x,w,q),1e-12)

    def test_gauge_does_not_diagonalize_arbitrary_latent_covariance(self):
        q,_,_=principal_subspace(self.x,3)
        self.assertGreater(canonical_offdiag_mass(self.x,self.w,q),1e-3)

    def test_rank_loss_cannot_report_perfect_alignment(self):
        q,_,_=principal_subspace(self.x,3)
        w=q.T.copy(); w[2]=w[1]
        self.assertEqual(subspace_alignment_error(w,q),1)
        self.assertTrue(np.isnan(canonical_offdiag_mass(self.x,w,q)))

    def test_zero_energy_decorrelation_is_undefined(self):
        q,_,_=principal_subspace(self.x,3)
        self.assertTrue(np.isnan(canonical_offdiag_mass(self.x*0,self.w,q)))

    def test_pairwise_mixing_preserves_mean_and_contracts(self):
        a,b=self.w,self.w+1
        na,nb=pairwise_mix(a,b,.3)
        np.testing.assert_allclose(na+nb,a+b)
        np.testing.assert_allclose(na-nb,.4*(a-b))

    def test_adcrl_inactive_agents_unchanged(self):
        c=DynamicsConfig(rank=3)
        a=ADCRL(self.x,c,Domain("ball",radius=100),seed=4)
        x,w=a.x.copy(),a.w_agents.copy()
        a.advance(edge=(1,2))
        ids=[i for i in range(len(x)) if i not in (1,2)]
        np.testing.assert_array_equal(a.x[ids],x[ids])
        np.testing.assert_array_equal(a.w_agents[ids],w[ids])
        np.testing.assert_array_equal(a.x[2],x[2])
        self.assertGreater(np.linalg.norm(a.x[1]-x[1]),0)

    def test_gossip_only_population_mean_is_preserved(self):
        a=ADCRL(self.x,DynamicsConfig(rank=3,spatial=False,plasticity=False),Domain("identity"))
        mean=a.w.copy()
        for _ in range(10): a.advance()
        np.testing.assert_allclose(a.w,mean,atol=1e-14)

    def test_graph_constant_function_is_annihilated(self):
        t=np.linspace(0,2*np.pi,512,endpoint=False)
        g=np.linspace(.01,6.2,20)
        out,degrees=graph_generator(t,g,np.ones((512,1)),np.ones((20,1)),np.zeros(512),np.zeros(20),.3)
        self.assertTrue((degrees>0).all())
        np.testing.assert_array_equal(out,np.zeros_like(out))

    def test_circle_fourier_generator_matches_quadrature_limit(self):
        t=np.linspace(0,2*np.pi,100000,endpoint=False)
        # Queries coincide with the uniform grid: symmetric neighborhoods
        # isolate the quadrature/truncation error from off-grid sampling bias.
        g=t[::2500]
        f,u,_=circle_functions(t,1,0,0)
        fq,uq,exact=circle_functions(g,1,0,0)
        out,_=graph_generator(t,g,f,fq,u,uq,.1,beta=0)
        self.assertLess(np.max(abs(out-exact)),.02)

    def test_missing_graph_neighbors_are_not_silent_zeroes(self):
        out,degrees=graph_generator(np.array([0.,1.]),np.array([3.]),np.ones((2,1)),np.ones((1,1)),np.zeros(2),np.zeros(1),.01)
        self.assertEqual(degrees[0],0)
        self.assertTrue(np.isnan(out).all())

    def test_gibbs_half_tilt_has_full_langevin_drift(self):
        t=np.linspace(0,2*np.pi,100000,endpoint=False)
        g=t[::2500]
        f,u,_=circle_functions(t,1,.7,.5)
        fq,uq,exact=circle_functions(g,1,.7,.5)
        out,_=graph_generator(t,g,f,fq,u,uq,.1,beta=.7)
        self.assertLess(np.max(abs(out-exact)),.025)

    def test_sphere_coordinate_laplacian(self):
        x=evaluation_grid("sphere",20)
        f,_,exact=sphere_functions(x,1,0,.5)
        np.testing.assert_allclose(exact[:,:2],-2*f[:,:2])


if __name__ == "__main__":
    unittest.main()

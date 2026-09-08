import math
import torch
from adaptive_vsv import steering_angle, direct_steering_angle, legacy_lambda_sim, solve_lambda_for_angle

def test_geometry_matches_direct():
    for c in (-.999999, -.5, 0., .5, .999999):
        u=torch.tensor([1.,0.], dtype=torch.float64); d=torch.tensor([c, math.sqrt(max(0.,1-c*c))], dtype=torch.float64)
        assert torch.allclose(steering_angle(c,.17), direct_steering_angle(u,d,.17), atol=1e-6)

def test_zero_and_gate():
    assert float(steering_angle(0.,0.)) == 0.
    assert abs(float(legacy_lambda_sim(torch.tensor([-.4]))[0]) - 1.4) < 1e-6
    assert abs(float(legacy_lambda_sim(torch.tensor([.4]))[0]) - 1.) < 1e-6

def test_bisection_and_clamp():
    target=float(steering_angle(.2,.12)); solved,status=solve_lambda_for_angle(.2,target)
    assert abs(solved-.12)<1e-5 and status=="solved"
    assert solve_lambda_for_angle(.2,100.,hi=.3)[1]=="upper_clamped"

from black_scholes.model import BlackScholes

def test_call_price():
    bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
    assert round(bs.price(), 2) == 10.45

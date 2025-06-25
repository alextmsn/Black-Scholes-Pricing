import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class BlackScholes:
    def __init__(self, S, K, T, r, sigma, option_type='call'):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()

        if self.option_type not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")

    def _d1_d2(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def price(self):
        d1, d2 = self._d1_d2()
        if self.option_type == 'call':
            return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)

    def greeks(self):
        d1, d2 = self._d1_d2()
        pdf_d1 = norm.pdf(d1)

        delta = norm.cdf(d1) if self.option_type == 'call' else -norm.cdf(-d1)
        gamma = pdf_d1 / (self.S * self.sigma * np.sqrt(self.T))
        vega = self.S * pdf_d1 * np.sqrt(self.T) / 100
        theta_call = (-self.S * pdf_d1 * self.sigma / (2 * np.sqrt(self.T)) 
                      - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)) / 365
        theta_put = (-self.S * pdf_d1 * self.sigma / (2 * np.sqrt(self.T)) 
                     + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)) / 365
        rho_call = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) / 100
        rho_put = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2) / 100

        theta = theta_call if self.option_type == 'call' else theta_put
        rho = rho_call if self.option_type == 'call' else rho_put

        return {
            'Delta': round(delta, 4),
            'Gamma': round(gamma, 4),
            'Vega': round(vega, 4),
            'Theta': round(theta, 4),
            'Rho': round(rho, 4)
        }

    def plot_option_value(self):
        S_range = np.linspace(0.5 * self.K, 1.5 * self.K, 300)
        payoff = [BlackScholes(S, self.K, self.T, self.r, self.sigma, self.option_type).price()
                  for S in S_range]

        plt.style.use('seaborn-v0_8-muted')
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#f5f5f5')
        
        ax.plot(S_range, payoff, color='#003f5c', lw=2.5, label=f"{self.option_type.capitalize()} Option Value")
        ax.fill_between(S_range, payoff, color='#7a5195', alpha=0.15)

        ax.axvline(self.K, color='crimson', linestyle='--', lw=2, label='Strike Price (K)')

        atm_index = np.abs(S_range - self.K).argmin()
        ax.scatter(S_range[atm_index], payoff[atm_index], color='black', zorder=5)
        ax.annotate('ATM', (S_range[atm_index], payoff[atm_index]), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=10, family='monospace')

        ax.set_title(f"{self.option_type.capitalize()} Option Value vs Underlying (Black-Scholes)", 
                     fontsize=14, weight='bold', family='monospace')
        ax.set_xlabel("Underlying Price (S)", fontsize=12, family='monospace')
        ax.set_ylabel("Option Value", fontsize=12, family='monospace')
        ax.grid(True, alpha=0.3)

        legend = ax.legend(title='Legend', title_fontsize='13', fontsize=11, loc='upper left')
        for text in legend.get_texts():
            text.set_family('monospace')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.show()

    def implied_volatility(self, market_price, tol=1e-5, max_iter=100):
        sigma = 0.2
        for _ in range(max_iter):
            price = BlackScholes(self.S, self.K, self.T, self.r, sigma, self.option_type).price()
            vega = self.S * norm.pdf((np.log(self.S / self.K) + (self.r + 0.5 * sigma**2) * self.T) / 
                                     (sigma * np.sqrt(self.T))) * np.sqrt(self.T)
            diff = price - market_price
            if abs(diff) < tol:
                return round(sigma, 4)
            sigma -= diff / vega
        return None


if __name__ == '__main__':
    bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
    print("Option Price:", bs.price())
    print("Greeks:", bs.greeks())
    bs.plot_option_value()

    # Example of implied volatility estimation
    print("Implied Vol (from market price 10):", bs.implied_volatility(market_price=10))

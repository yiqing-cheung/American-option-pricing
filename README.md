# American-option-pricing

This is my Advanced Derivatives Pricing coursework project. Two pricing methods are included:
#### 1. Crank-Nicolson method
#### 2. Spectral Collocation Method

## Abstract
In this paper, we presents a study on the application of the Spectral Collocation Method for pricing American put options. Our primary objective is to implement this method and evaluate its performance under various conditions. We begin by implementing this algorithm and replicating Table 2 from the influential paper "High Performance American Option Pricing".

The core of our study involves an in-depth analysis of the results obtained from the Spectral Collocation Method. We meticulously evaluate the accuracy, numerical stability, and convergence speed of this method. These factors are examined in the context of varying model parameters: the degree of the polynomial approximation (l, m, n), the spot price (S), interest rate (r), dividend yield (q), and time to maturity (τ). Such an analysis provides a nuanced understanding of the method's effectiveness and reliability across different market conditions. We also measured the stability using the smoothness of the greeks.

Furthermore, we extend our research by implementing the Crank-Nicolson method, a well-established technique in option pricing. This implementation allows us to perform a comparative analysis between the Crank-Nicolson and Spectral Collocation Methods. By juxtaposing these two methods, we shed light on their respective strengths and limitations, offering valuable insights into their applicability in various option pricing scenarios.

This project offers practical insights for financial practitioners interested in applying these techniques in real-world contexts.

## References:
[1] Andersen, Leif BG, Mark Lake, and Dimitri Offengenden. ”High performance American option pricing.”Available at SSRN 2547027 (2015).

[2] FastAmericanOptionPricing: https://github.com/antdvid/FastAmericanOptionPricing

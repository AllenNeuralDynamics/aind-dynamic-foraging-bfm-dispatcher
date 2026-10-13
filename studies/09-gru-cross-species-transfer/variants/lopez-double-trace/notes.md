# López-Yépez (mouse) double-trace baseline

Fit the paper-selected reward-value model with fast and slow choice traces
separately for each mouse on the same odd-session adaptation half, then score
the same even-session test half used by GRU and Bari2019 common Q.

This is an equation-level reproduction. The paper does not release its fitting
code or fully specify initialization, so we use symmetric initial values of 0.5
and disclose that deviation. CPU-only Allen HPC job; never submit to Beaker.

Tracking: dispatcher #159.

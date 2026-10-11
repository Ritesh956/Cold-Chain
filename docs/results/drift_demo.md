# Input-drift check on SYNTHETIC data

Each row scores zone-days (24 h, 1,440 readings) of the synthetic history against the shipped
reference with PSI, flagging 'drift' at PSI >= 0.25 on any feature. Days are split by whether the
simulator injected an anomaly (door left open, compressor drift, power cut) that day. The reference
was built from this same history, so the 'none' rows are the false-alarm rate on data the reference
has seen, not on fresh data. Injected drift is a synthetic edit of the anomaly-free days.

| Injected drift | Zone-days | Flagged | Share | Flagged by |
|---|---|---|---|---|
| none, days with an injected fault | 88 | 46 | 52.3% | compressor_current_a 43, temp_c 25, compressor_on 19 |
| none (normal operation) | 272 | 1 | 0.4% | temp_c 1 |
| temperature sensor +0.5 C | 272 | 272 | 100.0% | temp_c 272 |
| temperature sensor +1.0 C | 272 | 272 | 100.0% | temp_c 272 |
| temperature sensor +2.0 C | 272 | 272 | 100.0% | temp_c 272 |
| compressor current +10 % | 272 | 272 | 100.0% | compressor_current_a 272, temp_c 1 |
| compressor current +25 % | 272 | 272 | 100.0% | compressor_current_a 272, temp_c 1 |
| compressor stuck on | 272 | 272 | 100.0% | compressor_on 272, temp_c 1 |

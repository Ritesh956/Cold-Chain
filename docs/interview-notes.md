# Resume bullets and interview Q&A

Every number below is measured and sits in a model card or doc linked beside it. **Do not add a number
that is not in this repo.** Nearly everything is on synthetic data; say so first, before an interviewer finds it.

One honesty point to settle before using any of this: the project was built with an AI assistant (the
data sheet says so for the independent incident test set). Be ready to say plainly what you directed,
what you checked and what you can explain without the code open. The Q&A below is only useful if you have
read the linked docs and can rebuild the argument yourself.

## Resume bullets (pick three or four)

- Built an end-to-end ML platform for a **simulated** automated cold store: a labelled-fault freezer
  simulator, NATS JetStream ingest into TimescaleDB, model serving from an MLflow registry, a drift
  monitor with Prometheus/Grafana, and a live Next.js dashboard (Python, FastAPI, TypeScript, Docker Compose, CI).
- Designed freezer anomaly detection with **event-level** metrics and showed that adding compressor current to a robust
  z-score catches gradual compressor drift in about 6 hours versus about 3 days with temperature alone
  (12x earlier), and that a label-free compressor-off-duration rule lifts power-cut recall from 0.78 to 0.97
  (event-F1 0.95 vs 0.93 for the baseline) on synthetic data; an isolation forest did
  not beat it. Sanity-checked on a real public series (2 of 3 failures, 0 false alarms/day).
- Trained a regularised LightGBM breach early-warning model (-18 C within 30 min): PR-AUC 0.57 vs 0.31 for a
  current-temperature threshold, with false alarms cut from 0.22 to 0.03 per zone-day, and compared the models
  at matched false-alarm budgets, where it warns 0.96 of breaches >= 5 min ahead vs 0.62 for the threshold at
  0.22 false alarms per zone-day (0.80 at 0.03), while documenting that at its F1-tuned point it misses most
  drift breaches (0.42).
- Fine-tuned DistilBERT for incident text and served it as ONNX int8 (4.0x faster, 4x smaller), while
  reporting honestly that it did **not** beat a TF-IDF baseline (macro-F1 0.733 vs 0.755), on held-out
  template families plus 120 independently written reports.
- Engineered the pipeline for correctness: idempotent at-least-once ingest, a scorer that never skips
  minutes after an ml-api outage, an alert state machine with hysteresis, and one contract-test suite
  run against both the in-memory and Postgres stores.
- Added PSI-based drift monitoring: on synthetic data it flagged 1 of 272 anomaly-free days while catching every
  injected sensor-bias and compressor shift.

## Q&A

**Why is the project built on synthetic data? Isn't that a weakness?**
It is, and the README says so at the top. There is no public freezer-telemetry dataset with labelled faults,
so I built a simulator with a thermal model and injected faults with ground truth, which lets me measure
event-level detection at all. The cost is that the results measure fit to my simulator. I added one real
series (NAB) as a sanity check, and the model cards list what that check cannot show: notably the
compressor-current signal, which no public dataset has.

**Your BERT model did not beat TF-IDF. Why ship it?**
Because the comparison was the point and the result is honest: 0.733 ± 0.014 vs 0.755 macro-F1 on held-out
families. The corpus is template-generated, so a bag-of-words model can reach most of what is learnable;
a transformer should matter on real, messy text, so I tested it: 3,000 real OSHA injury narratives, all of which
should be `safety`. It said safety for 89% (83% for refrigerated-warehouse reports), and some misses carried
scores above 0.9, so the confidence score cannot be trusted to catch them. That is one class only; I have no real
text for the others. I kept it served because the project demonstrates the serving stack (ONNX int8, registry gate), and the
card says when to switch to the baseline.

**Why event-level metrics instead of F1 per minute?**
A door left open for 20 minutes is one incident, not 20 samples. Per-minute metrics reward a detector for
labelling the middle of a long event and hide that it was late or never fired. I measure recall over events,
precision over alarm segments, false alarms per zone-day and detection delay, with confidence intervals. Those are
the numbers an operator feels.

**How do you know the anomaly model is not just learning your simulator?**
I cannot fully rule it out, and I say so. What I did: hard negatives (defrost and routine door openings look
like faults), a threshold chosen on a validation history and evaluated on separate histories, a simple
baseline that ties on F1 so the gain is shown only where it should be (drift latency), and the NAB check. The
12x drift result is evidence for the method, not a field measurement, because I built the current rise in.

**Walk me through what happens when the ml-api is down.**
The scorer keeps a position per (zone, model) in event time. It moves the position only after the ml-api
answered and the alerts were stored, so an outage delays alerts but never skips minutes. On recovery it asks
for every minute since the last position, so a short fault inside the outage is still scored. A 4xx is treated
as the request being wrong (not retried forever); 5xx and timeouts retry. This is covered by tests.

**How do you avoid duplicate readings with NATS?**
JetStream is at-least-once, so the insert is `ON CONFLICT (zone, ts) DO NOTHING` and a message is acked only
after the commit. A failed write NAKs the batch; an undecodable message is terminated and counted so one bad
message cannot block the stream. The same repository contract tests run against the in-memory fake and the real
Postgres in CI, which is what makes the fake trustworthy in other tests.

**What does the drift monitor actually tell you, and what doesn't it?**
It tells you the inputs or output scores moved (PSI against a training reference, and against the model's own
early baseline). It does not measure accuracy: there are no labels. It would miss concept drift that leaves inputs
unchanged. It also flags real faults, since a fault day looks different from the reference: 46 of 88 fault days
were flagged, so a flag is not by itself a model problem. Score drift is relative to deployment, not validation.

**What would you do differently, or do next?**
Fix the incident model's real-text weakness (it was not trained on, and nothing was tuned with, the OSHA text, so a
held-out slice is still clean) and get real text for the other classes; put TLS in front of the services (the API-key auth is only as safe
as the channel it travels on); bring
the full Compose stack up end to end (CI builds the images and runs integration tests, but I have not run the
whole stack in Docker locally); and finish the parts not built: forecasting/slotting, vision, the LLM assistant.

**What is not built?**
Demand forecasting and slotting, computer vision, the LLM assistant, cloud deployment. They are listed in the
README rather than implied.

## Things to be able to explain without notes

1. Why adding compressor current detects drift earlier than temperature alone (the card's ablation: the gain comes from the signal, not the model), and why that is something I built into the simulator.
2. Why LightGBM needs 1,500 minutes of history and what that costs per request.
3. Why dynamic int8 quantisation made batched incident labels depend on their neighbours, and why the service scores one report at a time.
4. PSI in one sentence, and why the cut-offs 0.1 and 0.25 are conventions.
5. Why the registry gate refused a model (see [docs/ml-api.md](ml-api.md)).

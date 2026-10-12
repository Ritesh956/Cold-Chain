# Incident classifier on REAL injury text (OSHA Severe Injury Reports)

Every OSHA report is a workplace injury (hospitalisation, amputation or loss of an eye), so the right label is `safety` for all of them. This is a **one-sided** check: recall on one class, on industrial text. It cannot measure precision or the other three classes (no real text for them), and cold-store reports are a tiny part of it. Model: coldchain-incident v1 (trained on synthetic text only). 105,993 reports with a usable narrative.

| Subset | Reports | Labelled `safety` (95% CI) | Flagged needs-review | Safety share among confident | Labels |
|---|---|---|---|---|---|
| Random sample | 3,000 | **0.887** (0.875-0.898) | 0.268 | 0.971 | safety 2661, delay 159, temperature 95, damage 85 |
| Refrigerated warehousing (NAICS 493120) | 340 | **0.829** (0.786-0.866) | 0.359 | 0.894 | safety 282, damage 31, delay 21, temperature 6 |

Median top-score by predicted label (a ranking signal, not a probability): damage 0.802, delay 0.812, safety 0.993, temperature 0.833

## Reports it did not call safety (random sample of 12)

- predicted **delay** (0.94): A delivery driver was organizing his truck for daily deliveries and fell from the liftgate to the ground. The employee suffered a dislocated/fractured elbow.
- predicted **delay** (0.94): An employee had been detailing batteries. The employee had a physical altercation with a coworker and sustained a right leg fracture.
- predicted **delay** (0.5): At 9:45 p.m. on February 23, 2022, a guard slipped and fell while walking security rounds near a dock door. The employee landed on the ground and suffered a broken pelvis, as well as a soft tissue injury.
- predicted **damage** (0.9): An employee was removing concrete forms and loading them onto a flatbed tractor trailer when his left index finger got caught in the chain rigging that was in use to move the forms.  The load shifted and amputated his le
- predicted **damage** (0.87): On May 15, 2015, an employee picked up an approximately 20-pound box of oil, sat it back down, and experienced pain in his left arm and back. He later found out he pulled a muscle and had a collapsed lung.
- predicted **damage** (0.64): On April 2, 2020, an employee was sending pieces of wood stock through a chop saw's infeed table. The wood jammed, and the sawblade cut the top of the employee's left hand while he tried to clear the machine.
- predicted **temperature** (0.92): An employee was replacing an LED light fixture in the box room, and working to identify why it was not functioning. The employee sustained an electrical shock that resulted in a cardiac event requiring hospitalization.
- predicted **delay** (0.54): An employee was working to set a drive pin on an S-hook so a forklift could pick up a part and bring it to the paint department. The employee's left middle finger was pinched between the pin and the S-hook resulting in a
- predicted **damage** (0.98): An employee was using a forklift to restack pallets. The employee's left index finger was caught between the lift and pallet as the employee was sliding a pallet off resulting in a partial amputation.
- predicted **delay** (0.98): At 6:49 AM, on April 2, 2024, an employee was assembling grinder equipment and installing the cylinder of the equipment when their left index finger became pinched between the cylinder and the grinding equipment. This re
- predicted **delay** (0.92): An employee was delivering packages. He experienced a gradual onset of cramping in his arms and legs, followed by nausea. He was hospitalized for treatment of heat stress and dehydration.
- predicted **temperature** (0.98): An employee was checking a recompression evaporator when a moving fan blade inside the machine lacerated his left ring and little fingers. The employee underwent surgery, and the little finger was medically amputated.

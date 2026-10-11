"""An INDEPENDENT test set: 120 incident reports (30 per class) written by hand, in a narrative style
that does not use the generator's templates or event phrases.

Provenance, stated plainly: these were authored by the AI assistant that built this project, not
collected from a warehouse and not written by the project owner. They are still synthetic. Their
value is distribution shift (different sentence shapes, longer narrative, mixed issues, domain
words used in other ways), which a held-out template family cannot provide. The label is the main
issue the writer is reporting. It is used ONCE, after all modelling decisions were made on the
validation set.
"""

from __future__ import annotations

INDEPENDENT: list[tuple[str, str]] = [
    # ------------------------------------------------------------------ damage
    (
        "damage",
        "Driver backed into the end of the racking and a full pallet of ice cream slid off the beam, cartons everywhere.",
    ),
    (
        "damage",
        "On unloading trailer 5509 we counted nine pallets with the bottom layers squashed and the wrap hanging off in strips.",
    ),
    (
        "damage",
        "Looked fine from the front but the back two rows of boxes were flattened, probably crushed on the road.",
    ),
    (
        "damage",
        "A pallet of prawns came off the forks at the top of the ramp. Half the cartons split open and product is all over the floor.",
    ),
    (
        "damage",
        "Three cases of garlic bread are wet through and the cardboard has gone to mush, bags inside torn.",
    ),
    (
        "damage",
        "The wooden base is snapped in two so the whole stack is sitting on the forks at an angle. Cannot be moved safely, needs re-palletising.",
    ),
    (
        "damage",
        "Customer returned 14 outers from last week's delivery, all with dented corners and tape peeled away.",
    ),
    (
        "damage",
        "Shrink wrap was shredded on one side so the cartons shifted and the top layer is hanging over the edge.",
    ),
    (
        "damage",
        "When we pulled the pallet from rack F-09 the lowest carton had collapsed under the weight above it.",
    ),
    (
        "damage",
        "The crane picked the wrong way and the load hit the upright. Corner of the pallet is mangled and several boxes burst.",
    ),
    ("damage", "Two tubs in every carton cracked, lids popped, seems it was dropped at the supplier end."),
    (
        "damage",
        "Pallet arrived with a big hole punched through the side, looks like a fork tine went through three cartons.",
    ),
    (
        "damage",
        "Boxes at the bottom are bulging and wet, bags have split and peas are loose inside the pallet.",
    ),
    ("damage", "A cage of retail packs fell over in the lane. About 60 packs flattened, a mess to clean up."),
    (
        "damage",
        "Banding strap snapped when the pallet was lifted and the layers fanned out across the floor.",
    ),
    ("damage", "Corner post missing and the carton edges crumpled all the way down one side of the load."),
    ("damage", "Inbound pallet badly misshapen, leaning nearly 20 degrees, boxes at the base collapsed."),
    (
        "damage",
        "Found chewed-up cartons at the base of the stack where the pallet jack ran into them. Several packs inside ripped.",
    ),
    ("damage", "Whole top row of fish fingers crushed because someone stacked a heavier pallet on top."),
    ("damage", "The forks pierced the carton wall during putaway and pizza boxes are sliced open."),
    (
        "damage",
        "Delivery note says damaged pallet. In practice most of the cartons look fine but the wrap is gone and two boxes are torn.",
    ),
    ("damage", "Cardboard is crushed along the whole left face, probably rubbed against the trailer wall."),
    (
        "damage",
        "We lost about a dozen cartons when the mezzanine pallet tipped. They hit the concrete and split.",
    ),
    (
        "damage",
        "Pallet collapsed in the freezer during inventory, boxes broke open and frozen product scattered.",
    ),
    ("damage", "Broken pallet board stuck out of the underside and tore a hole through the carton above it."),
    (
        "damage",
        "Dented tins and cracked cartons across the whole delivery, the driver says it slid in the trailer.",
    ),
    ("damage", "Left corner of the load smashed in. Cartons look like they were rammed by a pallet truck."),
    (
        "damage",
        "The reach truck clipped the pallet and sent two layers sideways. Packaging is torn and some product is ruined.",
    ),
    ("damage", "Bags of chips split inside the cartons so the boxes are full of loose product and ice."),
    ("damage", "Rejected the lot at the gate. Crushed boxes, mangled wrap, one pallet in pieces."),
    # ------------------------------------------------------------- temperature
    (
        "temperature",
        "Freezer 3 has crept up to minus 14 overnight and the alarm has been going since about four.",
    ),
    (
        "temperature",
        "Someone left the loading door open for most of the lunch break and the room is nowhere near setpoint.",
    ),
    (
        "temperature",
        "The pallet of ice cream feels soft in the middle and the probe says minus 8 at the core.",
    ),
    (
        "temperature",
        "Compressor tripped on the main plant, zone is slowly warming and engineers are on the way.",
    ),
    ("temperature", "Data logger printout shows the trailer sat above minus 15 for almost four hours."),
    (
        "temperature",
        "Product temperature check failed. Readings between minus 10 and minus 12 where we need minus 18.",
    ),
    (
        "temperature",
        "The chill room is reading 9 degrees when it should be holding 2. Fan on the evaporator seems dead.",
    ),
    (
        "temperature",
        "Frost has melted on the cartons and re-frozen, classic sign the load has been warm at some point.",
    ),
    ("temperature", "Alarm panel shows a high temperature fault for aisle 6 and it will not reset."),
    ("temperature", "The reefer unit was off when it got to us. Boxes were warm to the touch at the back."),
    (
        "temperature",
        "Air probe at the far end of the store is reading minus 16 and the compressors are running flat out.",
    ),
    (
        "temperature",
        "After the defrost the temperature never came back down, it is still hovering at minus 19.",
    ),
    (
        "temperature",
        "We think the strip curtain was damaged because warm air is flooding in and the room cannot recover.",
    ),
    (
        "temperature",
        "Fish fingers on pallet 2231 have gone soft and some bags are sticking together, needs QA decision.",
    ),
    (
        "temperature",
        "Overnight log shows two excursions to above minus 18, each lasting around forty minutes.",
    ),
    (
        "temperature",
        "Blast freezer failed to pull down the product. Core temp is still minus 6 after the full cycle.",
    ),
    ("temperature", "Warm load. Driver admits the unit ran out of fuel somewhere on the motorway."),
    (
        "temperature",
        "Zone C is up 7 degrees on yesterday and nobody knows why. Sensors agree so it is not a faulty probe.",
    ),
    ("temperature", "The door heater has failed and ice is jamming the door open, cold air pouring out."),
    (
        "temperature",
        "Product was out of the freezer on the dock for too long before putaway, surface feels thawed.",
    ),
    (
        "temperature",
        "Supplier sent a certificate showing storage at minus 12 for two days. Not acceptable for frozen.",
    ),
    (
        "temperature",
        "High temp warning on the monitoring screen for the whole of bay 4. Looks like a refrigerant leak.",
    ),
    (
        "temperature",
        "Pallet was left on the ramp overnight and the cartons are damp with condensation, product is no longer hard.",
    ),
    (
        "temperature",
        "Humidity and temperature both drifting upward in the picking room, setpoint no longer holding.",
    ),
    (
        "temperature",
        "Mid-day reading was minus 11. By the evening it had climbed to minus 9. The plant is struggling.",
    ),
    (
        "temperature",
        "Temperature deviation on the inbound check. Probe in a carton read minus 7 against our limit of minus 15.",
    ),
    (
        "temperature",
        "Thermostat in the cold store seems stuck, the compressor never switches on and the room is warm.",
    ),
    (
        "temperature",
        "Condenser fault, head pressure high, and as a result the freezer cannot hold temperature.",
    ),
    (
        "temperature",
        "Fridge unit setpoint had been changed to minus 5 by mistake and the whole trailer is soft.",
    ),
    (
        "temperature",
        "Beeping all night. Display says minus 13 and the maintenance team have not been called yet.",
    ),
    # ------------------------------------------------------------------- delay
    (
        "delay",
        "Truck was supposed to be here at six and it is now after nine, driver says he is stuck in traffic.",
    ),
    (
        "delay",
        "We will miss the afternoon cutoff because the order is still being picked and half the stock has not arrived.",
    ),
    ("delay", "Booking for door 8 has been pushed back three times, the customer is going to be furious."),
    (
        "delay",
        "Lorry held at customs and we do not know when it will be released. Production is waiting on this.",
    ),
    ("delay", "The system went down for most of the morning so nothing was scanned and everything ran late."),
    ("delay", "Paperwork was not ready so the driver sat on the bay for two hours before we could load."),
    ("delay", "Not enough operators on the late shift, which means the putaway queue is already hours deep."),
    ("delay", "The supplier loaded the wrong trailer and we have to wait for the correct one to be sent."),
    ("delay", "Collection moved to tomorrow because the haulier ran out of driving hours."),
    (
        "delay",
        "Everything is behind. We started the wave forty minutes late and nothing has caught up since.",
    ),
    (
        "delay",
        "Customer order cannot ship until the purchase order mismatch is fixed, ops are waiting for the account manager.",
    ),
    ("delay", "Vehicle broke down on the way in so the delivery will arrive tomorrow at the earliest."),
    ("delay", "Yard is full, trucks queued back out of the gate for more than an hour."),
    (
        "delay",
        "Delay to the replenishment from the reserve racking means pickers are standing around waiting.",
    ),
    ("delay", "They missed their slot because the previous load overran, so now everything knocks on."),
    ("delay", "Trailer arrived on time but there was no free door so it waited for ninety minutes."),
    (
        "delay",
        "Stock for the promotion was due on Monday and has still not arrived. Store deliveries will go out late.",
    ),
    ("delay", "Release of the shipment is held up while the paperwork is checked, no ETA yet."),
    (
        "delay",
        "Outbound loading was slowed by a shortage of pallets so the truck left an hour past its time.",
    ),
    ("delay", "Night team is two people short so the container will not be emptied before the morning."),
    (
        "delay",
        "Driver has not turned up for the booked collection and the carrier is not answering the phone.",
    ),
    ("delay", "We are running about three hours late on all outbound because the conveyor was down."),
    (
        "delay",
        "Slow turnaround today, the whole schedule has slipped and the next three customers will all be late.",
    ),
    ("delay", "Delivery window missed again. The retailer will refuse the truck if it arrives after four."),
    ("delay", "Inbound container delayed at the port, the shipping line has no revised arrival time yet."),
    ("delay", "Pick list was generated late so the team only started at ten instead of seven."),
    (
        "delay",
        "The booking system double-booked the dock, so one carrier has to wait for the other to finish.",
    ),
    ("delay", "Trucks were turned away at the gate until the yard cleared, causing a long hold up."),
    (
        "delay",
        "We need an extension. The order will not be ready by the deadline because stock is still being counted.",
    ),
    ("delay", "Lost almost a full shift to the scanner outage and have not recovered the backlog."),
    # ------------------------------------------------------------------ safety
    (
        "safety",
        "A picker slid on a patch of ice outside the freezer door and landed hard on his hip, ambulance called.",
    ),
    (
        "safety",
        "Reach truck nearly ran over a colleague at the end of the aisle, he had to throw himself against the racking.",
    ),
    (
        "safety",
        "There is a strong smell of ammonia in the plant room corridor, we evacuated the area and rang the engineer.",
    ),
    ("safety", "Employee caught a finger between the pallet and the rack upright and needs stitches."),
    (
        "safety",
        "The fire exit by dock 5 is blocked by two pallets of stock. Cleared it at once and told the team lead.",
    ),
    (
        "safety",
        "Worker showed up to the freezer without his thermal jacket and was shivering badly after half an hour.",
    ),
    ("safety", "Someone was riding on the forks of the truck again. Spoke to him and wrote it up."),
    (
        "safety",
        "Colleague complained of chest pain after lifting heavy cartons, first aider attended and he went home.",
    ),
    (
        "safety",
        "Water leak from the ceiling in the corridor has made the floor dangerously slippery, cone and sign placed.",
    ),
    (
        "safety",
        "A pedestrian walked through the forklift lane while the truck was loaded and the driver had to brake hard.",
    ),
    (
        "safety",
        "Man was trapped in the freezer for several minutes when the door release failed, he was calm but cold.",
    ),
    (
        "safety",
        "One of the racking beams looks bent and a pallet is sitting right on it. Cordoned off the aisle.",
    ),
    (
        "safety",
        "Temp agency worker was not wearing safety boots on the shop floor and had a box land on his foot.",
    ),
    (
        "safety",
        "Gas alarm sounded in the battery charging room. People were moved out and the room ventilated.",
    ),
    ("safety", "Pallet truck handle hit an operator in the stomach when the pallet moved unexpectedly."),
    (
        "safety",
        "Spilled hydraulic oil near the conveyor and no one put up a warning sign, somebody almost slipped.",
    ),
    ("safety", "Employee burned his hand on a hot pipe in the plant room while checking a valve."),
    ("safety", "Forklift drove through the pedestrian walkway at speed and missed a man by a hair."),
    (
        "safety",
        "Fire alarm went off, everyone assembled outside, and it turned out a heater had been left on near cardboard.",
    ),
    ("safety", "Operator got a cut above the eye from a falling strap while opening a cage."),
    (
        "safety",
        "Worker fainted in the minus 25 room and was dragged out by a colleague, ambulance requested.",
    ),
    (
        "safety",
        "Loose cabling by the charging point is a trip hazard, one person has already stumbled on it.",
    ),
    (
        "safety",
        "Someone propped the emergency door open with a pallet. That is a serious breach of fire rules.",
    ),
    (
        "safety",
        "Pallet of cartons overhanging the aisle and two people nearly walked into it, unsafe stacking.",
    ),
    ("safety", "A temp injured his shoulder pulling a heavy pallet jack. First aid log completed."),
    (
        "safety",
        "Slipped on spilled brine near the freezer entrance and cut her knee, needed a plaster and a rest.",
    ),
    (
        "safety",
        "Hearing protection not used in the compressor room despite the signs, noise levels are really high.",
    ),
    (
        "safety",
        "Gate guard was missing from the loading dock edge. A driver nearly stepped off it in the dark.",
    ),
    ("safety", "Truck came out of the freezer without lights and nearly struck a person in the half dark."),
    ("safety", "Colleague hurt his back moving a cage by hand, no lifting equipment was available."),
]

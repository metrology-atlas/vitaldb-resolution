# Timing and availability

The distributions here describe **what** a value was and **when** an event happened. They do not describe
**when a value became known**, which is what matters when a model uses the value as an input.

## Lab timestamps

VitalDB's lab timestamp (`lab_dt`) is a per-specimen time, not the time the result was known and not the
time the blood was drawn:

- every analyte of a panel shares one timestamp to the second;
- panels run on different analysers from one draw are stamped seconds apart, in no fixed order;
- timed against the arterial-line access of the draw, central-lab panels are stamped about 10 minutes or
  more after it (the specimen logged at the lab), and blood gases 3-12 minutes after it.

The result follows after a turnaround that the release does not record. The registry entry for `lab_dt`
therefore carries `available_at: lab_time_plus_turnaround`.

!!! warning "The availability rule"
    No value may be used before the earliest moment it could have been known. Using a lab value at its
    timestamp in a model that predicts forward in time leaks information from the future. Add an assumed
    turnaround before using it, and use the draw time (earlier than the timestamp) only to say what the
    blood reflected.

## Charted times

A charted time such as operation start was typed by a person, possibly late. Its distribution here is of
when the event happened. A model that runs in real time cannot know a charted time before it was charted;
the charting time is not in the release.

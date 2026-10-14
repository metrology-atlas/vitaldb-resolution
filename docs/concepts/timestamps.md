# Charted times and the clock

VitalDB stores times as seconds from the recording start, which is set to 0. The anaesthesia and operation
start and end times were charted by people, to the clock minute, so their true value is uncertain by at
least a minute, and by more where people round to 5 minutes or chart late.

## Reconstructing the clock

A charted time is a clock minute, but the release only gives seconds from the recording start. The
recording start's clock time can be recovered from the admission field: `adm` marks midnight of the
admission day, a whole number of days before the start (seconds from the start). So

```text
recording start clock time = (-adm) mod 86,400 s
event clock time           = (-adm + t) mod 86,400 s
```

`Registry.timestamp_context` does this and returns the event's clock minute. Clock statements are accurate
to about a minute where the recording start was itself truncated to the minute.

## The model

The true event time is modelled as an offset from a device timestamp of the same case that is exact to the
second: the recording start for anaesthesia start and operation start, the recording end for anaesthesia
end and operation end. The offset has a log-spline density, and rounding modes on the clock grid turn the
true clock time into the charted minute:

| Time | Reference | Rounding modes (besides the clock minute) |
|---|---|---|
| Anaesthesia start | recording start | nearest 10 and 5 min; nominal 07:30 and 07:40 (first cases charted at a standard time) |
| Operation start | recording start | nearest 10 and 5 min |
| Anaesthesia end | recording end | rounded up to 30, 10 and 5 min |
| Operation end | recording end | rounded down to 30, 10 and 5 min |

The two start models include a time-of-day prior: the reference window (first case of the morning, day,
evening and night) enters as a covariate.

The response's `extra` gives the recorded clock time, the posterior mean clock time, and the reference
used. Flags mark problems the rules found: `clock_uncertain` (admission not on a whole minute),
`clock_offset_mismatch`, `order_violation` (operation end after anaesthesia end) and `same_time`.

## Imputed values

537 cases have an operation end exactly 598-604 s before the recording end: set from the recording end,
not charted. 130 have an anaesthesia end derived from it. These are returned with status `imputed` and no
distribution, because they carry no information about when the event happened.

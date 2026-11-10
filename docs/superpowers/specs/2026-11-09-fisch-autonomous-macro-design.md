# Fisch Autonomous Macro Design

**Date:** 2026-11-09  
**Status:** Draft for review  
**Source evidence:** `fisch-fishing-flow.md` (single cast/catch cycle analysis)  
**Approach:** Classical computer vision + feedback controllers

## 1. Goal

Build a fully autonomous desktop fishing loop for Roblox Fisch at a **fixed fishing spot**, that **adapts within a session** across different rods, waters, and fish difficulty.

Success for v1 means the bot can repeatedly:

1. Cast with meter-aware release timing
2. Click relocating SHAKE targets of varying size/speed
3. Detect bite / reel entry automatically
4. Control the reel bar with closed-loop hold/release
5. Detect catch/fail and recast when ready

without relying on recorded click coordinate playback.

## 2. Product decisions

| Decision | Choice |
|---|---|
| Autonomy scope | Fixed spot loop + adapt across rods/waters/fish difficulty |
| Reel policy | Safe-first; tighten only when progress/fish tracking look healthy |
| Learning model | Auto-adapt by default; optionally save/load named profiles |
| Code direction | Bigger rethink; open to heavier vision/control tooling |
| Architecture style | Classical CV + modular feedback controllers |

### Out of scope for v1

- Selling fish, bait management, inventory automation
- Walking / boat travel / spot finding
- Audio-based bite detection
- ML-trained end-to-end policies
- Generic keyboard/mouse macro recorder as the primary runtime path

## 3. Architecture

```text
[Capture] → [Perception] → [State Machine] → [Controllers] → [Input]
                 ↑               │                │
            [Profiler] ←─────────┴────────────────┘
                 ↑
              [UI / Hotkeys / Logs]
```

### Modules

| Module | Responsibility |
|---|---|
| Capture | Grab Roblox client ROI at ~30–60 Hz using window-relative coordinates |
| Perception | Detect cast meter, SHAKE, bite/reel UI, fish/bar/progress, result overlays |
| State machine | Own stage transitions; gate which controller may act |
| Controllers | Cast, shake, reel, recovery; interruptible and evidence-driven |
| Profiler | Online estimates + optional profile save/load |
| Input | Mouse move / down / up with cancel support; no blind queued sequences |
| UI / ops | Start, stop, pause, kill switch, profile controls, debug/logging |

### Core rule

Perception decides state. Controllers act only inside the current state.  
**Result and reel evidence always outrank stale SHAKE detections.**

## 4. State machine

| State | Entry evidence | Main action | Exit |
|---|---|---|---|
| `IDLE` | Bot stopped or client unavailable | Keep inputs released | Operator start + client recognized |
| `PRE_CAST` | Ready gameplay view; no fishing UI | Confirm rod / safe cursor | Cast start |
| `CASTING` | Cast input or meter visible | Hold, track meter, release | Cast response / SHAKE / reel |
| `WAITING_SHAKE` | Cast done; no reel yet | Search SHAKE + bite cues | SHAKE found or bite/reel |
| `SHAKING` | Active labeled target | Transactional revalidate + click | Target gone/moved or bite preempts |
| `BITE_DETECTED` | Shake end + reel/`!` cues | Cancel queued shake actions | Reel geometry usable |
| `REEL_INITIALIZING` | Reel entering / stabilizing | Track without forcing control | Reliable tracking evidence |
| `REELING` | Track, bar, fish detected | Adaptive hold/release control | Result, fail, or tracking loss |
| `CATCH_RESULT` | Success/fail notification | Release LMB; log outcome | Result settles |
| `RESET_TO_CAST` | Reel gone; gameplay returns | Confirm readiness | `PRE_CAST` |
| `RECOVERY` | Missing/contradictory evidence | Release inputs; reacquire state | Recognized state or stop |

### Transition priority

1. Operator stop / lost game context
2. Confirmed catch/fail result
3. Active reel interface
4. Bite transition
5. Active SHAKE target
6. Cast / idle detection

## 5. Perception design

Every detector returns structured observations:

```text
{ found, confidence, geometry, age_ms, extras }
```

Never return only a click coordinate.

### Cast meter

- Detect vertical track near character
- Track white fill height and green target band
- Estimate release timing from fill velocity + calibrated input/render latency
- Treat post-release green flash as confirmation, not the release trigger

### SHAKE

- Detect active labeled circular target (shape + label), not fading rings
- Measure radius, center, and time since last successful activation
- Revalidate immediately before clicking
- Abort if bite/reel appears mid-action

### Bite / reel entry

- Fuse cues: SHAKE disappearance + reel track appearance (+ red `!` when visible)
- Do not require a separate hook click
- Begin reel tracking as soon as geometry is stable enough

### Reel geometry

- Track bounds, white bar left/right, fish marker x, progress fraction
- Estimate fish and bar velocities
- Distinguish thin fish marker from wider arrow graphics via shape + temporal continuity
- Avoid fixed absolute desktop coordinates; use client-relative ROI

### Result

- Detect catch/fail text and/or reel exit with completed/failed progress
- Immediately force LMB release
- Result evidence overrides ongoing reel control

## 6. Controllers

### 6.1 Cast controller

1. Place cursor over safe gameplay content
2. Press LMB down
3. Watch meter fill
4. Release when predicted fill reaches green band, accounting for latency
5. Confirm via cast response / SHAKE / reel appearance
6. On failure, enter recovery rather than blind recast spam

Initial prior from evidence video: ~583 ms hold produced a PERFECT cast in one sample. This is a calibration seed, not a constant.

### 6.2 Shake controller (transactional)

1. Detect current target
2. Move into circle interior (off-center acceptable)
3. Revalidate target still active
4. Short click (~50–80 ms)
5. Wait for relocate/disappear acknowledgment
6. If bite/reel wins, abort immediately

**Do not** impose the human ~525 ms average inter-click interval as a game cooldown. That interval includes search and mouse travel.

#### Online shake adaptation

| Observation | Adaptation |
|---|---|
| Smaller radius | Tighter click tolerance; possibly slower mouse approach |
| Larger radius | Looser tolerance; faster acquire |
| Faster relocate cadence | Higher search rate; shorter post-click wait |
| Miss / stale click | Back off, reacquire; never spam the old point |

### 6.3 Reel controller (safe → tighten)

Inputs: fish x, bar bounds, velocities, progress slope, detection confidence.

Modes:

- **Safe (default):** keep larger interior margin; prioritize not draining progress
- **Tighten:** if progress rising steadily and fish remains trackable, allow smaller margins and snappier corrections to chase Perfect Catch
- **Emergency:** if fish near bar edge or progress falling, widen margins and prioritize containment

Control style:

- Hold LMB to drive bar right; release to drift left
- Cursor stays in a safe static gameplay location during reel
- Support short pulses and sustained holds
- Use hysteresis so noisy detections do not flicker hold/release

### 6.4 Catch / reset controller

1. On result: release LMB immediately
2. Wait for ready-state evidence (reel gone, overlays settled, castable view)
3. Transition to `PRE_CAST`
4. On timeout: `RECOVERY`, never unconditional recast

Post-catch delay from the evidence video (~2.9 s to a late click) is **not** a validated recast interval.

## 7. Profiles and online learning

### Default behavior

Start with generic priors. Adapt continuously from detections during the session.

### Online stats to maintain

- Shake radius distribution
- Shake click success rate
- Relocate / reacquire interval
- Cast release latency / meter velocity
- White-bar width
- Fish volatility (velocity / reversal rate)
- Reel progress slope
- Capture-to-input latency estimate

### Optional named profiles

Operator may save a successful session as a named profile (for example by rod or water).

Suggested profile fields (JSON):

```json
{
  "name": "steady-rod-mooswood-night",
  "saved_at": "2026-11-09",
  "cast_hold_ms_prior": 580,
  "cast_latency_ms": 90,
  "shake_radius_px_mean": 110,
  "shake_click_hold_ms": 70,
  "bar_width_ratio": 0.40,
  "fish_volatility": 0.5,
  "reel_safe_margin_ratio": 0.25,
  "reel_tighten_margin_ratio": 0.12
}
```

On load:

1. Seed controllers with saved priors
2. Continue adapting online
3. Allow re-save / overwrite

Profiles improve startup stability; they do not freeze adaptation.

## 8. Recovery and safety

### Recovery

- Contradictory or missing evidence → `RECOVERY`
- Always release synthetic inputs first
- Re-scan for known UI states
- Only resume casting after confirmed ready state
- Stage timeouts escalate to recovery, not forced next-step actions

### Operator safety

- Global kill hotkey: hard stop + release inputs
- Pause / resume
- Fail-streak stop (configurable consecutive failures)
- Optional max runtime stop
- Optional debug overlay: current state, confidence, fish/bar boxes
- Structured logs: timestamps, transitions, detections, inputs

## 9. Relationship to current codebase

Current `Fisch Macro.py` is a prototype with:

- Generic keyboard/mouse recording/playback
- Hardcoded SHAKE template path
- No cast meter tracking
- No closed-loop reel control
- No state machine / profiles / recovery

v1 replaces that runtime model. Useful carry-overs:

- Hotkey-driven start/stop UX concept
- Screenshot click automation idea for SHAKE
- Tkinter (or successor) operator UI shell

Recorded playback remains useful only as a diagnostic/reference aid, not as the autonomy engine.

## 10. Validation plan

### Fixture from existing video

Use the analyzed clip as a regression fixture. Expected detections:

- One successful cast
- 21 SHAKE presses
- One brief unclicked final SHAKE interrupted by bite
- Automatic reel entry
- One perfect catch
- No confirmed second cast in-clip

### Live validation goals

- Multiple consecutive casts at one spot
- At least one harder fish / narrower-feeling reel if available
- Shake size or cadence change across rod/water if available
- Clean stop via kill hotkey
- Recovery after intentional interruption (manual click / menu open)

### Logging for tuning

Compare bot logs against the measured timeline in `fisch-fishing-flow.md` before expanding to broader autonomy claims.

## 11. Phased delivery

1. **Skeleton:** capture + state machine + logging + kill switch
2. **Shake loop:** transactional detector/clicker with size adaptation
3. **Cast controller:** meter tracking + latency calibration
4. **Reel controller:** safe/tighten/emergency hold-release control
5. **Result + reset:** reliable recast loop
6. **Profiles:** save/load priors
7. **Hardening:** multi-fish sessions, fail-streak handling, debug overlay

## 12. Open risks

- Only one analyzed catch cycle exists; harder fish behavior is unproven
- Cast release tolerance window is not measured across repeats
- Reel input-lock duration is unproven
- SHAKE misclick-cancel setting is unknown
- Windowed Roblox + overlays can disturb ROI detection
- Latency differs per machine; priors must be recalibrated

## 13. Non-goals / compliance note

This design is for a personal external vision/input assistant. Roblox rules restrict third-party automation; account risk remains the operator’s responsibility and is outside the engineering scope of this document.

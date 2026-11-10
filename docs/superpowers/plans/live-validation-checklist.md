# Live Validation Checklist

1. Launch Roblox Fisch windowed at a fixed spot with rod equipped
2. `python -m fisch_macro` → Start
3. Confirm casting shows `LMB held` while the vertical meter rises, then `LMB released` near the release threshold. Check `cast_control` entries in `logs/session.jsonl` for held state and increasing fill. If the status says `meter not detected`, the visual cast response still needs verification
4. Confirm SHAKE clicks follow relocating targets and stop on bite
5. Confirm reel hold/release keeps fish roughly inside bar
6. Confirm catch result releases mouse and recasts
7. Press F10 mid-shake and mid-reel: mouse must release immediately
8. Save profile after a good session; restart and load priors
9. Confirm stop after configured consecutive failures

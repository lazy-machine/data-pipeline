# When the gate blocks

1. **Read** the message and `hint`, or run `data-verify explain <report>`. Several problems
   can hide each other (rows injected after scaling blur the scaling signature, for
   instance), so rerun after each fix.
2. **Fix the data step** (split by group, cut by time, fit transforms on train only, drop
   the column) and rerun. That is the expected outcome almost every time.
3. **Only if the finding is a property of the domain**, tune the narrowest knob and say why
   in a comment:

   ```toml
   "duplicates.cross_split" = { max_fraction = 0.001 }  # identical sensor readings across machines are normal
   ```

   Turning a check `"off"` is a last resort.

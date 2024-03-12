# A List of TODO for Scott to Explore

1. Use MJCF and dm_control's [`rodent.py`](https://github.com/google-deepmind/dm_control/blob/main/dm_control/locomotion/walkers/rodent.py), for better reducing the observation space of the rodent. 

   - [ ] learn about [MJCF](https://github.com/google-deepmind/dm_control/blob/main/dm_control/mjcf/README.md)
   - [ ] modify the observation space

2. Use the new observation space to have better criterion for the healthy termination.

    - [ ] Better Training Performance

3. Fix the Rodent Model XML file to make it more efficient

    - [ ] Understand the number of collision points and eliminate any unnecessary contacts.

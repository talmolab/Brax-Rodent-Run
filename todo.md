# A List of TODO to Explore

1. Use MJCF and dm_control's [`rodent.py`](https://github.com/google-deepmind/dm_control/blob/main/dm_control/locomotion/walkers/rodent.py), for better reducing the observation space of the rodent. 

   - [ ] learn about [MJCF](https://github.com/google-deepmind/dm_control/blob/main/dm_control/mjcf/README.md)
   - [ ] modify the observation space

2. Use the new observation space to have better criterion for the healthy termination.

    - [ ] Better Training Performance

3. Fix the Rodent Model XML file to make it more efficient

    - [ ] Understand the number of collision points and eliminate any unnecessary contacts.



## Adapt the MJCF in brax and in dm_control

> TODO: We can adapt the `rodent.py`'s mjcf system to brax, to allow it to interact with the brax system.

- The MJCF in brax is pretty bare-bone, and only support loading the xml output form the mjcf package
- if we want full flexibility of quick access to joints and contact, we still need to import mjcf from dm_control, which introduce a whole new sets of dependencies.
    - better logics for contact termination, terminate when the system is unhealthy
    - simplified observation space other than the huge array of joint angles and positions

**How-to:**
- Insert the `dm_control.rodent()` (loaded by mjcf) to the brax training system. 
    - _Challenges_: we need to bind the physics to `Mjx_model` in order to pass the simulation data to dm_control.

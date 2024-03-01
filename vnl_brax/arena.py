from dm_control import mjcf as mjcf_dm
from dm_control import composer
from dm_control.locomotion.examples import basic_rodent_2020
from dm_control.composer.variation import distributions
from dm_control.locomotion.arenas import corridors as corr_arenas
from dm_control.locomotion.tasks import corridors as corr_tasks
from dm_control.locomotion.walkers import rodent, ant
# from dm_control import viewer
from dm_control import mujoco as mujoco_dm

'''Arerna Class Intergrated from dm_control'''

# Define inheritence relationships from dm_control
class Gap_Vnl(corr_arenas.GapsCorridor):
    def _build(self, corridor_width, corridor_length, visible_side_planes, aesthetic, platform_length, gap_length):
        super()._build(corridor_width=corridor_width,
                       corridor_length=corridor_length,
                       visible_side_planes=visible_side_planes,
                       aesthetic = aesthetic,
                       platform_length = platform_length,
                       gap_length = gap_length)
    
    def regenerate(self, random_state):
        super().regenerate(random_state)

# Task now just serve as a wrapper
class Task_Vnl(corr_tasks.RunThroughCorridor):
    def __init__(self,
               walker,
               arena,
               walker_spawn_position):
        
        # we don't really need the rest of the reward setup in dm_control, just how the walker is attached to the arena
        spawn_site =  arena._mjcf_root.worldbody.add('site', pos = walker_spawn_position)
        self._arena = arena
        self._walker = walker
        self._walker.create_root_joints(self._arena.attach(self._walker, attach_site=spawn_site)) # customize starting environment

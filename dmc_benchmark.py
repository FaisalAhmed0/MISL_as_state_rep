DOMAINS = [
    'walker',
    'quadruped',
    'jaco',
    'cheetah'.
]


CHEETAH_TASKS = [
    'cheetah_run'
]

HOPPER_TASKS = [
        'hopper_hop',
        'hopper_stand'
]

WALKER_TASKS = [
    'walker_stand',
    'walker_walk',
    'walker_run',
    'walker_flip',
    
]

QUADRUPED_TASKS = [
    'quadruped_walk',
    'quadruped_run',
    'quadruped_stand',
    'quadruped_jump',
    
]

JACO_TASKS = [
    'jaco_reach_top_left',
    'jaco_reach_top_right',
    'jaco_reach_bottom_left',
    'jaco_reach_bottom_right',
    
]

TASKS = WALKER_TASKS + QUADRUPED_TASKS + JACO_TASKS + CHEETAH_TASKS + HOPPER_TASKS

PRIMAL_TASKS = {
    'walker': 'walker_stand',
    'jaco': 'jaco_reach_top_left',
    'quadruped': 'quadruped_walk',
    'cheetah': 'cheetah_run',
    'hopper': 'hopper_hop'
}
DOMAINS = [
    'walker',
    'quadruped',
    'jaco',
    'cheetah',
    'hopper',
    'humanoid',
    'ball_in_cup',
    'stacker'
]

STACKER_TASKS = [
    'stacker_stack_2',
    'stacker_stack_4'
]

BALL_IN_CUP_TASKS = [
    'ball_in_cup_catch'
]

HUMANOID_TASKS = [
    'humanoid_stand',
    'humanoid_run',
    'humanoid_walk'
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

TASKS = WALKER_TASKS + QUADRUPED_TASKS + JACO_TASKS + CHEETAH_TASKS + HOPPER_TASKS + HUMANOID_TASKS

PRIMAL_TASKS = {
    'walker': 'walker_stand',
    'jaco': 'jaco_reach_top_left',
    'quadruped': 'quadruped_walk',
    'cheetah': 'cheetah_run',
    'hopper': 'hopper_hop',
    'humanoid': 'humanoid_stand',
    'ball_in_cup':'ball_in_cup_catch',
    'stacker': 'stacker_stack_2'
}
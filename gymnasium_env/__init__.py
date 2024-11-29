from gymnasium.envs.registration import register

register(
    id='ChordWorldEnv-v0',
    entry_point='gymnasium_env.envs.chord_world:ChordWorldEnv',
)

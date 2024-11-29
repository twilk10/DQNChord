from gymnasium.envs.registration import register

register(
    id='gymnasium_env/ChordWorldEnv-v0',
    entry_point='gymnasium_env.envs.chord_world:ChordWorldEnv',
)

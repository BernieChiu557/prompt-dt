from gymnasium.envs.registration import register

register(
     id="Snake-v0",
     entry_point="nervenet_envs.envs:SnakeEnv",
     max_episode_steps=1000,
)

register(
     id="SnakeDir-v0",
     entry_point="nervenet_envs.envs:SnakeDirEnv",
     max_episode_steps=1000,
)

register(
     id="NerveNetCentipede-v0",
     entry_point="nervenet_envs.envs:NerveNetCentipedeEnv",
     max_episode_steps=1000,
)

register(
     id="NerveNetCentipedeDir-v0",
     entry_point="nervenet_envs.envs:NerveNetCentipedeDirEnv",
     max_episode_steps=1000,
)
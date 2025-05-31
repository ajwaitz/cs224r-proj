# 224r final proj

# commands
```
python ppo.py \
    --seed 1 \
    --env-id CartPole-v0 \
    --total-timesteps 50000 \
    --track \
    --wandb_project_name cs224r-proj \
    --wandb_entity ajwaitz \
    --capture_video
```

# Generate Trajectories
```
python ppo_mlp_expert.py \
    --env-id "CartPole-v1" \
    --model_path "cartpole_agent.pth" \
    --model_type "mlp" \
    --intermediate-size 64 \
    --num-expert-episodes 1000 \
    --seed 1
```


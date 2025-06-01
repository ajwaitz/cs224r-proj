import numpy as np
import os
import pickle
import time
import torch
import wandb

from collections import deque
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from buffer import Buffer
from model import ActorCriticModel
from utils import batched_index_select, create_env, polynomial_decay, process_episode_info
from worker import Worker

from tqdm import tqdm

class PPOTrainer:
    def __init__(self, config:dict, run_id:str="run", device:torch.device=torch.device("cpu"), resume_from_checkpoint:str=None) -> None:
        """Initializes all needed training components.

        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            run_id {str, optional} -- A tag used to save Tensorboard Summaries and the trained model. Defaults to "run".
            device {torch.device, optional} -- Determines the training device. Defaults to cpu.
            resume_from_checkpoint {str, optional} -- Path to checkpoint to resume from. Defaults to None.
        """
        # Set members
        self.config = config
        self.device = device
        self.run_id = run_id
        self.num_workers = config["n_workers"]
        self.lr_schedule = config["learning_rate_schedule"]
        self.beta_schedule = config["beta_schedule"]
        self.cr_schedule = config["clip_range_schedule"]
        self.memory_length = config["transformer"]["memory_length"]
        self.num_blocks = config["transformer"]["num_blocks"]
        self.embed_dim = config["transformer"]["embed_dim"]

        # Checkpointing configuration
        self.checkpoint_frequency = config.get("checkpoint_frequency", 100)  # Save every N updates
        self.keep_best_checkpoint = config.get("keep_best_checkpoint", True)
        self.max_checkpoints = config.get("max_checkpoints", 5)  # Keep only last N checkpoints
        self.best_reward = float('-inf')
        self.start_update = 0

        # Initialize members for BC data
        self.bc_obs_all = None
        self.bc_actions_all = None

        # Initialize DAGGER components
        self.expert_model = None
        self.dagger_dataset_obs = []
        self.dagger_dataset_actions = []
        self.dagger_iteration = 0

        bc_config = self.config.get("bc_pretraining", {})
        if bc_config.get("enabled", False):
            self._load_and_prepare_bc_data(bc_config)

        # Load expert model for DAGGER if enabled
        dagger_config = self.config.get("dagger", {})
        if dagger_config.get("enabled", False):
            self._load_expert_model(dagger_config)

        # Initialize wandb
        wandb.init(project="cs224r-proj", entity="ajwaitz", name=run_id, config=config)
        
        # Setup Tensorboard Summary Writer
        if not os.path.exists("./summaries"):
            os.makedirs("./summaries")
        timestamp = time.strftime("/%Y%m%d-%H%M%S" + "/")
        self.writer = SummaryWriter("./summaries/" + run_id + timestamp)

        # Init dummy environment to retrieve action space, observation space and max episode length
        print("Step 1: Init dummy environment")
        dummy_env = create_env(self.config["environment"])
        observation_space = dummy_env.observation_space
        self.action_space_shape = (dummy_env.action_space.n,)
        self.max_episode_length = dummy_env.max_episode_steps
        dummy_env.close()

        # Init buffer
        print("Step 2: Init buffer")
        self.buffer = Buffer(self.config, observation_space, self.action_space_shape, self.max_episode_length, self.device)

        # Init model
        print("Step 3: Init model and optimizer")
        self.model = ActorCriticModel(self.config, observation_space, self.action_space_shape, self.max_episode_length).to(self.device)
        self.model.train()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr_schedule["initial"])

        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)

        # Init workers
        print("Step 4: Init environment workers")
        self.workers = [Worker(self.config["environment"]) for w in range(self.num_workers)]
        self.worker_ids = range(self.num_workers)
        self.worker_current_episode_step = torch.zeros((self.num_workers, ), dtype=torch.long)
        # Reset workers (i.e. environments)
        print("Step 5: Reset workers")
        for worker in self.workers:
            worker.child.send(("reset", None))
        # Grab initial observations and store them in their respective placeholder location
        self.obs = np.zeros((self.num_workers,) + observation_space.shape, dtype=np.float32)
        for w, worker in enumerate(self.workers):
            self.obs[w] = worker.child.recv()

        # Setup placeholders for each worker's current episodic memory
        print("--- ", observation_space.shape, " ---")
        self.observation_space_dim = observation_space.shape[0]
        self.memory = torch.zeros((self.num_workers, self.max_episode_length, self.num_blocks, self.observation_space_dim), dtype=torch.float32)
        # Generate episodic memory mask used in attention
        self.memory_mask = torch.tril(torch.ones((self.memory_length, self.memory_length)), diagonal=-1)
        """ e.g. memory mask tensor looks like this if memory_length = 6
        0, 0, 0, 0, 0, 0
        1, 0, 0, 0, 0, 0
        1, 1, 0, 0, 0, 0
        1, 1, 1, 0, 0, 0
        1, 1, 1, 1, 0, 0
        1, 1, 1, 1, 1, 0
        """         
        # Setup memory window indices to support a sliding window over the episodic memory
        repetitions = torch.repeat_interleave(torch.arange(0, self.memory_length).unsqueeze(0), self.memory_length - 1, dim = 0).long()
        self.memory_indices = torch.stack([torch.arange(i, i + self.memory_length) for i in range(self.max_episode_length - self.memory_length + 1)]).long()
        self.memory_indices = torch.cat((repetitions, self.memory_indices))
        """ e.g. the memory window indices tensor looks like this if memory_length = 4 and max_episode_length = 7:
        0, 1, 2, 3
        0, 1, 2, 3
        0, 1, 2, 3
        0, 1, 2, 3
        1, 2, 3, 4
        2, 3, 4, 5
        3, 4, 5, 6
        """
        # Log model architecture to wandb
        wandb.watch(self.model)

    def _load_and_prepare_bc_data(self, bc_config: dict):
        expert_data_path = bc_config.get("expert_data_path")
        if not expert_data_path:
            print("WARNING: BC pre-training enabled but 'expert_data_path' not specified. Skipping BC.")
            self.config["bc_pretraining"]["enabled"] = False # Disable to avoid errors
            return

        # with open(expert_data_path, "rb") as f:
        #     trajectories = pickle.load(f)

        print(f"Loading expert trajectories for BC from {expert_data_path}...")
        # try:
        import json
        with open("/home/waitz/cs224r-proj/trajectories/mlp/CartPole-v1__ppo_mlp_expert__mlp__expert_trajectories_1000eps.json", "r") as f:
        # with open(expert_data_path.replace(".pkl", ".json"), "r") as f:
            trajectories = json.load(f)
        print(f"Successfully loaded trajectories from JSON file: {expert_data_path}")
        

        # print(f"Loading expert trajectories for BC from {expert_data_path}...")
        # try:
        #     trajectories = torch.load(expert_data_path, map_location="cpu") # Load to CPU first
        # except (pickle.UnpicklingError, TypeError, AttributeError, RuntimeError) as e_torch:
        #     print(f"torch.load failed ({e_torch}), attempting pickle.load...")
        #     try:
        #         with open(expert_data_path, "rb") as f:
        #             trajectories = pickle.load(f)
        #     except Exception as e_pickle:
        #         print(f"ERROR: Failed to load trajectories with torch.load and pickle.load from {expert_data_path}: {e_pickle}")
        #         self.config["bc_pretraining"]["enabled"] = False
        #         return
        
        obs_list = []
        actions_list = []
        for traj in trajectories:
            obs_list.append(torch.tensor(traj["observations"], dtype=torch.float32))
            actions_list.append(torch.tensor(traj["actions"], dtype=torch.long))

        self.bc_obs_all = torch.cat(obs_list, dim=0).to(self.device)
        self.bc_actions_all = torch.cat(actions_list, dim=0).to(self.device)
        print(f"Successfully loaded {self.bc_obs_all.shape[0]} expert (observation, action) pairs for BC.")

    def _run_bc_pretraining(self):
        bc_config = self.config["bc_pretraining"]
        epochs = bc_config.get("epochs", 10)
        batch_size = bc_config.get("batch_size", 64)
        bc_learning_rate = bc_config.get("learning_rate", self.lr_schedule["initial"]) 

        if self.bc_obs_all is None or self.bc_actions_all is None:
            print("BC data not available. Skipping BC pre-training.")
            return

        print(f"Starting Behavioral Cloning pre-training for {epochs} epochs...")
        self.model.train() # Ensure model is in training mode

        # Temporarily set learning rate for BC if different
        original_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        for pg in self.optimizer.param_groups:
            pg["lr"] = bc_learning_rate

        num_samples = self.bc_obs_all.shape[0]
        num_batches_per_epoch = (num_samples + batch_size - 1) // batch_size

        for epoch in range(epochs):
            permutation = torch.randperm(num_samples, device=self.device)
            epoch_total_loss = 0.0
            
            batch_iterator = range(0, num_samples, batch_size)
            progress_bar = tqdm(batch_iterator, 
                                desc=f"BC Epoch {epoch + 1}/{epochs}", 
                                total=num_batches_per_epoch, 
                                leave=True) 

            for i in progress_bar:
                indices = permutation[i : i + batch_size]
                obs_batch = self.bc_obs_all[indices]
                actions_batch = self.bc_actions_all[indices]
                current_actual_batch_size = obs_batch.shape[0]

                # --- CRITICAL: Constructing Inputs for your ActorCriticModel ---
                # Your model expects: model(obs, sliced_memory, memory_mask, memory_indices)
                # Expert trajectories (s, a) from an MLP agent don't have 'sliced_memory' etc.
                # We need to provide "dummy" or "default" values for these memory components,
                # assuming BC focuses on the current observation. This typically means simulating
                # the state of memory at the beginning of an episode (step 0).

                # 1. Dummy `sliced_memory` (e.g., zeros):
                #    Shape: (batch_size, memory_length, num_blocks, obs_dim_for_memory)
                #    The `self.observation_space_dim` is used for `self.memory` in __init__.
                dummy_sliced_memory = torch.zeros(
                    current_actual_batch_size, self.memory_length, self.num_blocks, self.observation_space_dim,
                    device=self.device, dtype=torch.float32
                )

                # 2. `memory_mask` for step 0:
                #    `self.memory_mask` is (max_episode_length, memory_length, memory_length)
                #    Use the mask for the first step of an episode.
                mask_for_step0 = self.memory_mask[0]  # Shape: (memory_length, memory_length)
                batch_memory_mask = mask_for_step0.unsqueeze(0).repeat(current_actual_batch_size, 1, 1)

                # 3. `memory_indices` for step 0:
                #    `self.memory_indices` is (max_episode_length, memory_length)
                #    Use indices for the first step.
                indices_for_step0 = self.memory_indices[0] # Shape: (memory_length)
                batch_memory_indices = indices_for_step0.unsqueeze(0).repeat(current_actual_batch_size, 1)
                # --- End of Critical Input Construction ---

                policy_distributions_list, _, _ = self.model(
                    obs_batch, dummy_sliced_memory, batch_memory_mask, batch_memory_indices
                )

                # Calculate BC loss (Negative Log Likelihood)
                log_probs = policy_distributions_list[0].log_prob(actions_batch)
                bc_loss = -log_probs.mean()

                self.optimizer.zero_grad()
                bc_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config["max_grad_norm"])
                self.optimizer.step()

                epoch_total_loss += bc_loss.item()
                progress_bar.set_postfix({"loss": f"{bc_loss.item():.4f}"})
            
            progress_bar.close() # Explicitly close the progress bar for the current epoch

            avg_epoch_loss = epoch_total_loss / num_batches_per_epoch if num_batches_per_epoch > 0 else 0.0
            # The print statement below will now follow the completed progress bar for the epoch
            print(f"BC Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_epoch_loss:.4f}") 
            
            if wandb.run:
                 wandb.log({"bc_pretraining/avg_epoch_loss": avg_epoch_loss, "bc_pretraining/epoch": epoch + 1})
            self.writer.add_scalar("bc_pretraining/avg_epoch_loss", avg_epoch_loss, epoch + 1)

        # Restore original PPO learning rates
        for i, pg in enumerate(self.optimizer.param_groups):
            pg["lr"] = original_lrs[i]
        print("Behavioral Cloning pre-training finished.")

    def _load_expert_model(self, dagger_config: dict):
        """Load expert model for DAGGER training."""
        expert_model_path = dagger_config.get("expert_model_path")
        expert_model_type = dagger_config.get("expert_model_type", "mlp")
        
        if not expert_model_path:
            print("WARNING: DAGGER enabled but 'expert_model_path' not specified. Disabling DAGGER.")
            self.config["dagger"]["enabled"] = False
            return

        print(f"Loading expert model for DAGGER from {expert_model_path}...")
        
        try:
            if expert_model_type == "mlp":
                # Load MLP expert model
                import gymnasium as gym
                from ppo_mlp_expert import MLPAgent
                
                # Create dummy environment to get spaces
                dummy_env = create_env(self.config["environment"])
                
                # Create a mock envs object for MLPAgent
                class MockEnvs:
                    def __init__(self, observation_space, action_space):
                        self.single_observation_space = observation_space
                        self.single_action_space = action_space
                
                mock_envs = MockEnvs(dummy_env.observation_space, dummy_env.action_space)
                dummy_env.close()
                
                # Load expert model
                self.expert_model = MLPAgent(mock_envs, intermediate_size=dagger_config.get("expert_intermediate_size", 64))
                self.expert_model.load_state_dict(torch.load(expert_model_path, map_location=self.device))
                self.expert_model.to(self.device)
                self.expert_model.eval()
                
                print(f"Successfully loaded MLP expert model from {expert_model_path}")
                
            elif expert_model_type == "transformer":
                # Load transformer expert model (same architecture as current model)
                expert_state_dict, expert_config = pickle.load(open(expert_model_path, "rb"))
                
                # Create expert model with same architecture
                dummy_env = create_env(self.config["environment"])
                self.expert_model = ActorCriticModel(expert_config, dummy_env.observation_space, 
                                                   (dummy_env.action_space.n,), dummy_env.max_episode_steps)
                self.expert_model.load_state_dict(expert_state_dict)
                self.expert_model.to(self.device)
                self.expert_model.eval()
                dummy_env.close()
                
                print(f"Successfully loaded transformer expert model from {expert_model_path}")
                
            else:
                raise ValueError(f"Unsupported expert model type: {expert_model_type}")
                
        except Exception as e:
            print(f"ERROR: Failed to load expert model from {expert_model_path}: {e}")
            self.config["dagger"]["enabled"] = False
            self.expert_model = None

    def _query_expert_actions(self, observations):
        """Query expert model for actions on given observations."""
        if self.expert_model is None:
            raise ValueError("Expert model not loaded")
            
        expert_model_type = self.config["dagger"].get("expert_model_type", "mlp")
        
        with torch.no_grad():
            if expert_model_type == "mlp":
                # For MLP expert, just use current observations
                obs_tensor = torch.tensor(observations, dtype=torch.float32).to(self.device)
                actions, _, _, _ = self.expert_model.get_action_and_value(obs_tensor)
                return actions.cpu().numpy()
                
            elif expert_model_type == "transformer":
                # For transformer expert, need to provide memory components
                batch_size = observations.shape[0]
                obs_tensor = torch.tensor(observations, dtype=torch.float32).to(self.device)
                
                # Create dummy memory components for expert queries
                dummy_memory = torch.zeros(
                    batch_size, self.memory_length, self.num_blocks, self.observation_space_dim,
                    device=self.device, dtype=torch.float32
                )
                dummy_mask = self.memory_mask[0].unsqueeze(0).repeat(batch_size, 1, 1)
                dummy_indices = self.memory_indices[0].unsqueeze(0).repeat(batch_size, 1)
                
                policy, _, _ = self.expert_model(obs_tensor, dummy_memory, dummy_mask, dummy_indices)
                actions = policy[0].sample()
                return actions.cpu().numpy()
            
            else:
                raise ValueError(f"Unsupported expert model type: {expert_model_type}")

    def _run_dagger_iteration(self):
        """Run one iteration of DAGGER training."""
        dagger_config = self.config["dagger"]
        collect_episodes = dagger_config.get("collect_episodes_per_iteration", 10)
        
        print(f"Starting DAGGER iteration {self.dagger_iteration + 1}")
        print(f"Collecting {collect_episodes} episodes with current policy...")
        
        # Collect trajectories using current policy
        collected_obs = []
        collected_expert_actions = []
        episodes_collected = 0
        
        # Reset environments
        for worker in self.workers:
            worker.child.send(("reset", None))
        
        # Get initial observations
        current_obs = np.zeros((self.num_workers,) + (self.observation_space_dim,), dtype=np.float32)
        for w, worker in enumerate(self.workers):
            current_obs[w] = worker.child.recv()
        
        # Reset memory for collection
        current_memory = torch.zeros((self.num_workers, self.max_episode_length, self.num_blocks, self.observation_space_dim), dtype=torch.float32)
        current_episode_step = torch.zeros((self.num_workers,), dtype=torch.long)
        
        episode_obs_buffers = [[] for _ in range(self.num_workers)]
        
        with tqdm(total=collect_episodes, desc=f"DAGGER Iteration {self.dagger_iteration + 1}") as pbar:
            while episodes_collected < collect_episodes:
                # Store current observations
                for w in range(self.num_workers):
                    episode_obs_buffers[w].append(current_obs[w].copy())
                
                # Get actions from current policy
                with torch.no_grad():
                    obs_tensor = torch.tensor(current_obs).to(self.device)
                    
                    # Get memory components for current policy
                    memory_mask = self.memory_mask[torch.clip(current_episode_step, 0, self.memory_length - 1)]
                    memory_indices = self.memory_indices[current_episode_step]
                    sliced_memory = batched_index_select(current_memory, 1, memory_indices)
                    
                    # Forward current policy
                    policy, _, memory_update = self.model(obs_tensor, sliced_memory, memory_mask, memory_indices)
                    
                    # Update memory
                    current_memory[range(self.num_workers), current_episode_step] = memory_update
                    
                    # Sample actions
                    actions = policy[0].sample().cpu().numpy()
                
                # Send actions to environments
                for w, worker in enumerate(self.workers):
                    worker.child.send(("step", actions[w]))
                
                # Receive results
                for w, worker in enumerate(self.workers):
                    obs, reward, done, info = worker.child.recv()
                    
                    if done:
                        # Episode finished - query expert for all observations in this episode
                        episode_observations = np.array(episode_obs_buffers[w])
                        expert_actions = self._query_expert_actions(episode_observations)
                        
                        # Add to DAGGER dataset
                        collected_obs.extend(episode_observations)
                        collected_expert_actions.extend(expert_actions)
                        
                        episodes_collected += 1
                        pbar.update(1)
                        
                        # Reset for next episode
                        episode_obs_buffers[w] = []
                        current_episode_step[w] = 0
                        current_memory[w] = torch.zeros((self.max_episode_length, self.num_blocks, self.observation_space_dim), dtype=torch.float32)
                        
                        # Reset environment
                        worker.child.send(("reset", None))
                        obs = worker.child.recv()
                        
                        if episodes_collected >= collect_episodes:
                            break
                    else:
                        current_episode_step[w] += 1
                    
                    current_obs[w] = obs
                
                if episodes_collected >= collect_episodes:
                    break
        
        # Add collected data to DAGGER dataset
        if collected_obs:
            self.dagger_dataset_obs.extend(collected_obs)
            self.dagger_dataset_actions.extend(collected_expert_actions)
            
            print(f"Added {len(collected_obs)} new expert-labeled transitions to DAGGER dataset")
            print(f"Total DAGGER dataset size: {len(self.dagger_dataset_obs)} transitions")
        
        # Train on aggregated dataset
        self._train_dagger_epoch()
        
        self.dagger_iteration += 1

    def _train_dagger_epoch(self):
        """Train the model on the current DAGGER dataset."""
        if not self.dagger_dataset_obs:
            print("No DAGGER data available for training")
            return
            
        dagger_config = self.config["dagger"]
        epochs = dagger_config.get("epochs_per_iteration", 5)
        batch_size = dagger_config.get("batch_size", 64)
        dagger_lr = dagger_config.get("learning_rate", self.lr_schedule["initial"])
        
        print(f"Training on DAGGER dataset for {epochs} epochs...")
        
        # Convert to tensors
        obs_tensor = torch.tensor(self.dagger_dataset_obs, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(self.dagger_dataset_actions, dtype=torch.long).to(self.device)
        
        # Temporarily set learning rate for DAGGER
        original_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        for pg in self.optimizer.param_groups:
            pg["lr"] = dagger_lr
        
        num_samples = obs_tensor.shape[0]
        num_batches_per_epoch = (num_samples + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            permutation = torch.randperm(num_samples, device=self.device)
            epoch_total_loss = 0.0
            
            for i in range(0, num_samples, batch_size):
                indices = permutation[i:i + batch_size]
                obs_batch = obs_tensor[indices]
                actions_batch = actions_tensor[indices]
                current_batch_size = obs_batch.shape[0]
                
                # Create dummy memory components for DAGGER training
                dummy_memory = torch.zeros(
                    current_batch_size, self.memory_length, self.num_blocks, self.observation_space_dim,
                    device=self.device, dtype=torch.float32
                )
                dummy_mask = self.memory_mask[0].unsqueeze(0).repeat(current_batch_size, 1, 1)
                dummy_indices = self.memory_indices[0].unsqueeze(0).repeat(current_batch_size, 1)
                
                # Forward pass
                policy, _, _ = self.model(obs_batch, dummy_memory, dummy_mask, dummy_indices)
                
                # Calculate loss (negative log likelihood)
                log_probs = policy[0].log_prob(actions_batch)
                dagger_loss = -log_probs.mean()
                
                # Backward pass
                self.optimizer.zero_grad()
                dagger_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config["max_grad_norm"])
                self.optimizer.step()
                
                epoch_total_loss += dagger_loss.item()
            
            avg_epoch_loss = epoch_total_loss / num_batches_per_epoch if num_batches_per_epoch > 0 else 0.0
            print(f"DAGGER Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_epoch_loss:.4f}")
            
            # Log to wandb and tensorboard
            if wandb.run:
                wandb.log({
                    "dagger/avg_epoch_loss": avg_epoch_loss,
                    "dagger/epoch": epoch + 1,
                    "dagger/iteration": self.dagger_iteration + 1,
                    "dagger/dataset_size": len(self.dagger_dataset_obs)
                })
            self.writer.add_scalar("dagger/avg_epoch_loss", avg_epoch_loss, 
                                 self.dagger_iteration * epochs + epoch + 1)
        
        # Restore original learning rates
        for i, pg in enumerate(self.optimizer.param_groups):
            pg["lr"] = original_lrs[i]
        
        print(f"DAGGER iteration {self.dagger_iteration + 1} training completed.")

    def run_training(self) -> None:
        """Runs the entire training logic from sampling data to optimizing the model. Only the final model is saved."""
        # Check if BC pre-training is enabled and data is loaded
        bc_train_config = self.config.get("bc_pretraining", {})
        if bc_train_config.get("enabled", False) and self.bc_obs_all is not None:
            self._run_bc_pretraining()
        else:
            if bc_train_config.get("enabled", False):
                 print("BC pre-training was enabled but data loading failed. Skipping BC.")

        # Check if DAGGER is enabled
        dagger_config = self.config.get("dagger", {})
        dagger_enabled = dagger_config.get("enabled", False)
        dagger_iterations = dagger_config.get("iterations", 5)
        dagger_frequency = dagger_config.get("frequency", 50)  # Run DAGGER every N PPO updates
        
        if dagger_enabled and self.expert_model is not None:
            print(f"DAGGER training enabled: {dagger_iterations} iterations, every {dagger_frequency} PPO updates")
            
            # Initialize DAGGER dataset with expert data if available
            if self.bc_obs_all is not None and self.bc_actions_all is not None:
                print("Initializing DAGGER dataset with BC expert data...")
                self.dagger_dataset_obs = self.bc_obs_all.cpu().numpy().tolist()
                self.dagger_dataset_actions = self.bc_actions_all.cpu().numpy().tolist()
                print(f"Initialized DAGGER dataset with {len(self.dagger_dataset_obs)} expert transitions")
        
        print("Step 6: Starting training using " + str(self.device))
        # Store episode results for monitoring statistics
        episode_infos = deque(maxlen=100)

        for update in range(self.start_update, self.config["updates"]):
            # Run DAGGER iteration if enabled and it's time
            if (dagger_enabled and self.expert_model is not None and 
                self.dagger_iteration < dagger_iterations and 
                update % dagger_frequency == 0 and update > 0):
                self._run_dagger_iteration()
            
            # Decay hyperparameters polynomially based on the provided config
            learning_rate = polynomial_decay(self.lr_schedule["initial"], self.lr_schedule["final"], self.lr_schedule["max_decay_steps"], self.lr_schedule["power"], update)
            beta = polynomial_decay(self.beta_schedule["initial"], self.beta_schedule["final"], self.beta_schedule["max_decay_steps"], self.beta_schedule["power"], update)
            clip_range = polynomial_decay(self.cr_schedule["initial"], self.cr_schedule["final"], self.cr_schedule["max_decay_steps"], self.cr_schedule["power"], update)

            # Sample training data
            sampled_episode_info = self._sample_training_data()

            # Prepare the sampled data inside the buffer (splits data into sequences)
            self.buffer.prepare_batch_dict()

            # Train epochs
            training_stats, grad_info = self._train_epochs(learning_rate, clip_range, beta)
            training_stats = np.mean(training_stats, axis=0)

            # Store recent episode infos
            episode_infos.extend(sampled_episode_info)
            episode_result = process_episode_info(episode_infos)

            # Print training statistics
            if "success" in episode_result:
                result = "{:4} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} success={:.2f} pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f} value={:.3f} advantage={:.3f}".format(
                    update, episode_result["reward_mean"], episode_result["reward_std"], episode_result["length_mean"], episode_result["length_std"], episode_result["success"],
                    training_stats[0], training_stats[1], training_stats[3], training_stats[2], torch.mean(self.buffer.values), torch.mean(self.buffer.advantages))
            else:
                result = "{:4} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f} value={:.3f} advantage={:.3f}".format(
                    update, episode_result["reward_mean"], episode_result["reward_std"], episode_result["length_mean"], episode_result["length_std"], 
                    training_stats[0], training_stats[1], training_stats[3], training_stats[2], torch.mean(self.buffer.values), torch.mean(self.buffer.advantages))
            
            # Add DAGGER info to result if enabled
            if dagger_enabled and self.expert_model is not None:
                result += f" dagger_iter={self.dagger_iteration}/{dagger_iterations} dagger_data={len(self.dagger_dataset_obs)}"
            
            print(result)

            # Write training statistics to tensorboard
            self._write_gradient_summary(update, grad_info)
            self._write_training_summary(update, training_stats, episode_result)

            # Periodic checkpointing
            if (update + 1) % self.checkpoint_frequency == 0:
                self._save_checkpoint(update, episode_result)

            # Save best model if performance improved
            if self.keep_best_checkpoint and episode_result and "reward_mean" in episode_result:
                current_reward = episode_result["reward_mean"]
                if current_reward > self.best_reward:
                    self.best_reward = current_reward
                    self._save_best_checkpoint(update, current_reward)

        # Run any remaining DAGGER iterations at the end
        if (dagger_enabled and self.expert_model is not None and 
            self.dagger_iteration < dagger_iterations):
            remaining_iterations = dagger_iterations - self.dagger_iteration
            print(f"Running {remaining_iterations} remaining DAGGER iterations...")
            for _ in range(remaining_iterations):
                self._run_dagger_iteration()

        # Save the trained model at the end of the training
        self._save_model()

    def _sample_training_data(self) -> list:
        """Runs all n workers for n steps to sample training data.

        Returns:
            {list} -- list of results of completed episodes.
        """
        episode_infos = []
        
        # Init episodic memory buffer using each workers' current episodic memory
        self.buffer.memories = [self.memory[w] for w in range(self.num_workers)]
        for w in range(self.num_workers):
            self.buffer.memory_index[w] = w

        # Sample actions from the model and collect experiences for optimization
        for t in range(self.config["worker_steps"]):
            # Gradients can be omitted for sampling training data
            with torch.no_grad():
                # Store the initial observations inside the buffer
                self.buffer.obs[:, t] = torch.tensor(self.obs)
                # Store mask and memory indices inside the buffer
                self.buffer.memory_mask[:, t] = self.memory_mask[torch.clip(self.worker_current_episode_step, 0, self.memory_length - 1)]
                self.buffer.memory_indices[:, t] = self.memory_indices[self.worker_current_episode_step]
                # Retrieve the memory window from the entire episodic memory
                sliced_memory = batched_index_select(self.memory, 1, self.buffer.memory_indices[:,t])
                # Forward the model to retrieve the policy, the states' value and the new memory item
                policy, value, memory = self.model(torch.tensor(self.obs), sliced_memory, self.buffer.memory_mask[:, t],
                                                   self.buffer.memory_indices[:,t])
                
                # Add new memory item to the episodic memory
                self.memory[self.worker_ids, self.worker_current_episode_step] = memory

                # Sample actions from each individual policy branch
                actions = []
                log_probs = []
                for action_branch in policy:
                    action = action_branch.sample()
                    actions.append(action)
                    log_probs.append(action_branch.log_prob(action))
                # Write actions, log_probs and values to buffer
                self.buffer.actions[:, t] = torch.stack(actions, dim=1)
                self.buffer.log_probs[:, t] = torch.stack(log_probs, dim=1)
                self.buffer.values[:, t] = value

            # Send actions to the environments
            for w, worker in enumerate(self.workers):
                worker.child.send(("step", self.buffer.actions[w, t].cpu().numpy()))

            # Retrieve step results from the environments
            for w, worker in enumerate(self.workers):
                obs, self.buffer.rewards[w, t], self.buffer.dones[w, t], info = worker.child.recv()
                if info: # i.e. done
                    # Reset the worker's current timestep
                    self.worker_current_episode_step[w] = 0
                    # Store the information of the completed episode (e.g. total reward, episode length)
                    episode_infos.append(info)
                    # Reset the agent (potential interface for providing reset parameters)
                    worker.child.send(("reset", None))
                    # Get data from reset
                    obs = worker.child.recv()
                    # Break the reference to the worker's memory
                    mem_index = self.buffer.memory_index[w, t]
                    self.buffer.memories[mem_index] = self.buffer.memories[mem_index].clone()
                    # Reset episodic memory
                    self.memory[w] = torch.zeros((self.max_episode_length, self.num_blocks, self.observation_space_dim), dtype=torch.float32)
                    if t < self.config["worker_steps"] - 1:
                        # Store memory inside the buffer
                        self.buffer.memories.append(self.memory[w])
                        # Store the reference of to the current episodic memory inside the buffer
                        self.buffer.memory_index[w, t + 1:] = len(self.buffer.memories) - 1
                else:
                    # Increment worker timestep
                    self.worker_current_episode_step[w] +=1
                # Store latest observations
                self.obs[w] = obs
                            
        # Compute the last value of the current observation and memory window to compute GAE
        last_value = self.get_last_value()
        # Compute advantages
        self.buffer.calc_advantages(last_value, self.config["gamma"], self.config["lamda"])

        return episode_infos

    def get_last_value(self):
        """Returns:
                {torch.tensor} -- Last value of the current observation and memory window to compute GAE"""
        start = torch.clip(self.worker_current_episode_step - self.memory_length, 0)
        end = torch.clip(self.worker_current_episode_step, self.memory_length)
        indices = torch.stack([torch.arange(start[b],end[b]) for b in range(self.num_workers)]).long()
        sliced_memory = batched_index_select(self.memory, 1, indices) # Retrieve the memory window from the entire episode
        _, last_value, _ = self.model(torch.tensor(self.obs),
                                        sliced_memory, self.memory_mask[torch.clip(self.worker_current_episode_step, 0, self.memory_length - 1)],
                                        self.buffer.memory_indices[:,-1])
        return last_value

    def _train_epochs(self, learning_rate:float, clip_range:float, beta:float) -> list:
        """Trains several PPO epochs over one batch of data while dividing the batch into mini batches.
        
        Arguments:
            learning_rate {float} -- The current learning rate
            clip_range {float} -- The current clip range
            beta {float} -- The current entropy bonus coefficient
            
        Returns:
            {tuple} -- Training and gradient statistics of one training epoch"""
        train_info, grad_info = [], {}
        for _ in range(self.config["epochs"]):
            mini_batch_generator = self.buffer.mini_batch_generator()
            for mini_batch in mini_batch_generator:
                train_info.append(self._train_mini_batch(mini_batch, learning_rate, clip_range, beta))
                # for key, value in self.model.get_grad_norm().items():
                #     grad_info.setdefault(key, []).append(value)
        return train_info, grad_info

    def _train_mini_batch(self, samples:dict, learning_rate:float, clip_range:float, beta:float) -> list:
        """Uses one mini batch to optimize the model.

        Arguments:
            mini_batch {dict} -- The to be used mini batch data to optimize the model
            learning_rate {float} -- Current learning rate
            clip_range {float} -- Current clip range
            beta {float} -- Current entropy bonus coefficient

        Returns:
            {list} -- list of trainig statistics (e.g. loss)
        """
        # Select episodic memory windows
        memory = batched_index_select(samples["memories"], 1, samples["memory_indices"])
        
        # Forward model
        policy, value, _ = self.model(samples["obs"], memory, samples["memory_mask"], samples["memory_indices"])

        # Retrieve and process log_probs from each policy branch
        log_probs, entropies = [], []
        for i, policy_branch in enumerate(policy):
            log_probs.append(policy_branch.log_prob(samples["actions"][:, i]))
            entropies.append(policy_branch.entropy())
        log_probs = torch.stack(log_probs, dim=1)
        entropies = torch.stack(entropies, dim=1).sum(1).reshape(-1)

        # Compute policy surrogates to establish the policy loss
        normalized_advantage = (samples["advantages"] - samples["advantages"].mean()) / (samples["advantages"].std() + 1e-8)
        normalized_advantage = normalized_advantage.unsqueeze(1).repeat(1, len(self.action_space_shape)) # Repeat is necessary for multi-discrete action spaces
        log_ratio = log_probs - samples["log_probs"]
        ratio = torch.exp(log_ratio)
        surr1 = ratio * normalized_advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * normalized_advantage
        policy_loss = torch.min(surr1, surr2)
        policy_loss = policy_loss.mean()

        # Value  function loss
        sampled_return = samples["values"] + samples["advantages"]
        clipped_value = samples["values"] + (value - samples["values"]).clamp(min=-clip_range, max=clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = vf_loss.mean()

        # Entropy Bonus
        entropy_bonus = entropies.mean()

        # Complete loss
        loss = -(policy_loss - self.config["value_loss_coefficient"] * vf_loss + beta * entropy_bonus)

        # Compute gradients
        for pg in self.optimizer.param_groups:
            pg["lr"] = learning_rate
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config["max_grad_norm"])
        self.optimizer.step()

        # Monitor additional training stats
        approx_kl = (ratio - 1.0) - log_ratio # http://joschu.net/blog/kl-approx.html
        clip_fraction = (abs((ratio - 1.0)) > clip_range).float().mean()

        return [policy_loss.cpu().data.numpy(),
                vf_loss.cpu().data.numpy(),
                loss.cpu().data.numpy(),
                entropy_bonus.cpu().data.numpy(),
                approx_kl.mean().cpu().data.numpy(),
                clip_fraction.cpu().data.numpy()]

    def _write_training_summary(self, update, training_stats, episode_result) -> None:
        """Writes to an event file based on the run-id argument.

        Arguments:
            update {int} -- Current PPO Update
            training_stats {list} -- Statistics of the training algorithm
            episode_result {dict} -- Statistics of completed episodes
        """
        # Log to wandb
        wandb_log = {}
        
        if episode_result:
            for key in episode_result:
                if "std" not in key:
                    self.writer.add_scalar("episode/" + key, episode_result[key], update)
                    wandb_log["episode/" + key] = episode_result[key]
                    
        self.writer.add_scalar("losses/loss", training_stats[2], update)
        self.writer.add_scalar("losses/policy_loss", training_stats[0], update)
        self.writer.add_scalar("losses/value_loss", training_stats[1], update)
        self.writer.add_scalar("losses/entropy", training_stats[3], update)
        self.writer.add_scalar("training/value_mean", torch.mean(self.buffer.values), update)
        self.writer.add_scalar("training/advantage_mean", torch.mean(self.buffer.advantages), update)
        self.writer.add_scalar("other/clip_fraction", training_stats[4], update)
        self.writer.add_scalar("other/kl", training_stats[5], update)
        
        # Add metrics to wandb log dict
        wandb_log.update({
            "losses/loss": training_stats[2],
            "losses/policy_loss": training_stats[0],
            "losses/value_loss": training_stats[1],
            "losses/entropy": training_stats[3],
            "training/value_mean": torch.mean(self.buffer.values).item(),
            "training/advantage_mean": torch.mean(self.buffer.advantages).item(),
            "other/clip_fraction": training_stats[4],
            "other/kl": training_stats[5],
            "update": update
        })
        
        # Log to wandb
        wandb.log(wandb_log)
        
    def _write_gradient_summary(self, update, grad_info):
        """Adds gradient statistics to the tensorboard event file.

        Arguments:
            update {int} -- Current PPO Update
            grad_info {dict} -- Gradient statistics
        """
        wandb_log = {}
        for key, value in grad_info.items():
            mean_value = np.mean(value)
            self.writer.add_scalar("gradients/" + key, mean_value, update)
            wandb_log["gradients/" + key] = mean_value
            
        wandb.log(wandb_log, commit=False)  # commit=False to avoid duplicate step increments

    def _save_checkpoint(self, update: int, episode_result: dict = None) -> None:
        """Saves a training checkpoint including model, optimizer, and training state.
        
        Arguments:
            update {int} -- Current update number
            episode_result {dict, optional} -- Episode statistics for this update
        """
        if not os.path.exists("./checkpoints"):
            os.makedirs("./checkpoints")
            
        checkpoint = {
            'update': update,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_reward': self.best_reward,
            'episode_result': episode_result,
            'memory': self.memory.clone(),
            'worker_current_episode_step': self.worker_current_episode_step.clone(),
            # DAGGER state
            'dagger_iteration': self.dagger_iteration,
            'dagger_dataset_obs': self.dagger_dataset_obs.copy() if self.dagger_dataset_obs else [],
            'dagger_dataset_actions': self.dagger_dataset_actions.copy() if self.dagger_dataset_actions else [],
        }
        
        checkpoint_path = f"./checkpoints/{self.run_id}_update_{update}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Save to wandb
        wandb.save(checkpoint_path)
        
        # Clean up old checkpoints (keep only the most recent ones)
        self._cleanup_old_checkpoints()

    def _save_best_checkpoint(self, update: int, reward: float) -> None:
        """Saves the best performing model checkpoint.
        
        Arguments:
            update {int} -- Current update number
            reward {float} -- Current reward that is the new best
        """
        if not os.path.exists("./checkpoints"):
            os.makedirs("./checkpoints")
            
        checkpoint = {
            'update': update,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_reward': reward,
            'memory': self.memory.clone(),
            'worker_current_episode_step': self.worker_current_episode_step.clone(),
            # DAGGER state
            'dagger_iteration': self.dagger_iteration,
            'dagger_dataset_obs': self.dagger_dataset_obs.copy() if self.dagger_dataset_obs else [],
            'dagger_dataset_actions': self.dagger_dataset_actions.copy() if self.dagger_dataset_actions else [],
        }
        
        best_checkpoint_path = f"./checkpoints/{self.run_id}_best.pt"
        torch.save(checkpoint, best_checkpoint_path)
        print(f"New best model saved to {best_checkpoint_path} (reward: {reward:.2f})")
        
        # Save to wandb
        wandb.save(best_checkpoint_path)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Loads a training checkpoint to resume training.
        
        Arguments:
            checkpoint_path {str} -- Path to the checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        self.start_update = checkpoint['update'] + 1
        self.best_reward = checkpoint.get('best_reward', float('-inf'))
        
        # Load memory states if available
        if 'memory' in checkpoint:
            self.memory = checkpoint['memory'].to(self.device)
        if 'worker_current_episode_step' in checkpoint:
            self.worker_current_episode_step = checkpoint['worker_current_episode_step'].to(self.device)
        
        # Load DAGGER state if available
        if 'dagger_iteration' in checkpoint:
            self.dagger_iteration = checkpoint['dagger_iteration']
        if 'dagger_dataset_obs' in checkpoint:
            self.dagger_dataset_obs = checkpoint['dagger_dataset_obs']
        if 'dagger_dataset_actions' in checkpoint:
            self.dagger_dataset_actions = checkpoint['dagger_dataset_actions']
            
        print(f"Resumed from update {checkpoint['update']}, best reward: {self.best_reward:.2f}")
        if hasattr(self, 'dagger_iteration'):
            print(f"DAGGER state: iteration {self.dagger_iteration}, dataset size {len(self.dagger_dataset_obs)}")

    def _cleanup_old_checkpoints(self) -> None:
        """Removes old checkpoint files, keeping only the most recent ones."""
        checkpoint_dir = "./checkpoints"
        if not os.path.exists(checkpoint_dir):
            return
            
        # Get all checkpoint files for this run (excluding best checkpoint)
        checkpoint_files = []
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith(f"{self.run_id}_update_") and filename.endswith(".pt"):
                filepath = os.path.join(checkpoint_dir, filename)
                # Extract update number from filename
                try:
                    update_num = int(filename.split("_update_")[1].split(".pt")[0])
                    checkpoint_files.append((update_num, filepath))
                except (ValueError, IndexError):
                    continue
        
        # Sort by update number and keep only the most recent ones
        checkpoint_files.sort(key=lambda x: x[0])
        if len(checkpoint_files) > self.max_checkpoints:
            files_to_remove = checkpoint_files[:-self.max_checkpoints]
            for _, filepath in files_to_remove:
                try:
                    os.remove(filepath)
                    print(f"Removed old checkpoint: {filepath}")
                except OSError:
                    pass

    def _save_model(self) -> None:
        """Saves the model and the used training config to the models directory. The filename is based on the run id."""
        if not os.path.exists("./models"):
            os.makedirs("./models")
        self.model.cpu()
        model_path = "./models/" + self.run_id + ".nn"
        pickle.dump((self.model.state_dict(), self.config), open(model_path, "wb"))
        print("Model saved to " + model_path)
        # Save model to wandb
        wandb.save(model_path)
        # Move model back to device
        self.model.to(self.device)

    def close(self) -> None:
        """Terminates the trainer and all related processes."""
        try:
            self.dummy_env.close()
        except:
            pass

        try:
            self.writer.close()
        except:
            pass

        try:
            for worker in self.workers:
                worker.child.send(("close", None))
        except:
            pass
            
        # Finish wandb run
        try:
            wandb.finish()
        except:
            pass

        time.sleep(1.0)
        exit(0)
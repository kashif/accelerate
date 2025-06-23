# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import DistributedType, TorchTitanPlugin


class SimpleModel(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 10)
        
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return self.linear3(x)


class TorchTitanIntegrationTest(unittest.TestCase):
    """Test TorchTitan integration functionality."""

    def setUp(self):
        # Reset accelerator state before each test
        AcceleratorState._reset_state(reset_partial_state=True)

    def tearDown(self):
        # Clean up after each test
        AcceleratorState._reset_state(reset_partial_state=True)

    def test_torchtitan_plugin_creation(self):
        """Test basic TorchTitanPlugin creation and configuration."""
        # Test with default config
        plugin = TorchTitanPlugin()
        self.assertEqual(plugin.model_name, "llama3")
        self.assertIsNotNone(plugin.job_config)
        
        # Test with custom model name
        plugin_custom = TorchTitanPlugin(model_name="llama4")
        self.assertEqual(plugin_custom.model_name, "llama4")
        
        # Test with custom job config
        custom_config = {
            "training": {"batch_size": 16, "seq_len": 1024},
            "model": {"name": "gpt2"}
        }
        plugin_config = TorchTitanPlugin(job_config=custom_config, model_name="gpt2")
        self.assertEqual(plugin_config.job_config, custom_config)
        self.assertEqual(plugin_config.model_name, "gpt2")

    def test_accelerator_with_torchtitan_plugin(self):
        """Test Accelerator initialization with TorchTitanPlugin."""
        plugin = TorchTitanPlugin(model_name="llama4")
        accelerator = Accelerator(torchtitan_plugin=plugin)
        
        # Check that the plugin is stored correctly
        self.assertEqual(accelerator.state.torchtitan_plugin, plugin)
        self.assertEqual(accelerator.state.torchtitan_plugin.model_name, "llama4")

    @patch.dict(os.environ, {"ACCELERATE_USE_TORCHTITAN": "true"})
    def test_torchtitan_distributed_type_detection(self):
        """Test that TorchTitan distributed type is detected when environment variable is set."""
        plugin = TorchTitanPlugin(model_name="llama3")
        accelerator = Accelerator(torchtitan_plugin=plugin)
        
        # Check that distributed type is set to TORCHTITAN
        self.assertEqual(accelerator.state.distributed_type, DistributedType.TORCHTITAN)

    @patch.dict(os.environ, {"ACCELERATE_USE_TORCHTITAN": "true"})
    def test_torchtitan_model_preparation(self):
        """Test model preparation with TorchTitan."""
        plugin = TorchTitanPlugin(
            job_config={
                "training": {"batch_size": 8, "seq_len": 512},
                "model": {"name": "llama3"}
            },
            model_name="llama3"
        )
        
        accelerator = Accelerator(torchtitan_plugin=plugin)
        model = SimpleModel()
        
        # Prepare the model
        prepared_model = accelerator.prepare(model)
        
        # Check that the model was processed
        self.assertIsNotNone(prepared_model)
        self.assertEqual(len(accelerator._models), 1)

    @patch.dict(os.environ, {"ACCELERATE_USE_TORCHTITAN": "true"})
    def test_torchtitan_optimizer_preparation(self):
        """Test optimizer preparation with TorchTitan."""
        plugin = TorchTitanPlugin(model_name="llama3")
        accelerator = Accelerator(torchtitan_plugin=plugin)
        
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Prepare model and optimizer together
        prepared_model, prepared_optimizer = accelerator.prepare(model, optimizer)
        
        # Check that both were processed
        self.assertIsNotNone(prepared_model)
        self.assertIsNotNone(prepared_optimizer)
        self.assertEqual(len(accelerator._models), 1)
        self.assertEqual(len(accelerator._optimizers), 1)

    @patch.dict(os.environ, {"ACCELERATE_USE_TORCHTITAN": "true"})
    def test_torchtitan_dataloader_preparation(self):
        """Test DataLoader preparation with TorchTitan."""
        plugin = TorchTitanPlugin(model_name="llama3")
        accelerator = Accelerator(torchtitan_plugin=plugin)
        
        # Create a simple dataset and dataloader
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 10), torch.randint(0, 2, (100,))
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
        
        # Prepare the dataloader
        prepared_dataloader = accelerator.prepare(dataloader)
        
        # Check that the dataloader was processed
        self.assertIsNotNone(prepared_dataloader)

    @patch.dict(os.environ, {"ACCELERATE_USE_TORCHTITAN": "true"})
    def test_torchtitan_scheduler_preparation(self):
        """Test scheduler preparation with TorchTitan."""
        plugin = TorchTitanPlugin(model_name="llama3")
        accelerator = Accelerator(torchtitan_plugin=plugin)
        
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        
        # Prepare all components
        prepared_model, prepared_optimizer, prepared_scheduler = accelerator.prepare(
            model, optimizer, scheduler
        )
        
        # Check that all were processed
        self.assertIsNotNone(prepared_model)
        self.assertIsNotNone(prepared_optimizer)
        self.assertIsNotNone(prepared_scheduler)
        self.assertEqual(len(accelerator._models), 1)
        self.assertEqual(len(accelerator._optimizers), 1)
        self.assertEqual(len(accelerator._schedulers), 1)

    @patch.dict(os.environ, {"ACCELERATE_USE_TORCHTITAN": "true"})
    def test_torchtitan_complete_training_setup(self):
        """Test complete training setup with TorchTitan including all components."""
        plugin = TorchTitanPlugin(
            job_config={
                "training": {"batch_size": 4, "seq_len": 256},
                "model": {"name": "llama3", "hidden_size": 512},
                "optimizer": {"name": "adam", "lr": 0.0001}
            },
            model_name="llama3"
        )
        accelerator = Accelerator(torchtitan_plugin=plugin)
        
        # Create all training components
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        dataset = torch.utils.data.TensorDataset(
            torch.randn(32, 10), torch.randint(0, 2, (32,))
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)
        
        # Prepare all components
        prepared_model, prepared_optimizer, prepared_dataloader, prepared_scheduler = (
            accelerator.prepare(model, optimizer, dataloader, scheduler)
        )
        
        # Verify all components are prepared
        self.assertIsNotNone(prepared_model)
        self.assertIsNotNone(prepared_optimizer)
        self.assertIsNotNone(prepared_dataloader)
        self.assertIsNotNone(prepared_scheduler)
        
        # Check internal state
        self.assertEqual(len(accelerator._models), 1)
        self.assertEqual(len(accelerator._optimizers), 1)
        self.assertEqual(len(accelerator._schedulers), 1)

    def test_torchtitan_plugin_config_validation(self):
        """Test TorchTitanPlugin configuration validation."""
        # Test valid configurations
        valid_configs = [
            None,  # Should use default
            {"training": {"batch_size": 8}},
            {"model": {"name": "llama3"}},
            {
                "training": {"batch_size": 16, "seq_len": 1024},
                "model": {"name": "gpt2"}
            }
        ]
        
        for config in valid_configs:
            plugin = TorchTitanPlugin(job_config=config)
            self.assertIsNotNone(plugin.job_config)

    @patch.dict(os.environ, {"ACCELERATE_USE_TORCHTITAN": "true"})
    def test_torchtitan_environment_variable_detection(self):
        """Test that TorchTitan is properly detected via environment variables."""
        plugin = TorchTitanPlugin()
        
        # Initialize accelerator - should detect TorchTitan from env var
        accelerator = Accelerator(torchtitan_plugin=plugin)
        
        # Verify distributed type is set correctly
        self.assertEqual(accelerator.distributed_type, DistributedType.TORCHTITAN)

    def test_state_reset_functionality(self):
        """Test that AcceleratorState can be reset properly when using TorchTitan."""
        plugin = TorchTitanPlugin(model_name="test_model")
        accelerator = Accelerator(torchtitan_plugin=plugin)
        
        # Initialize with some state
        model = SimpleModel()
        prepared_model = accelerator.prepare(model)
        self.assertEqual(len(accelerator._models), 1)
        
        # Reset state
        AcceleratorState._reset_state(reset_partial_state=True)
        
        # Create new accelerator and verify clean state
        accelerator2 = Accelerator(torchtitan_plugin=TorchTitanPlugin(model_name="llama4"))
        self.assertEqual(accelerator2.state.torchtitan_plugin.model_name, "llama4")

    @patch.dict(os.environ, {"ACCELERATE_USE_TORCHTITAN": "true"})
    def test_torchtitan_mixed_precision_compatibility(self):
        """Test TorchTitan works with mixed precision settings."""
        plugin = TorchTitanPlugin(model_name="llama3")
        accelerator = Accelerator(
            torchtitan_plugin=plugin,
            mixed_precision="fp16"
        )
        
        model = SimpleModel()
        prepared_model = accelerator.prepare(model)
        
        # Check that preparation worked with mixed precision
        self.assertIsNotNone(prepared_model)
        self.assertEqual(accelerator.mixed_precision, "fp16")


class TestTorchTitanSharding(unittest.TestCase):
    def setUp(self):
        """Clean up state before each test."""
        AcceleratorState._reset_state()

    def tearDown(self):
        """Clean up state after each test."""
        AcceleratorState._reset_state()

    def test_plugin_basic_parallelism_config(self):
        """Test basic parallelism configuration in TorchTitanPlugin."""
        plugin = TorchTitanPlugin(
            tp_degree=2,
            pp_degree=2,
            dp_degree=1
        )
        
        self.assertEqual(plugin.tp_degree, 2)
        self.assertEqual(plugin.pp_degree, 2) 
        self.assertEqual(plugin.dp_degree, 1)
        self.assertEqual(plugin.get_world_size_requirements(), 4)

    def test_plugin_auto_dp_calculation(self):
        """Test automatic data parallelism degree calculation."""
        plugin = TorchTitanPlugin(
            tp_degree=2,
            pp_degree=2,
            dp_degree=None  # Should be auto-calculated
        )
        
        # Test with world size 8 (should give dp_degree=2)  
        plugin.validate_world_size(8)
        self.assertEqual(plugin.dp_degree, 2)
        
        # Test with world size 4 (should give dp_degree=1)
        plugin = TorchTitanPlugin(tp_degree=2, pp_degree=2, dp_degree=None)
        plugin.validate_world_size(4)
        self.assertEqual(plugin.dp_degree, 1)

    def test_plugin_world_size_validation_exact(self):
        """Test world size validation with exact dp_degree."""
        plugin = TorchTitanPlugin(tp_degree=2, pp_degree=2, dp_degree=2)
        
        # Should work with exact world size
        plugin.validate_world_size(8)
        
        # Should fail with wrong world size
        with self.assertRaises(ValueError):
            plugin.validate_world_size(6)

    def test_plugin_world_size_validation_auto(self):
        """Test world size validation with auto dp_degree."""
        plugin = TorchTitanPlugin(tp_degree=2, pp_degree=2, dp_degree=None)
        
        # Should work with divisible world sizes
        plugin.validate_world_size(4)
        plugin.validate_world_size(8)
        plugin.validate_world_size(12)
        
        # Should fail with non-divisible world size
        with self.assertRaises(ValueError):
            plugin.validate_world_size(5)

    def test_plugin_fsdp_configuration(self):
        """Test FSDP configuration options."""
        plugin = TorchTitanPlugin(
            enable_fsdp=True,
            fsdp_sharding_strategy="FULL_SHARD",
            fsdp_backward_prefetch="BACKWARD_PRE",
            fsdp_cpu_offload=True,
            fsdp_mixed_precision_policy={"param": "fp16", "reduce": "fp32"}
        )
        
        self.assertTrue(plugin.enable_fsdp)
        self.assertEqual(plugin.fsdp_sharding_strategy, "FULL_SHARD")
        self.assertEqual(plugin.fsdp_backward_prefetch, "BACKWARD_PRE")
        self.assertTrue(plugin.fsdp_cpu_offload)
        
        fsdp_config = plugin.get_fsdp_config()
        self.assertEqual(fsdp_config["sharding_strategy"], "FULL_SHARD")
        self.assertEqual(fsdp_config["backward_prefetch"], "BACKWARD_PRE")
        self.assertTrue(fsdp_config["cpu_offload"])
        self.assertEqual(fsdp_config["mixed_precision_policy"]["param"], "fp16")

    def test_plugin_invalid_fsdp_strategy(self):
        """Test invalid FSDP sharding strategy raises error."""
        with self.assertRaises(ValueError):
            TorchTitanPlugin(
                enable_fsdp=True,
                fsdp_sharding_strategy="INVALID_STRATEGY"
            )

    def test_plugin_invalid_fsdp_prefetch(self):
        """Test invalid FSDP backward prefetch raises error."""
        with self.assertRaises(ValueError):
            TorchTitanPlugin(
                enable_fsdp=True,
                fsdp_backward_prefetch="INVALID_PREFETCH"
            )

    def test_plugin_activation_checkpointing_config(self):
        """Test activation checkpointing configuration."""
        plugin = TorchTitanPlugin(
            activation_checkpointing=True,
            selective_checkpointing_layers=["attention", "mlp"]
        )
        
        self.assertTrue(plugin.activation_checkpointing)
        self.assertEqual(plugin.selective_checkpointing_layers, ["attention", "mlp"])
        
        checkpointing_config = plugin.get_checkpointing_config()
        self.assertTrue(checkpointing_config["enabled"])
        self.assertEqual(checkpointing_config["selective_layers"], ["attention", "mlp"])

    def test_plugin_compilation_config(self):
        """Test model compilation configuration."""
        plugin = TorchTitanPlugin(
            compile_model=True,
            compile_config={
                "backend": "inductor",
                "mode": "max-autotune",
                "fullgraph": True
            }
        )
        
        self.assertTrue(plugin.compile_model)
        self.assertEqual(plugin.compile_config["backend"], "inductor")
        self.assertEqual(plugin.compile_config["mode"], "max-autotune")
        self.assertTrue(plugin.compile_config["fullgraph"])

    def test_plugin_environment_variables(self):
        """Test plugin configuration via environment variables."""
        with patch.dict(os.environ, {
            "TORCHTITAN_TP_DEGREE": "4",
            "TORCHTITAN_PP_DEGREE": "2", 
            "TORCHTITAN_ENABLE_FSDP": "true",
            "TORCHTITAN_ACTIVATION_CHECKPOINTING": "true",
            "TORCHTITAN_COMPILE_MODEL": "true"
        }):
            plugin = TorchTitanPlugin()
            
            self.assertEqual(plugin.tp_degree, 4)
            self.assertEqual(plugin.pp_degree, 2)
            self.assertTrue(plugin.enable_fsdp)
            self.assertTrue(plugin.activation_checkpointing)
            self.assertTrue(plugin.compile_model)

    def test_accelerator_with_sharding_plugin(self):
        """Test Accelerator initialization with comprehensive sharding configuration."""
        plugin = TorchTitanPlugin(
            tp_degree=2,
            pp_degree=1,
            enable_fsdp=True,
            fsdp_sharding_strategy="SHARD_GRAD_OP",
            activation_checkpointing=True,
            compile_model=True
        )
        
        accelerator = Accelerator(torchtitan_plugin=plugin)
        
        self.assertEqual(accelerator.distributed_type, DistributedType.TORCHTITAN)
        self.assertEqual(accelerator.state.torchtitan_plugin.tp_degree, 2)
        self.assertEqual(accelerator.state.torchtitan_plugin.pp_degree, 1)
        self.assertTrue(accelerator.state.torchtitan_plugin.enable_fsdp)
        self.assertTrue(accelerator.state.torchtitan_plugin.activation_checkpointing)
        self.assertTrue(accelerator.state.torchtitan_plugin.compile_model)

    def test_model_preparation_with_parallelism(self):
        """Test model preparation with multi-dimensional parallelism."""
        plugin = TorchTitanPlugin(
            tp_degree=2,
            pp_degree=1,
            dp_degree=1,
            enable_fsdp=False
        )
        
        accelerator = Accelerator(torchtitan_plugin=plugin)
        model = SimpleModel()
        
        # Mock world size to be compatible with parallelism config
        with patch.object(accelerator, 'num_processes', 2):
            prepared_model = accelerator.prepare(model)
            
            # Verify model is returned and on correct device
            self.assertIsInstance(prepared_model, nn.Module)
            if torch.cuda.is_available():
                self.assertEqual(next(prepared_model.parameters()).device.type, 'cuda')

    def test_model_preparation_with_fsdp(self):
        """Test model preparation with FSDP sharding."""
        plugin = TorchTitanPlugin(
            tp_degree=1,
            pp_degree=1,
            dp_degree=2,
            enable_fsdp=True,
            fsdp_sharding_strategy="FULL_SHARD",
            fsdp_backward_prefetch="BACKWARD_PRE"
        )
        
        accelerator = Accelerator(torchtitan_plugin=plugin)
        model = SimpleModel()
        
        with patch.object(accelerator, 'num_processes', 2):
            prepared_model = accelerator.prepare(model)
            
            # Verify model is returned
            self.assertIsInstance(prepared_model, nn.Module)

    def test_model_preparation_with_activation_checkpointing(self):
        """Test model preparation with activation checkpointing."""
        plugin = TorchTitanPlugin(
            activation_checkpointing=True,
            selective_checkpointing_layers=["linear1", "linear2"]
        )
        
        accelerator = Accelerator(torchtitan_plugin=plugin)
        model = SimpleModel()
        
        prepared_model = accelerator.prepare(model)
        
        # Verify model is returned
        self.assertIsInstance(prepared_model, nn.Module)

    def test_model_preparation_with_compilation(self):
        """Test model preparation with torch.compile."""
        plugin = TorchTitanPlugin(
            compile_model=True,
            compile_config={
                "backend": "inductor",
                "mode": "reduce-overhead"
            }
        )
        
        accelerator = Accelerator(torchtitan_plugin=plugin)
        model = SimpleModel()
        
        prepared_model = accelerator.prepare(model)
        
        # Verify model is returned
        self.assertIsInstance(prepared_model, nn.Module)

    def test_optimizer_preparation_with_sharding(self):
        """Test optimizer preparation with FSDP and distributed optimization."""
        plugin = TorchTitanPlugin(
            tp_degree=2,
            pp_degree=1,
            dp_degree=1,
            enable_fsdp=True
        )
        
        accelerator = Accelerator(torchtitan_plugin=plugin)
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        with patch.object(accelerator, 'num_processes', 2):
            prepared_model, prepared_optimizer = accelerator.prepare(model, optimizer)
            
            # Verify both components are returned
            self.assertIsInstance(prepared_model, nn.Module)
            self.assertIsInstance(prepared_optimizer, torch.optim.Optimizer)

    def test_scheduler_preparation_with_parallelism(self):
        """Test scheduler preparation with multi-dimensional parallelism."""
        plugin = TorchTitanPlugin(
            tp_degree=2,
            pp_degree=2,
            dp_degree=1
        )
        
        accelerator = Accelerator(torchtitan_plugin=plugin)
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        
        with patch.object(accelerator, 'num_processes', 4):
            prepared_model, prepared_optimizer, prepared_scheduler = accelerator.prepare(
                model, optimizer, scheduler
            )
            
            # Verify all components are returned
            self.assertIsInstance(prepared_model, nn.Module)
            self.assertIsInstance(prepared_optimizer, torch.optim.Optimizer)
            self.assertIsInstance(prepared_scheduler, torch.optim.lr_scheduler._LRScheduler)

    def test_dataloader_preparation_with_data_parallelism(self):
        """Test dataloader preparation with data parallelism."""
        plugin = TorchTitanPlugin(
            tp_degree=1,
            pp_degree=1,
            dp_degree=4
        )
        
        accelerator = Accelerator(torchtitan_plugin=plugin)
        
        # Create dummy dataset and dataloader
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 64),
            torch.randint(0, 10, (100,))
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
        
        with patch.object(accelerator, 'num_processes', 4):
            prepared_dataloader = accelerator.prepare(dataloader)
            
            # Verify dataloader is returned
            self.assertIsInstance(prepared_dataloader, torch.utils.data.DataLoader)

    def test_mixed_precision_with_sharding(self):
        """Test mixed precision training with sharding configurations."""
        plugin = TorchTitanPlugin(
            enable_fsdp=True,
            fsdp_mixed_precision_policy={
                "param": "fp16",
                "reduce": "fp32",
                "buffer": "fp32"
            }
        )
        
        accelerator = Accelerator(
            torchtitan_plugin=plugin,
            mixed_precision="bf16"
        )
        
        self.assertEqual(accelerator.mixed_precision, "bf16")
        self.assertTrue(accelerator.state.torchtitan_plugin.enable_fsdp)

    def test_device_mesh_configuration(self):
        """Test device mesh configuration for multi-dimensional parallelism."""
        plugin = TorchTitanPlugin(
            tp_degree=2,
            pp_degree=2,
            dp_degree=2
        )
        
        # Test device mesh shape calculation
        dp_size, tp_size, pp_size = plugin.get_device_mesh_config(8)
        self.assertEqual((dp_size, tp_size, pp_size), (2, 2, 2))

    def test_complete_training_setup_with_sharding(self):
        """Test complete training setup with comprehensive sharding configuration."""
        plugin = TorchTitanPlugin(
            tp_degree=2,
            pp_degree=1,
            dp_degree=2,
            enable_fsdp=True,
            fsdp_sharding_strategy="FULL_SHARD",
            activation_checkpointing=True,
            compile_model=True,
            job_config={
                "training": {"batch_size": 16, "seq_len": 512},
                "model": {"name": "llama3", "layers": 12}
            }
        )
        
        accelerator = Accelerator(
            torchtitan_plugin=plugin,
            mixed_precision="bf16"
        )
        
        # Create training components
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        
        dataset = torch.utils.data.TensorDataset(
            torch.randn(200, 64),
            torch.randint(0, 10, (200,))
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
        
        with patch.object(accelerator, 'num_processes', 4):
            prepared_components = accelerator.prepare(
                model, optimizer, scheduler, dataloader
            )
            
            # Verify all components are prepared
            self.assertEqual(len(prepared_components), 4)
            prepared_model, prepared_optimizer, prepared_scheduler, prepared_dataloader = prepared_components
            
            self.assertIsInstance(prepared_model, nn.Module)
            self.assertIsInstance(prepared_optimizer, torch.optim.Optimizer)
            self.assertIsInstance(prepared_scheduler, torch.optim.lr_scheduler._LRScheduler)
            self.assertIsInstance(prepared_dataloader, torch.utils.data.DataLoader)

    def test_memory_optimization_features(self):
        """Test memory optimization features integration."""
        plugin = TorchTitanPlugin(
            enable_fsdp=True,
            fsdp_cpu_offload=True,
            activation_checkpointing=True,
            selective_checkpointing_layers=["attention", "feed_forward"]
        )
        
        accelerator = Accelerator(torchtitan_plugin=plugin)
        model = SimpleModel()
        
        prepared_model = accelerator.prepare(model)
        
        # Verify model preparation completes without errors
        self.assertIsInstance(prepared_model, nn.Module)

    def test_state_reset_with_sharding_config(self):
        """Test accelerator state reset with sharding configurations."""
        plugin = TorchTitanPlugin(
            tp_degree=4,
            pp_degree=2,
            enable_fsdp=True
        )
        
        accelerator = Accelerator(torchtitan_plugin=plugin)
        self.assertEqual(accelerator.distributed_type, DistributedType.TORCHTITAN)
        
        # Reset state
        AcceleratorState._reset_state()
        
        # Create new accelerator
        new_accelerator = Accelerator()
        self.assertEqual(new_accelerator.distributed_type, DistributedType.NO)

    def test_error_handling_in_sharding_setup(self):
        """Test error handling in sharding setup methods."""
        plugin = TorchTitanPlugin(
            tp_degree=2,
            pp_degree=2,
            enable_fsdp=True,
            activation_checkpointing=True,
            compile_model=True
        )
        
        accelerator = Accelerator(torchtitan_plugin=plugin)
        model = SimpleModel()
        
        with patch.object(accelerator, 'num_processes', 4):
            # This should handle any errors gracefully and still return a model
            prepared_model = accelerator.prepare(model)
            self.assertIsInstance(prepared_model, nn.Module)


if __name__ == "__main__":
    unittest.main()

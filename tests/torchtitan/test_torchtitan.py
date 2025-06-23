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
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


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


if __name__ == "__main__":
    unittest.main()

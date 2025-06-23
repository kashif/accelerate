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

import unittest

from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import TorchTitanPlugin


class TorchTitanIntegrationTest(unittest.TestCase):
    """Test TorchTitan plugin integration with Accelerate."""

    def tearDown(self):
        AcceleratorState._reset_state(reset_partial_state=True)

    def test_torchtitan_plugin_initialization(self):
        """Test that TorchTitanPlugin can be initialized with default values."""
        plugin = TorchTitanPlugin()
        self.assertEqual(plugin.model_name, "llama3")
        self.assertIsNotNone(plugin.job_config)

    def test_torchtitan_plugin_with_accelerator(self):
        """Test that Accelerator can be initialized with TorchTitanPlugin."""
        plugin = TorchTitanPlugin()
        accelerator = Accelerator(torchtitan_plugin=plugin)
        
        # Verify that the plugin is stored correctly in the state
        self.assertIsNotNone(accelerator.state.torchtitan_plugin)
        self.assertEqual(accelerator.state.torchtitan_plugin.model_name, "llama3")

    def test_torchtitan_plugin_custom_model(self):
        """Test TorchTitanPlugin with different model names."""
        for model_name in ["llama3", "llama4"]:
            plugin = TorchTitanPlugin(model_name=model_name)
            self.assertEqual(plugin.model_name, model_name)
            
            accelerator = Accelerator(torchtitan_plugin=plugin)
            self.assertEqual(accelerator.state.torchtitan_plugin.model_name, model_name)
            AcceleratorState._reset_state(reset_partial_state=True)

    def test_torchtitan_plugin_state_reset(self):
        """Test that plugin state is properly reset."""
        plugin1 = TorchTitanPlugin(model_name="llama3")
        accelerator1 = Accelerator(torchtitan_plugin=plugin1)
        self.assertEqual(accelerator1.state.torchtitan_plugin.model_name, "llama3")
        
        # Reset state
        AcceleratorState._reset_state(reset_partial_state=True)
        
        # Create new accelerator
        plugin2 = TorchTitanPlugin(model_name="llama4")
        accelerator2 = Accelerator(torchtitan_plugin=plugin2)
        self.assertEqual(accelerator2.state.torchtitan_plugin.model_name, "llama4")

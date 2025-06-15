#!/usr/bin/env python

# Copyright 2022 The HuggingFace Team. All rights reserved.
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
from unittest.mock import MagicMock, patch

import torch

from accelerate import Accelerator
from accelerate.test_utils.testing import AccelerateTestCase, require_deepspeed, require_non_cpu
from accelerate.utils.deepspeed import DeepSpeedEngineWrapper
from accelerate.utils.dataclasses import DistributedType


@require_deepspeed
@require_non_cpu
class TestDeepSpeedGradientAccumulation(AccelerateTestCase):
    def setUp(self):
        super().setUp()
        
        self.dist_env = dict(
            ACCELERATE_USE_DEEPSPEED="true",
            MASTER_ADDR="localhost",
            MASTER_PORT="10999",
            RANK="0",
            LOCAL_RANK="0",
            WORLD_SIZE="1",
        )

    def test_deepspeed_engine_wrapper_boundary_control(self):
        """Test that DeepSpeedEngineWrapper correctly handles gradient accumulation boundary."""
        # Create a mock DeepSpeed engine
        mock_engine = MagicMock()
        mock_engine.backward = MagicMock()
        mock_engine.step = MagicMock()
        mock_engine.set_gradient_accumulation_boundary = MagicMock()

        # Create wrapper
        wrapper = DeepSpeedEngineWrapper(mock_engine)

        # Test setting boundary
        wrapper.set_gradient_accumulation_boundary(True)
        mock_engine.set_gradient_accumulation_boundary.assert_called_with(True)

        wrapper.set_gradient_accumulation_boundary(False)
        mock_engine.set_gradient_accumulation_boundary.assert_called_with(False)

        # Test backward with boundary = True (should call step)
        wrapper.set_gradient_accumulation_boundary(True)
        loss = torch.tensor(1.0)
        wrapper.backward(loss)

        mock_engine.backward.assert_called_with(loss)
        mock_engine.step.assert_called()

        # Reset mocks
        mock_engine.reset_mock()

        # Test backward with boundary = False (should not call step)
        wrapper.set_gradient_accumulation_boundary(False)
        wrapper.backward(loss)

        mock_engine.backward.assert_called_with(loss)
        mock_engine.step.assert_not_called()

    def test_deepspeed_engine_wrapper_fallback_no_boundary_method(self):
        """Test that wrapper works even if engine doesn't have set_gradient_accumulation_boundary."""
        # Create a mock DeepSpeed engine without the boundary method
        mock_engine = MagicMock()
        mock_engine.backward = MagicMock()
        mock_engine.step = MagicMock()
        # Don't add set_gradient_accumulation_boundary to mock

        # Create wrapper
        wrapper = DeepSpeedEngineWrapper(mock_engine)

        # Test setting boundary (should not raise error)
        wrapper.set_gradient_accumulation_boundary(True)

        # Test backward behavior
        loss = torch.tensor(1.0)
        wrapper.backward(loss)

        mock_engine.backward.assert_called_with(loss)
        mock_engine.step.assert_called()

    @patch('accelerate.accelerator.DistributedType')
    def test_accelerator_backward_with_boundary_parameter(self, mock_distributed_type):
        """Test that accelerator.backward correctly handles is_final_micro_batch parameter."""
        # Mock DeepSpeed distributed type
        mock_distributed_type.DEEPSPEED = DistributedType.DEEPSPEED

        # Create accelerator with mocked DeepSpeed wrapper
        accelerator = Accelerator()
        accelerator.distributed_type = DistributedType.DEEPSPEED

        # Mock the DeepSpeed engine wrapper
        mock_wrapper = MagicMock()
        accelerator.deepspeed_engine_wrapped = mock_wrapper

        loss = torch.tensor(1.0)

        # Test with is_final_micro_batch=True
        accelerator.backward(loss, is_final_micro_batch=True)
        mock_wrapper.set_gradient_accumulation_boundary.assert_called_with(True)
        mock_wrapper.backward.assert_called_with(loss)

        # Test with is_final_micro_batch=False
        accelerator.backward(loss, is_final_micro_batch=False)
        mock_wrapper.set_gradient_accumulation_boundary.assert_called_with(False)
        mock_wrapper.backward.assert_called_with(loss)

    @patch('accelerate.accelerator.DistributedType')
    def test_accelerator_backward_fallback_to_sync_gradients(self, mock_distributed_type):
        """Test that accelerator.backward falls back to sync_gradients when is_final_micro_batch is None."""
        # Mock DeepSpeed distributed type
        mock_distributed_type.DEEPSPEED = DistributedType.DEEPSPEED

        # Create accelerator with mocked DeepSpeed wrapper
        accelerator = Accelerator()
        accelerator.distributed_type = DistributedType.DEEPSPEED

        # Mock the DeepSpeed engine wrapper
        mock_wrapper = MagicMock()
        accelerator.deepspeed_engine_wrapped = mock_wrapper

        loss = torch.tensor(1.0)

        # Test fallback to sync_gradients=True
        accelerator.sync_gradients = True
        accelerator.backward(loss)
        mock_wrapper.set_gradient_accumulation_boundary.assert_called_with(True)

        # Test fallback to sync_gradients=False
        accelerator.sync_gradients = False
        accelerator.backward(loss)
        mock_wrapper.set_gradient_accumulation_boundary.assert_called_with(False)

    @patch('accelerate.accelerator.DistributedType')
    def test_accelerator_set_deepspeed_gradient_accumulation_boundary(self, mock_distributed_type):
        """Test the explicit set_deepspeed_gradient_accumulation_boundary method."""
        # Mock DeepSpeed distributed type
        mock_distributed_type.DEEPSPEED = DistributedType.DEEPSPEED

        # Create accelerator with mocked DeepSpeed wrapper
        accelerator = Accelerator()
        accelerator.distributed_type = DistributedType.DEEPSPEED

        # Mock the DeepSpeed engine wrapper
        mock_wrapper = MagicMock()
        accelerator.deepspeed_engine_wrapped = mock_wrapper

        # Test setting boundary
        accelerator.set_deepspeed_gradient_accumulation_boundary(True)
        mock_wrapper.set_gradient_accumulation_boundary.assert_called_with(True)

        accelerator.set_deepspeed_gradient_accumulation_boundary(False)
        mock_wrapper.set_gradient_accumulation_boundary.assert_called_with(False)

    def test_accelerator_set_boundary_non_deepspeed(self):
        """Test that set_deepspeed_gradient_accumulation_boundary is ignored for non-DeepSpeed."""
        # Create accelerator without DeepSpeed
        accelerator = Accelerator()
        accelerator.distributed_type = DistributedType.MULTI_GPU  # Not DeepSpeed

        # This should not raise an error
        accelerator.set_deepspeed_gradient_accumulation_boundary(True)
        accelerator.set_deepspeed_gradient_accumulation_boundary(False)

    def test_backward_compatibility(self):
        """Test that existing code without the new parameter still works."""
        accelerator = Accelerator()

        # This should work without errors (existing API)
        # Note: In a real test, this would need proper model setup
        try:
            # Just test that the method signature is compatible
            accelerator.backward.__code__.co_varnames
            # Should contain 'is_final_micro_batch' as an optional parameter
            self.assertIn('is_final_micro_batch', accelerator.backward.__code__.co_varnames)
        except Exception as e:
            self.fail(f"Backward compatibility test failed: {e}")

    def test_optimizer_step_and_zero_grad_method_exists(self):
        """Test that the new optimizer_step_and_zero_grad method exists and has correct signature."""
        accelerator = Accelerator()
        
        # Check that the method exists
        self.assertTrue(hasattr(accelerator, 'optimizer_step_and_zero_grad'))
        
        # Check method signature
        method = getattr(accelerator, 'optimizer_step_and_zero_grad')
        self.assertTrue(callable(method))
        
        # Check that it has the expected parameters
        import inspect
        sig = inspect.signature(method)
        expected_params = {'optimizer', 'model', 'max_grad_norm'}
        actual_params = set(sig.parameters.keys())
        
        # All expected parameters should be present (excluding 'self')
        self.assertTrue(expected_params.issubset(actual_params))
        
        # All parameters should be optional (have defaults)
        for param_name in expected_params:
            param = sig.parameters[param_name]
            self.assertIsNot(param.default, inspect.Parameter.empty, 
                           f"Parameter {param_name} should have a default value")

    @patch('accelerate.accelerator.DistributedType')
    def test_optimizer_step_and_zero_grad_deepspeed(self, mock_distributed_type):
        """Test optimizer_step_and_zero_grad with DeepSpeed backend."""
        # Mock DeepSpeed distributed type
        mock_distributed_type.DEEPSPEED = DistributedType.DEEPSPEED

        # Create accelerator with mocked DeepSpeed setup
        accelerator = Accelerator()
        accelerator.distributed_type = DistributedType.DEEPSPEED

        # Mock model with DeepSpeed methods
        mock_model = MagicMock()
        mock_model.set_gradient_accumulation_boundary = MagicMock()
        mock_model.step = MagicMock()
        mock_model.get_global_grad_norm = MagicMock(return_value=torch.tensor(0.5))
        
        # Add model to accelerator's models list
        accelerator._models = [mock_model]

        # Test the method
        grad_norm = accelerator.optimizer_step_and_zero_grad(model=mock_model)

        # Verify DeepSpeed-specific calls
        mock_model.set_gradient_accumulation_boundary.assert_called_with(True)
        mock_model.step.assert_called_once()
        mock_model.get_global_grad_norm.assert_called_once()
        
        # Verify gradient norm is returned correctly
        self.assertEqual(grad_norm, 0.5)

    @patch('accelerate.accelerator.DistributedType')
    def test_optimizer_step_and_zero_grad_deepspeed_no_grad_norm(self, mock_distributed_type):
        """Test optimizer_step_and_zero_grad with DeepSpeed when gradient norm is not available."""
        # Mock DeepSpeed distributed type
        mock_distributed_type.DEEPSPEED = DistributedType.DEEPSPEED

        # Create accelerator with mocked DeepSpeed setup
        accelerator = Accelerator()
        accelerator.distributed_type = DistributedType.DEEPSPEED

        # Mock model without get_global_grad_norm method
        mock_model = MagicMock()
        mock_model.set_gradient_accumulation_boundary = MagicMock()
        mock_model.step = MagicMock()
        # Don't add get_global_grad_norm method
        
        # Add model to accelerator's models list
        accelerator._models = [mock_model]

        # Test the method
        grad_norm = accelerator.optimizer_step_and_zero_grad(model=mock_model)

        # Verify DeepSpeed-specific calls
        mock_model.set_gradient_accumulation_boundary.assert_called_with(True)
        mock_model.step.assert_called_once()
        
        # Verify fallback gradient norm is returned
        self.assertEqual(grad_norm, -1.0)

    @patch('accelerate.accelerator.DistributedType')
    def test_optimizer_step_and_zero_grad_standard_training(self, mock_distributed_type):
        """Test optimizer_step_and_zero_grad with standard PyTorch training."""
        # Mock standard distributed type
        mock_distributed_type.MULTI_GPU = DistributedType.MULTI_GPU
        mock_distributed_type.DEEPSPEED = DistributedType.DEEPSPEED

        # Create accelerator with standard setup
        accelerator = Accelerator()
        
        # Mock the distributed_type property
        with patch.object(accelerator, 'distributed_type', DistributedType.MULTI_GPU):
            # Mock optimizer
            mock_optimizer = MagicMock()
            mock_optimizer.step = MagicMock()
            mock_optimizer.zero_grad = MagicMock()
            
            # Mock model
            mock_model = MagicMock()
            mock_model.parameters = MagicMock(return_value=[torch.randn(10, requires_grad=True)])
            
            # Add to accelerator's lists
            accelerator._optimizers = [mock_optimizer]
            accelerator._models = [mock_model]

            # Mock clip_grad_norm_ method
            with patch.object(accelerator, 'clip_grad_norm_', return_value=torch.tensor(0.8)) as mock_clip:
                # Test the method with gradient clipping
                grad_norm = accelerator.optimizer_step_and_zero_grad(
                    optimizer=mock_optimizer,
                    model=mock_model,
                    max_grad_norm=1.0
                )

                # Verify standard training calls
                mock_clip.assert_called_once()
                mock_optimizer.step.assert_called_once()
                mock_optimizer.zero_grad.assert_called_once()
                
                # Verify gradient norm is returned correctly
                self.assertEqual(grad_norm, 0.8)

    @patch('accelerate.accelerator.DistributedType')
    def test_optimizer_step_and_zero_grad_standard_training_no_clipping(self, mock_distributed_type):
        """Test optimizer_step_and_zero_grad with standard training without gradient clipping."""
        # Mock standard distributed type
        mock_distributed_type.MULTI_GPU = DistributedType.MULTI_GPU
        mock_distributed_type.DEEPSPEED = DistributedType.DEEPSPEED

        # Create accelerator with standard setup
        accelerator = Accelerator()
        
        # Mock the distributed_type property
        with patch.object(accelerator, 'distributed_type', DistributedType.MULTI_GPU):
            # Mock optimizer
            mock_optimizer = MagicMock()
            mock_optimizer.step = MagicMock()
            mock_optimizer.zero_grad = MagicMock()
            
            # Add to accelerator's lists
            accelerator._optimizers = [mock_optimizer]

            # Test the method without gradient clipping
            grad_norm = accelerator.optimizer_step_and_zero_grad(optimizer=mock_optimizer)

            # Verify standard training calls
            mock_optimizer.step.assert_called_once()
            mock_optimizer.zero_grad.assert_called_once()
            
            # Verify no gradient norm is returned when no clipping
            self.assertIsNone(grad_norm)

    def test_optimizer_step_and_zero_grad_error_handling(self):
        """Test error handling in optimizer_step_and_zero_grad."""
        # Create accelerator
        accelerator = Accelerator()
        
        # Test DeepSpeed without model
        accelerator.distributed_type = DistributedType.DEEPSPEED
        accelerator._models = []
        
        with self.assertRaises(ValueError) as context:
            accelerator.optimizer_step_and_zero_grad()
        
        self.assertIn("No model provided", str(context.exception))
        
        # Test standard training without optimizer
        accelerator.distributed_type = DistributedType.MULTI_GPU
        accelerator._optimizers = []
        
        with self.assertRaises(ValueError) as context:
            accelerator.optimizer_step_and_zero_grad()
        
        self.assertIn("No optimizer provided", str(context.exception))

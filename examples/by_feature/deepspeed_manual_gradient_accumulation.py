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

"""
Example script demonstrating manual gradient accumulation control with DeepSpeed.

This script shows how to use the new `is_final_micro_batch` parameter and
`set_deepspeed_gradient_accumulation_boundary` method to have fine-grained
control over DeepSpeed's gradient accumulation behavior.
"""

import argparse

import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.test_utils import RegressionDataset, RegressionModel


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSpeed Manual Gradient Accumulation Example")
    parser.add_argument(
        "--deepspeed_config_file",
        type=str,
        default=None,
        help="path to deepspeed config file",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=16,
        help="Micro batch size per GPU",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["auto", "manual_param", "manual_method", "unified_step"],
        default="auto",
        help="Method for controlling gradient accumulation",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Initialize accelerator
    accelerator = Accelerator()

    # Create model and data
    model = RegressionModel()
    optimizer = AdamW(model.parameters(), lr=1e-3)

    # Create dataset and dataloader
    train_dataset = RegressionDataset(length=1000)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=True
    )

    # Prepare for training
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    model.train()

    if accelerator.is_main_process:
        print(f"Training with method: {args.method}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Micro batch size: {args.micro_batch_size}")

    for epoch in range(args.num_epochs):
        total_loss = 0
        num_batches = 0

        # Collect micro-batches for gradient accumulation
        micro_batches = []
        for i, batch in enumerate(train_dataloader):
            micro_batches.append(batch)

            # Process accumulated micro-batches
            if len(micro_batches) == args.gradient_accumulation_steps or i == len(train_dataloader) - 1:

                if args.method == "auto":
                    # Method 1: Use Accelerate's automatic gradient accumulation (existing behavior)
                    with accelerator.accumulate(model):
                        for batch in micro_batches:
                            outputs = model(batch["x"])
                            loss = F.mse_loss(outputs, batch["y"])
                            accelerator.backward(loss)
                            total_loss += loss.detach()

                        optimizer.step()
                        optimizer.zero_grad()

                elif args.method == "manual_param":
                    # Method 2: Use the new is_final_micro_batch parameter
                    for j, batch in enumerate(micro_batches):
                        is_final = (j == len(micro_batches) - 1)
                        outputs = model(batch["x"])
                        loss = F.mse_loss(outputs, batch["y"])
                        accelerator.backward(loss, is_final_micro_batch=is_final)
                        total_loss += loss.detach()

                    # For non-DeepSpeed setups, we still need to call optimizer.step()
                    if accelerator.distributed_type.name != "DEEPSPEED":
                        optimizer.step()
                        optimizer.zero_grad()

                elif args.method == "manual_method":
                    # Method 3: Use the explicit set_deepspeed_gradient_accumulation_boundary method
                    for j, batch in enumerate(micro_batches):
                        is_final = (j == len(micro_batches) - 1)
                        accelerator.set_deepspeed_gradient_accumulation_boundary(is_final)
                        outputs = model(batch["x"])
                        loss = F.mse_loss(outputs, batch["y"])
                        accelerator.backward(loss)
                        total_loss += loss.detach()

                    # For non-DeepSpeed setups, we still need to call optimizer.step()
                    if accelerator.distributed_type.name != "DEEPSPEED":
                        optimizer.step()
                        optimizer.zero_grad()

                elif args.method == "unified_step":
                    # Method 4: Use the unified optimizer_step_and_zero_grad method (Recommended)
                    with accelerator.accumulate(model):
                        for batch in micro_batches:
                            outputs = model(batch["x"])
                            loss = F.mse_loss(outputs, batch["y"])
                            accelerator.backward(loss)
                            total_loss += loss.detach()

                        # Unified step with gradient clipping
                        if accelerator.sync_gradients:
                            grad_norm = accelerator.optimizer_step_and_zero_grad(
                                optimizer=optimizer,
                                model=model,
                                max_grad_norm=1.0  # Optional gradient clipping
                            )
                            
                            if accelerator.is_main_process and grad_norm is not None:
                                if grad_norm == -1.0:
                                    print("DeepSpeed handled gradient norm internally")
                                else:
                                    print(f"Gradient norm: {grad_norm:.4f}")

                micro_batches = []
                num_batches += 1

        avg_loss = accelerator.gather(total_loss).mean() / num_batches

        if accelerator.is_main_process:
            print(f"Epoch {epoch + 1}/{args.num_epochs}, Average Loss: {avg_loss:.4f}")

    if accelerator.is_main_process:
        print("Training completed!")


if __name__ == "__main__":
    main()

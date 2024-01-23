import torch

model = torch.nn.Sequential(
    torch.nn.Linear(512, 1024),
    torch.nn.Linear(1024, 1024),
    torch.nn.Linear(1024, 1024),
    torch.nn.Linear(1024, 512)
)

def get_model_memory(model: torch.nn.Module):
    """
    Returns the memory usage of the given model
    """
    total_memory = 0
    for param in model.parameters():
        total_memory += param.numel() * param.element_size()
    return total_memory

print("Model Memory: {:,} bytes".format(get_model_memory(model)))

# Forward pass hooks
# https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks
class ActivationCounter:

    def __init__(self):
        self.activation_bytes = 0

    def add_activations(self, tensor):
        self.activation_bytes += tensor.numel() * tensor.element_size()

def activation_counter_hook(counter: ActivationCounter):

    def hook(self, input, output):
        counter.add_activations(output.data)

    return hook

# Recurively attach the hooks into every submodule
def register_hooks_recursive(model, counter: ActivationCounter):
  for module in model.children():
      module.register_forward_hook(activation_counter_hook(counter))
      register_hooks_recursive(module, counter)

activation_counter = ActivationCounter()
register_hooks_recursive(model, activation_counter)

# Now to find the total amount of Activation Memory consumed, we execute a forward pass through the model and then print the value of activation_counter.activation_bytes:
inputs = torch.randn(4, 512)
outputs = model(inputs)

# because the hooks only capture layer outputs, we need to add
# the size of the original input tensor separately
activation_counter.add_activations(inputs)

print("Activation Memory: {:,} bytes".format(
  activation_counter.activation_bytes
))

# idle time of different GPU's
import time
def idle_time_hook(self, device, forward=True, entering=True):
    """Creates a PyTorch hook which logs the idle time of a device."""

    def hook(*args, **kwargs):
        current_timestamp = time.time()
        last_timestamp = self.previous_timestamp.get(device, None)

        message = "{} {} pass on device {}".format(
            "Entering" if entering else "Finished",
            "forward" if forward else "backward",
            device
        )

        if entering and last_timestamp is not None:
            idle_time_ms = (current_timestamp - last_timestamp) * 1000
            self.device_idle_time[device] = (
                self.device_idle_time[device][0] + idle_time_ms,
                self.device_idle_time[device][1] + 1
            )

            message += f". Idle time: {idle_time_ms:.2f}ms"

        self.previous_timestamp[device] = current_timestamp
        self.log(message)

    return hook
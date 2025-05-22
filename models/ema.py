import torch
import copy

class EMA(object):
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        # Copy the model to create the EMA model and ensure it's on the same device
        self.ema_model = copy.deepcopy(model).to(self._get_device())
        self.ema_model.eval()  # EMA model is only for inference, so set to eval mode
        for param in self.ema_model.parameters():
            param.requires_grad = False  # EMA model parameters are not trainable

    def _get_device(self):
        # Helper function to determine the device of the model parameters
        return next(self.model.parameters()).device

    def update(self):
        # Update EMA model weights
        with torch.no_grad():
            model_params = self.model.parameters()
            ema_params = self.ema_model.parameters()
            for model_param, ema_param in zip(model_params, ema_params):
                # Ensure model_param and ema_param are on the same device
                ema_param.data = ema_param.data.to(model_param.device)  # Move EMA to model's device if needed
                ema_param.data = self.decay * ema_param.data + (1 - self.decay) * model_param.data

    def save(self, filepath):
        # Save the EMA model state_dict to a file
        torch.save(self.ema_model.state_dict(), filepath)

    def load(self, filepath):
        # Load the EMA model state_dict from a file
        self.ema_model.load_state_dict(torch.load(filepath, map_location=self._get_device()))

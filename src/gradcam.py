import torch


class GradCAM:
    """Records a layer's activations and gradients to build a class heatmap."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, _module, _inputs, output):
        self.activations = output
        output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad

    def __call__(self, x, class_idx=None):
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1).item()
        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1))[0]
        cam = cam / (cam.max() + 1e-8)
        confidence = torch.softmax(logits, dim=1)[0, class_idx].item()
        return cam.detach().cpu(), class_idx, confidence

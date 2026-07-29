import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, target_class):
        self.model.eval()

        output = self.model(input_tensor)

        self.model.zero_grad()
        target = output[0][target_class]
        target.backward(retain_graph=True)

        gradients = self.gradients.detach()
        activations = self.activations.detach()

        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)

        cam = torch.sum(weights * activations, dim=1, keepdim=True)

        cam = F.relu(cam)

        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)

        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam_max = cam.max()
        if cam_max != 0:
            cam = cam / cam_max

        return cam

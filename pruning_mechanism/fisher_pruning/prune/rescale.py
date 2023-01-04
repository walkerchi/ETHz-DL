from tqdm import tqdm
import torch

from utils.linalg import lsmr_cupy_solver
from utils.arch import (
    get_layers,
    get_mha_proj,
    get_ffn2,
    hijack_input,
    MaskNeurons,
    remove_padding,
    collect_layer_inputs,
)


@torch.no_grad()
def get_mha_lstsq(
    model,
    config,
    teacher_inputs,
    teacher_neuron_mask,
    student_inputs,
    student_head_mask,
    student_neuron_mask,
    layer_idx,
    device
):
    num_attention_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    attention_head_size = int(hidden_size / num_attention_heads)

    nonzero_heads = student_head_mask[layer_idx].nonzero().flatten()
    num_nonzero_heads = nonzero_heads.shape[0]

    layer = get_layers(model)[layer_idx]
    mha_proj = get_mha_proj(model, layer_idx)
    weights_per_head = mha_proj.weight.t().view(num_attention_heads, -1, hidden_size)
    weights_per_head = weights_per_head.index_select(dim=0, index=nonzero_heads)

    inputs = []
    handle = hijack_input(mha_proj, inputs)

    ATA = torch.zeros(num_nonzero_heads+1, num_nonzero_heads+1).to(device)
    ATB = torch.zeros(num_nonzero_heads+1).to(device)

    model.eval()
    for teacher_batch, student_batch in zip(teacher_inputs, student_inputs):
        student_batch[2] = student_head_mask[layer_idx]# .view(1, -1, 1, 1)

        # Get the outputs of the teacher model
        with MaskNeurons(model, teacher_neuron_mask):
            layer(teacher_batch[0], None, None, head_mask=teacher_batch[2])
        hidden_states, = inputs.pop(0) # removed , input_tensor
        teacher_output = mha_proj(hidden_states) + teacher_batch[0]# + input_tensor

        inputs.pop(0) # edit remove entry added through line above

        # Get the outputs of the student model
        with MaskNeurons(model, student_neuron_mask):
            layer(student_batch[0], None, None, head_mask=student_batch[2])
        hidden_states, = inputs.pop(0)  # removed , input_tensor
        # input_tensor = remove_padding(input_tensor, attention_mask)

        # hidden_states = hidden_states.view(-1, num_attention_heads, attention_head_size)
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(batch_size, -1, num_attention_heads, attention_head_size)
        hidden_states = hidden_states.view(-1, num_attention_heads, attention_head_size)
        hidden_states = hidden_states.permute(1, 0, 2)
        hidden_states = hidden_states.index_select(dim=0, index=nonzero_heads)

        outputs_per_head = hidden_states @ weights_per_head
        outputs_per_head = outputs_per_head.view(num_nonzero_heads, -1)

        A = outputs_per_head.t()
        A = torch.cat([A, torch.ones(A.shape[0], 1, device=device)], dim=1)
        B = teacher_output - mha_proj.bias - student_batch[0]  # input_tensor
        B = B.flatten()
        ATA += A.t() @ A
        ATB += A.t() @ B
        del A
        del B

    handle.remove()
    return ATA, ATB


@torch.no_grad()
def get_ffn_lstsq(
    model,
    config,
    teacher_inputs,
    teacher_neuron_mask,
    student_inputs,
    student_head_mask,
    student_neuron_mask,
    layer_idx,
    device,
    cls_only=False,
):
    layer = get_layers(model)[layer_idx]
    ffn2 = get_ffn2(model, layer_idx)
    weights_per_neuron = ffn2.weight.t()#dense.weight.t()

    nonzero_neurons = student_neuron_mask[layer_idx].nonzero().flatten()
    num_neurons = nonzero_neurons.shape[0]
    weights_per_neuron = weights_per_neuron.index_select(dim=0, index=nonzero_neurons).to(device)
    W = weights_per_neuron @ weights_per_neuron.t()
    residuals = []
    inputs = []
    handle = hijack_input(ffn2, inputs)
    handle_res = hijack_input(layer.layer_norm2, residuals)

    ATA = torch.zeros(num_neurons, num_neurons).to(device)
    ATB = torch.zeros(num_neurons).to(device)

    model.eval()
    for teacher_batch, student_batch in zip(teacher_inputs, student_inputs):
        student_batch[2] = student_head_mask[layer_idx] # .view(1, -1, 1, 1)

        # Get the outputs of the teacher model
        teacher_output, = layer(teacher_batch[0], None, None, head_mask=None)
        inputs.pop(0)
        residuals.pop(0)
        if cls_only:
            teacher_output = teacher_output[:, 0, :]

        # Get the outputs of the student model
        with MaskNeurons(model, student_neuron_mask):
            lo = layer(student_batch[0], None, None, head_mask=student_batch[2])
        hidden_states, = inputs.pop(0) # removed , input_tensor
        residual, = residuals.pop(0)
        if cls_only:
            hidden_states = hidden_states[:, 0, :]
        hidden_states = hidden_states.view(hidden_states.shape[0]*hidden_states.shape[1], hidden_states.shape[2])
        hidden_states = hidden_states.t()
        hidden_states = hidden_states.index_select(dim=0, index=nonzero_neurons)
        hst = hidden_states.t().to(device)
        B = (teacher_output - ffn2.bias - residual).to(device)
        batch_size = B.shape[0]
        seq_size = B.shape[1]
        num = B.numel()
        B = B.view(batch_size*seq_size, B.shape[2])
        B = B.flatten()
        A = torch.zeros(seq_size, weights_per_neuron.shape[1], weights_per_neuron.shape[0], device=device)
        # Do it image-wise to save working memory
        for j in range(batch_size):
            Bj = B[j*(num//batch_size):(j+1)*(num//batch_size)]
            for i in range(j*seq_size, (j+1)*seq_size):
                A[i-j*seq_size] = hst[i]*weights_per_neuron.t()

            A_view = A.view(-1, A.shape[2])
            ATA += A_view.t() @ A_view
            ATB += A_view.t() @ Bj
        # ATA += W * (hidden_states @ hidden_states.t()) # ??
        # ATB += (hidden_states.unsqueeze(1) @ (weights_per_neuron @ B.t()).unsqueeze(2)).squeeze()
    handle.remove()
    handle_res.remove()
    return ATA.cpu(), ATB.cpu()


@torch.no_grad()
def rescale_mask(
    model,
    config,
    teacher_head_mask,
    teacher_neuron_mask,
    student_head_mask,
    student_neuron_mask,
    dataloader,
    device,
    classification_task=False,
):
    num_hidden_layers = config.num_hidden_layers
    rescaled_head_mask = student_head_mask.clone()
    rescaled_neuron_mask = student_neuron_mask.clone()

    for layer_idx in tqdm(range(num_hidden_layers)):
        teacher_inputs = collect_layer_inputs(
            model,
            teacher_head_mask,
            teacher_neuron_mask,
            layer_idx,
            device,
            prev_inputs=dataloader if layer_idx == 0 else teacher_inputs,
        )
        student_inputs = collect_layer_inputs(
            model,
            rescaled_head_mask,
            rescaled_neuron_mask,
            layer_idx,
            device,
            prev_inputs=dataloader if layer_idx == 0 else student_inputs,
        )

        if torch.count_nonzero(student_head_mask[layer_idx]) != 0 and layer_idx != 0:
            ATA, ATB = get_mha_lstsq(
                model.vision_model,
                config,
                teacher_inputs,
                teacher_neuron_mask,
                student_inputs,
                rescaled_head_mask,
                rescaled_neuron_mask,
                layer_idx,
                device
            )
            scale_factor, success = lsmr_cupy_solver(ATA, ATB)
            if not success:
                print('No success in solving lsmr problem.') # break
            scale_factor = scale_factor[:-1]
            if scale_factor.max() > 10 or scale_factor.min() < -10:
                print('abnormal scale factor head:', scale_factor.min(), scale_factor.max())
                #break
            nonzero_heads = rescaled_head_mask[layer_idx].nonzero().flatten()
            for index, scale in zip(nonzero_heads, scale_factor):
                rescaled_head_mask[layer_idx][index] *= scale

        if torch.count_nonzero(student_neuron_mask[layer_idx]) != 0:
            cls_only = classification_task and (layer_idx == num_hidden_layers - 1)
            ATA, ATB = get_ffn_lstsq(
                model.vision_model,
                config,
                teacher_inputs,
                teacher_neuron_mask,
                student_inputs,
                rescaled_head_mask,
                rescaled_neuron_mask,
                layer_idx,
                device,
                cls_only=cls_only,
            )
            scale_factor, success = lsmr_cupy_solver(ATA, ATB) # lsmr_cupy_solver(ATA, ATB)
            if not success:
                print('No success in solving lsmr problem.') # break
            if scale_factor.max() > 10 or scale_factor.min() < -10:
                print('abnormal scale factor neuron:', scale_factor.min(), scale_factor.max())
                #break
            nonzero_neurons = rescaled_neuron_mask[layer_idx].nonzero().flatten()
            for index, scale in zip(nonzero_neurons, scale_factor):
                rescaled_neuron_mask[layer_idx][index] *= scale

    return rescaled_head_mask, rescaled_neuron_mask

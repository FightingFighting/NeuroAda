import numpy as np
import torch
import torch.nn as nn
import random
from tqdm import tqdm
import wandb

# #with head
# def select_perCell(model, time_para=1, mag_or_mag=None, return_=None):
#     statistic = {}
#     new_masks = {}
#
#     for name, param in model.named_parameters():
#         if "LayerNorm" in name or "position_embeddings" in name or "token_type_embeddings" in name or "word_embeddings" in name:
#             new_mask = None
#             param.requires_grad_(False)
#             trainable_param = 0
#             total_para = len(param.data.reshape(-1))
#             statistic[name] = [trainable_param, total_para]
#             print(name, ": ", trainable_param, "/", total_para, "(", np.round((trainable_param / total_para) * 100, 4),"%)", param.data.shape)
#         else:
#             if 'classifier' in name or "bias" in name:
#                 new_mask = torch.zeros_like(param.data, device="cpu")
#             else:
#                 if mag_or_mag == "mag":
#                     tensor = param.data.cpu()
#                 elif mag_or_mag == "gra":
#                     tensor = param.grad.data.cpu()
#
#                 if time_para > tensor.size(1):
#                     topk_values, topk_indices = torch.topk(abs(tensor), tensor.size(1), dim=1, largest=True)
#                 else:
#                     topk_values, topk_indices = torch.topk(abs(tensor), time_para, dim=1, largest=True)
#
#                 new_mask = torch.ones_like(tensor, device="cpu")
#                 new_mask.scatter_(1, topk_indices, 0)
#
#
#
#             trainable_param = len(new_mask.reshape(-1))-len(torch.nonzero(new_mask))
#             total_para = len(new_mask.reshape(-1))
#             new_masks[name] = new_mask
#
#             statistic[name] = [trainable_param, total_para]
#             print(name, ": ", trainable_param, "/", total_para, "(", np.round((trainable_param / total_para) * 100, 4),"%)", new_mask.shape)
#
#     print("---------------------------------------------------------------")
#     trainable_withouthead = 0
#     total_withouthead = 0
#     trainable_head = 0
#     total_head = 0
#     for na, [trainable_p, t_p] in statistic.items():
#         # print(na, ":", trainable_p, "/", t_p,"(",np.round((trainable_p/t_p)*100, 4), "%)")
#         if "classifier" not in na:
#             trainable_withouthead = trainable_withouthead + trainable_p
#             total_withouthead = total_withouthead + t_p
#         else:
#             trainable_head = trainable_head + trainable_p
#             total_head = total_head + t_p
#     print("---------------------------------------------------------------")
#
#     print("---------------------------------------------------------------")
#     print("Trainable parameter / Total (without head): ", trainable_withouthead, "/", total_withouthead, "(", np.round((trainable_withouthead/total_withouthead)*100,4), "%)")
#     print("Trainable parameter / Total (head): ", trainable_head, "/", total_head, "(", np.round((trainable_head/total_head)*100,4), "%)")
#     print("Trainable parameter / Total (total): ", trainable_head+trainable_withouthead, "/", total_head+total_withouthead, "(", np.round(((trainable_head+trainable_withouthead)/(total_head+total_withouthead))*100,4), "%)")
#
#     print("#######################################################################")
#
#
#     if return_ == "mask":
#         new_masks_temp={}
#         for name, mask in new_masks.items():
#             new_masks_temp[name]=mask.to("cuda")
#         return new_masks_temp
#     elif return_ == "position":
#         positions={}
#         for name, param in model.named_parameters():
#             if name not in new_masks:
#                 continue
#             if 'classifier' in name or "bias" in name:
#                 param.requires_grad_(True)
#             else:
#                 param.requires_grad_(False)
#                 mask = new_masks[name]
#                 positions[name] = torch.nonzero(mask == 0, as_tuple=False)
#
#         return positions
#

#
#
# def select_perCell(model, time_para=1, mag_or_mag=None, return_=None):
#     statistic = {}
#     new_masks = {}
#     print("===================== all parameters ================================")
#     for name, param in model.named_parameters():
#         print(name, param.size())
#     print("=====================================================")
#
#     for name, param in model.named_parameters():
#         if "Norm" in name or "norm" in name or "embeddings" in name or "embed_tokens" in name:
#             new_mask = None
#             param.requires_grad_(False)
#             trainable_param = 0
#             total_para = len(param.data.reshape(-1))
#             statistic[name] = [trainable_param, total_para]
#             print(name, ": ", trainable_param, "/", total_para, "(", np.round((trainable_param / total_para) * 100, 4),"%)", param.data.shape)
#         else:
#             if 'classifier' in name or "bias" in name or "lm_head" in name:
#                 new_mask = torch.zeros_like(param.data, device="cpu")
#             else:
#                 if mag_or_mag == "mag":
#                     tensor = param.data.cpu()
#                 elif mag_or_mag == "gra":
#                     tensor = param.grad.data.cpu()
#
#                 if time_para > tensor.size(1):
#                     topk_values, topk_indices = torch.topk(abs(tensor), tensor.size(1), dim=1, largest=True)
#                 else:
#                     topk_values, topk_indices = torch.topk(abs(tensor), time_para, dim=1, largest=True)
#
#                 new_mask = torch.ones_like(tensor, device="cpu")
#                 new_mask.scatter_(1, topk_indices, 0)
#
#
#
#             trainable_param = len(new_mask.reshape(-1))-len(torch.nonzero(new_mask))
#             total_para = len(new_mask.reshape(-1))
#             new_masks[name] = new_mask
#
#             statistic[name] = [trainable_param, total_para]
#             print(name, ": ", trainable_param, "/", total_para, "(", np.round((trainable_param / total_para) * 100, 4),"%)", new_mask.shape)
#
#     print("---------------------------------------------------------------")
#     trainable_withouthead = 0
#     total_withouthead = 0
#     trainable_head = 0
#     total_head = 0
#     for na, [trainable_p, t_p] in statistic.items():
#         # print(na, ":", trainable_p, "/", t_p,"(",np.round((trainable_p/t_p)*100, 4), "%)")
#         if "classifier" in na or "lm_head" in na:
#             trainable_head = trainable_head + trainable_p
#             total_head = total_head + t_p
#         else:
#             trainable_withouthead = trainable_withouthead + trainable_p
#             total_withouthead = total_withouthead + t_p
#     print("---------------------------------------------------------------")
#
#     print("---------------------------------------------------------------")
#     print("Trainable parameter / Total (without head): ", trainable_withouthead, "/", total_withouthead, "(", np.round((trainable_withouthead/total_withouthead)*100,4), "%)")
#     print("Trainable parameter / Total (head): ", trainable_head, "/", total_head, "(", np.round((trainable_head/total_head)*100,4), "%)")
#     print("Trainable parameter / Total (total): ", trainable_head+trainable_withouthead, "/", total_head+total_withouthead, "(", np.round(((trainable_head+trainable_withouthead)/(total_head+total_withouthead))*100,4), "%)")
#
#     print("#######################################################################")
#
#
#     if return_ == "mask":
#         new_masks_temp={}
#         for name, mask in new_masks.items():
#             new_masks_temp[name]=mask.to("cuda")
#         return new_masks_temp
#     elif return_ == "position":
#         positions={}
#         for name, param in model.named_parameters():
#             if name not in new_masks:
#                 continue
#             if 'classifier' in name or "bias" in name or "lm_head" in name:
#                 param.requires_grad_(True)
#             else:
#                 param.requires_grad_(False)
#                 mask = new_masks[name]
#                 positions[name] = torch.nonzero(mask == 0, as_tuple=False)
#
#         return positions




#without head and with head
# def select_perCell(model, time_para=1, mag_or_mag=None, return_=None, selecthead=None, selectallhead=None):
#     statistic = {}
#     new_masks = {}
#     print("===================== all parameters ================================")
#     for name, param in model.named_parameters():
#         print(name, param.size())
#     print("=====================================================")
#
#     for name, param in model.named_parameters():
#         if "Norm" in name or "norm" in name or "embeddings" in name or "embed_tokens" in name or ("lm_head" in name and not selecthead) or ("classifier" in name and not selecthead):
#             new_mask = None
#             param.requires_grad_(False)
#             trainable_param = 0
#             total_para = len(param.data.reshape(-1))
#             statistic[name] = [trainable_param, total_para]
#             print(name, ": ", trainable_param, "/", total_para, "(", np.round((trainable_param / total_para) * 100, 4),"%)", param.data.shape)
#         else:
#             if "bias" in name or (selectallhead and "classifier" in name) or (selectallhead and "lm_head" in name):
#                 new_mask = torch.zeros_like(param.data, device="cpu")
#             else:
#                 if mag_or_mag == "mag":
#                     tensor = param.data.cpu()
#                 elif mag_or_mag == "gra":
#                     tensor = param.grad.data.cpu()
#
#                 if time_para > tensor.size(1):
#                     topk_values, topk_indices = torch.topk(abs(tensor), tensor.size(1), dim=1, largest=True)
#                 else:
#                     topk_values, topk_indices = torch.topk(abs(tensor), time_para, dim=1, largest=True)
#
#                 new_mask = torch.ones_like(tensor, device="cpu")
#                 new_mask.scatter_(1, topk_indices, 0)
#
#
#
#             trainable_param = len(new_mask.reshape(-1))-len(torch.nonzero(new_mask))
#             total_para = len(new_mask.reshape(-1))
#             new_masks[name] = new_mask
#
#             statistic[name] = [trainable_param, total_para]
#             print(name, ": ", trainable_param, "/", total_para, "(", np.round((trainable_param / total_para) * 100, 4),"%)", new_mask.shape)
#
#     print("---------------------------------------------------------------")
#     trainable_withouthead = 0
#     total_withouthead = 0
#     trainable_head = 0
#     total_head = 0
#     for na, [trainable_p, t_p] in statistic.items():
#         # print(na, ":", trainable_p, "/", t_p,"(",np.round((trainable_p/t_p)*100, 4), "%)")
#         if "classifier" in na or "lm_head" in na:
#             trainable_head = trainable_head + trainable_p
#             total_head = total_head + t_p
#         else:
#             trainable_withouthead = trainable_withouthead + trainable_p
#             total_withouthead = total_withouthead + t_p
#     print("---------------------------------------------------------------")
#
#     print("---------------------------------------------------------------")
#     print("Trainable parameter / Total (without head): ", trainable_withouthead, "/", total_withouthead, "(", np.round((trainable_withouthead/total_withouthead)*100,4), "%)")
#     print("Trainable parameter / Total (head): ", trainable_head, "/", total_head, "(", np.round((trainable_head/total_head)*100,4), "%)")
#     print("Trainable parameter / Total (total): ", trainable_head+trainable_withouthead, "/", total_head+total_withouthead, "(", np.round(((trainable_head+trainable_withouthead)/(total_head+total_withouthead))*100,4), "%)")
#
#     print("#######################################################################")
#
#
#     if return_ == "mask":
#         new_masks_temp={}
#         for name, mask in new_masks.items():
#             new_masks_temp[name]=mask.to("cuda")
#         return new_masks_temp
#     elif return_ == "position":
#         positions={}
#         for name, param in model.named_parameters():
#             if name not in new_masks:
#                 continue
#             if "bias" in name or (selectallhead and "classifier" in name) or (selectallhead and "lm_head" in name):
#                 param.requires_grad_(True)
#             else:
#                 param.requires_grad_(False)
#                 mask = new_masks[name]
#                 positions[name] = torch.nonzero(mask == 0, as_tuple=False)
#
#         return positions
def select_perCell(model, time_para=1, mag_or_mag=None, return_=None, selectallhead=True, target_modules=None, tune_bias=False, peft_type=None, ig_layer_index=-1):
    statistic = {}
    new_masks = {}
    print("===================== all parameters ================================")
    for name, param in model.named_parameters():
        print(name, param.size())
    print("=====================================================")

    for name, param in model.named_parameters():

        selecte=False
        for tag in target_modules:
            if tag in name:
                selecte=True
                break
        if not tune_bias and "bias" in name and "classifier" not in name:
            selecte=False

        if "model.layers." in name and ig_layer_index!=-1:
            layer_index=int(name.split("model.layers.")[-1].split(".")[0])
            if layer_index == ig_layer_index:
                selecte=False

        if not selecte:
            new_mask = None
            param.requires_grad_(False)
            trainable_param = 0
            total_para = len(param.data.reshape(-1))
            statistic[name] = [trainable_param, total_para]
            print(name, ": ", trainable_param, "/", total_para, "(", np.round((trainable_param / total_para) * 100, 4),"%)", param.data.shape)
        else:
            if "bias" in name or (selectallhead and "classifier" in name) or (selectallhead and "lm_head" in name):
                new_mask = torch.zeros_like(param.data, device="cpu")
            else:
                if mag_or_mag == "mag":
                    tensor = param.data.cpu()
                elif mag_or_mag == "gra":
                    tensor = param.grad.data.cpu()

                if "reverse" in peft_type:
                    largest=False
                else:
                    largest=True


                # if "certainCell" in peft_type:
                #     ratio = float(peft_type.split('_')[-1])
                #     non_activate_num=int((1-ratio)*len(tensor))
                #     time_para_current = int(time_para/ratio)
                # else:
                #     time_para_current = time_para

                if "certainCell" in peft_type:
                    ratio = float(peft_type.split('_')[-1])
                    non_activate_num=int((1-ratio)*len(tensor))

                time_para_current = time_para


                if time_para_current > tensor.size(1):
                    topk_values, topk_indices = torch.topk(abs(tensor), tensor.size(1), dim=1, largest=largest)
                else:
                    topk_values, topk_indices = torch.topk(abs(tensor), time_para_current, dim=1, largest=largest)

                new_mask = torch.ones_like(tensor, device="cpu")
                new_mask.scatter_(1, topk_indices, 0)

                if "randomCell" in peft_type:
                    B, N = new_mask.shape
                    perm = torch.argsort(torch.rand(B, N), dim=1)
                    new_mask = new_mask.gather(1, perm)

                if "randomLayer" in peft_type:

                    B, N = new_mask.shape
                    total = B * N

                    # 随机采样 zero 的扁平索引
                    zero_indices = torch.randperm(total)[:time_para_current * B]

                    # 构造新的 mask，并将对应位置设为 0
                    new_mask = torch.ones_like(new_mask)
                    new_mask.view(-1)[zero_indices] = 0

                if "certainCell" in peft_type:
                    row_indices = torch.randperm(new_mask.size(0))[:non_activate_num]
                    new_mask[row_indices] = 1.0



            trainable_param = len(new_mask.reshape(-1))-len(torch.nonzero(new_mask))
            total_para = len(new_mask.reshape(-1))
            new_masks[name] = new_mask

            statistic[name] = [trainable_param, total_para]
            print(name, ": ", trainable_param, "/", total_para, "(", np.round((trainable_param / total_para) * 100, 4),"%)", new_mask.shape)



    print("---------------------------------------------------------------")
    trainable_withouthead = 0
    total_withouthead = 0
    trainable_head = 0
    total_head = 0
    for na, [trainable_p, t_p] in statistic.items():
        # print(na, ":", trainable_p, "/", t_p,"(",np.round((trainable_p/t_p)*100, 4), "%)")
        if "classifier" in na or "lm_head" in na:
            trainable_head = trainable_head + trainable_p
            total_head = total_head + t_p
        else:
            trainable_withouthead = trainable_withouthead + trainable_p
            total_withouthead = total_withouthead + t_p
    print("---------------------------------------------------------------")

    print("---------------------------------------------------------------")
    print("Trainable parameter / Total (without head): ", trainable_withouthead, "/", total_withouthead, "(", np.round((trainable_withouthead/total_withouthead)*100,4), "%)")
    print("Trainable parameter / Total (head): ", trainable_head, "/", total_head, "(", np.round((trainable_head/total_head)*100,4), "%)")
    print("Trainable parameter / Total (total): ", trainable_head+trainable_withouthead, "/", total_head+total_withouthead, "(", np.round(((trainable_head+trainable_withouthead)/(total_head+total_withouthead))*100,4), "%)")
    print("Trainable parameter (without head) / Total (with head): ", trainable_withouthead, "/", total_head+total_withouthead, "(", np.round((trainable_withouthead/(total_head+total_withouthead))*100,4), "%)")

    if "randomGlobal" in peft_type:
        for m_name in new_masks:
            new_masks[m_name] = torch.ones_like(new_masks[m_name], device="cpu")

        flat_tensors = []
        shapes_info = {}
        start_idx = 0

        for m_name, matrix in new_masks.items():
            B, N = matrix.shape
            numel = B * N
            shapes_info[m_name] = {
                "start": start_idx,
                "shape": (B, N)
            }
            flat_tensors.append(matrix.reshape(-1))
            start_idx += numel

        total_tensor = torch.cat(flat_tensors)  # Shape: [total_elements]

        trainable_param_num = trainable_head + trainable_withouthead
        zero_indices = random.sample(range(total_tensor.numel()), trainable_param_num)
        total_tensor[zero_indices] = 0

        for m_name, info in shapes_info.items():
            start = info["start"]
            B, N = info["shape"]
            new_masks[m_name] = total_tensor[start:start + B * N].reshape(B, N)


        print("+++++++++++++++++++++++random global+++++++++++++++++++++++++++++++++")
        for name, mask_ in new_masks.items():
            trainable_param = len(mask_.reshape(-1)) - len(torch.nonzero(mask_))
            total_para = len(mask_.reshape(-1))
            print(name, ": ", trainable_param, "/", total_para, "(", np.round((trainable_param / total_para) * 100, 4),
                  "%)", mask_.shape)

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


    if "randomBlock" in peft_type:
        for m_name in new_masks:
            new_masks[m_name] = torch.ones_like(new_masks[m_name], device="cpu")

        all_mask={}
        for layer_num in range(32):
            block_masks={}
            for m_name in new_masks:
                if f"model.layers.{layer_num}." in m_name:
                    block_masks[m_name]=new_masks[m_name]

            flat_tensors = []
            shapes_info = {}
            start_idx = 0
            trainable_param_num=0

            for m_name, matrix in block_masks.items():
                B, N = matrix.shape
                numel = B * N
                trainable_param_num+=B*time_para
                shapes_info[m_name] = {
                    "start": start_idx,
                    "shape": (B, N)
                }
                flat_tensors.append(matrix.reshape(-1))
                start_idx += numel

            total_tensor = torch.cat(flat_tensors)  # Shape: [total_elements]

            zero_indices = random.sample(range(total_tensor.numel()), trainable_param_num)
            total_tensor[zero_indices] = 0

            for m_name, info in shapes_info.items():
                start = info["start"]
                B, N = info["shape"]
                block_masks[m_name] = total_tensor[start:start + B * N].reshape(B, N)


            all_mask.update(block_masks)
        new_masks=all_mask


        print("+++++++++++++++++++++++random global+++++++++++++++++++++++++++++++++")
        for name, mask_ in new_masks.items():
            trainable_param = len(mask_.reshape(-1)) - len(torch.nonzero(mask_))
            total_para = len(mask_.reshape(-1))
            print(name, ": ", trainable_param, "/", total_para, "(", np.round((trainable_param / total_para) * 100, 4),
                  "%)", mask_.shape)

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


    if wandb.run is not None:
        wandb.log({f"Parameters Num. / Trainable parameter (without head)": trainable_withouthead}, step=0)
        wandb.log({f"Parameters Num. / Trainable parameter (head)": trainable_head}, step=0)
        wandb.log({f"Parameters Num. / Trainable parameter (total)": trainable_head+trainable_withouthead}, step=0)
        wandb.log({f"Parameters Num. / Head parameter": total_head}, step=0)
        wandb.log({f"Parameters Num. / Total parameter without head": total_withouthead}, step=0)
        wandb.log({f"Parameters Num. / Total parameter": total_head+total_withouthead}, step=0)
        wandb.log({f"Parameters (%) / Trainable parameter in Total (without head) (%)": np.round((trainable_withouthead/total_withouthead)*100,4)}, step=0)
        wandb.log({f"Parameters (%) / Trainable parameter in Total (head) (%)": np.round((trainable_head/total_head)*100,4)}, step=0)
        wandb.log({f"Parameters (%) / Trainable parameter in Total (total) (%)": np.round(((trainable_head+trainable_withouthead)/(total_head+total_withouthead))*100,4)}, step=0)
        wandb.log({f"Parameters (%) / Trainable parameter (without head) in Total (with head) (%)": np.round((trainable_withouthead/(total_head+total_withouthead))*100,4)}, step=0)


    print("#######################################################################")


    if return_ == "mask":
        new_masks_temp={}
        for name, mask in new_masks.items():
            new_masks_temp[name]=mask.to("cuda")
        return new_masks_temp
    elif return_ == "position":
        positions={}
        for name, param in model.named_parameters():
            if name not in new_masks:
                continue
            if ("bias" in name and tune_bias) or (selectallhead and "classifier" in name) or (selectallhead and "lm_head" in name):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
                mask = new_masks[name]
                positions[name] = torch.nonzero(mask == 0, as_tuple=False)

        return positions

def generate_new_params(position, dtype):
    new_params = {}

    for name, pos in position.items():
        param_size = len(pos)
        new_param = nn.Parameter(torch.zeros(param_size, dtype=dtype))
        new_param.requires_grad = True

        new_params[name+"_new"] = {
            'param': new_param,
            'positions': pos
        }

    return new_params



def generate_MaskOrPosition(model, peft_type, dataloader, time_para, target_modules, selectallhead,tuneBias,ig_layer_index=-1):

    if "add" in peft_type:
        if "mag" in peft_type or "random" in peft_type:
            position = select_perCell(model, time_para=time_para, mag_or_mag="mag", return_="position", selectallhead=selectallhead, target_modules=target_modules,tune_bias=tuneBias, peft_type=peft_type, ig_layer_index=ig_layer_index)
        elif "gra" in peft_type:
            model.train()
            for step, batch in enumerate(tqdm(dataloader)):
                batch.to(model.device)
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

            position = select_perCell(model, time_para=time_para, mag_or_mag="gra", return_="position", selectallhead=selectallhead, target_modules=target_modules,tune_bias=tuneBias, peft_type=peft_type)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.zero_()
        print("++++++++++++++++++++++++++position+++++++++++++++++++++++++++")
        for m, _ in position.items():
            print(m)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        return None, position
    else:
        if peft_type=="perCell_mag":
            mask = select_perCell(model, time_para=time_para, mag_or_mag="mag", return_="mask", selectallhead=selectallhead, target_modules=target_modules,tune_bias=tuneBias, peft_type=peft_type)
        elif peft_type=="perCell_gra":
            model.train()
            for step, batch in enumerate(tqdm(dataloader)):
                batch.to(model.device)
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
            mask = select_perCell(model, time_para=time_para, mag_or_mag="gra", return_="mask", selectallhead=selectallhead, target_modules=target_modules,tune_bias=tuneBias, peft_type=peft_type)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.zero_()
        print("++++++++++++++++++++++++++Mask+++++++++++++++++++++++++++")
        for m, _ in mask.items():
            print(m)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        return mask, None


def mask_gradient(model, mask):

    for name, param in model.named_parameters():
        if param.requires_grad:
            mask_ = mask[name]
            grad_tensor = param.grad.data
            grad_tensor = torch.where(mask_ == 1.0, mask_ - 1.0, grad_tensor)
            param.grad.data = grad_tensor
from collections import Counter
from .select_para import generate_new_params
import torch
import torch.nn.functional as F

def get_nb_trainable_parameters(model) -> tuple[int, int]:
    r"""
    Returns the number of trainable parameters and the number of all parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param



def appr_position(matrix, para_time):

    flattened_matrix = matrix.flatten()

    number_counts = Counter(flattened_matrix.tolist())
    top_k_numbers = number_counts.most_common(para_time)

    indices=[]
    for num, count in top_k_numbers:
        print(f"Col num: {num}, Frequency: {count}/ {matrix.size(0)}({count/matrix.size(0)})")
        indices.append(num)

    return indices



def update_model(model, positions, time_num=None, peft_type=None, dtype=None):

    new_params = generate_new_params(positions, dtype)

    print("++++++++++++++++++++++++++++++++++++++++++ New parameters ++++++++++++++++++++++++++++++++++++++++++++++++")
    indices_all=[]
    for name, new_param in new_params.items():
        module_name = name.replace("_new", "").rsplit('.', 1)[0]

        submodule = model
        for part in module_name.split('.'):
            submodule = getattr(submodule, part)

        if peft_type == "perCell_mag_add_appr":
            new_param["param"].data = new_param["param"].data.reshape(-1, time_num).to(submodule.weight.device)
            pos = new_param["positions"][:,1].reshape(-1, time_num)
            indices=appr_position(pos, time_num)
            indices_all.append(indices)
            submodule.register_buffer("indices", torch.tensor(indices))
            submodule.register_parameter('weight_new', new_param["param"])

            def modified_forward(x, original_weight=submodule.weight, new_weight=submodule.weight_new,
                                 bias=submodule.bias, indices=submodule.indices):

                result = F.linear(x, original_weight, bias)
                if x.dtype!=new_weight.dtype:
                    result += F.linear(x[..., indices].to(new_weight.dtype).contiguous(), new_weight, bias).to(x.dtype)
                else:
                    result += F.linear(x[..., indices].contiguous(), new_weight, bias)

                return result  # 20.61s 33.37g
        else:
            new_param["param"].data = new_param["param"].data.to(submodule.weight.device)
            submodule.register_parameter('weight_new', new_param["param"])
            new_param["positions"].data = (new_param["positions"][:, 0] * submodule.weight.size(1) + new_param["positions"][:, 1]).data.to(submodule.weight.device)
            submodule.register_buffer("weight_position_new", new_param["positions"])

            def modified_forward(x, original_weight=submodule.weight, new_weight=submodule.weight_new,
                                         new_weight_position=submodule.weight_position_new,
                                         bias = submodule.bias):

                return F.linear(x, original_weight.flatten().scatter_add(0, new_weight_position, new_weight).view_as(original_weight), bias)


        print(name, new_param["param"].shape, new_param["positions"].shape, submodule.weight.shape)


        submodule.forward = modified_forward
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    params_to_optimize = [info['param'] for info in new_params.values()]
    params_to_optimize_names = [name for name in new_params.keys()]
    for name, param in model.named_parameters():
        if param.requires_grad and name not in params_to_optimize_names:
            params_to_optimize.append(param)
            params_to_optimize_names.append(name)


    print("++++++++++++++++++++++++++++++++++++++++++ Parameters to Optimizer ++++++++++++++++++++++++++++++++++++++++++++++++")
    for name in params_to_optimize_names:
        print(name)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


    if peft_type == "perCell_mag_add_appr":
        return params_to_optimize_names, params_to_optimize, indices_all
    else:
        return params_to_optimize_names, params_to_optimize, None




def merge_model(model, tunable_para_keys, peft_type, merge_weight=False):

    for name in tunable_para_keys:
        if "new" not in name:
            continue
        module_name = name.replace("_new", "").rsplit('.', 1)[0]

        submodule = model
        for part in module_name.split('.'):
            submodule = getattr(submodule, part)


        if peft_type == "perCell_mag_add_appr":
            pass
        else:
            if submodule.weight.dtype != submodule.weight_new.dtype:
                if merge_weight:
                    submodule.weight.data = (submodule.weight.to(submodule.weight_new.dtype).flatten().scatter_add(0, submodule.weight_position_new, submodule.weight_new).view_as(submodule.weight)).to(submodule.weight.dtype).contiguous()
                else:
                    submodule.weight.data = submodule.weight.to(submodule.weight_new.dtype).flatten().scatter_add(0, submodule.weight_position_new, submodule.weight_new).view_as(submodule.weight).contiguous()
            else:
                submodule.weight.data = submodule.weight.flatten().scatter_add(0, submodule.weight_position_new, submodule.weight_new).view_as(submodule.weight).contiguous()

            del submodule.weight_position_new,
            del submodule.weight_new


            def modified_forward(x, original_weight=submodule.weight, bias = submodule.bias):
                #10.49s 51.52g
                if original_weight.dtype!=x.dtype:
                    return F.linear(x.to(original_weight.dtype), original_weight, bias).to(x.dtype)
                else:
                    return F.linear(x, original_weight, bias)


        submodule.forward = modified_forward



# 比较模型参数是否相等
def compare_models(model1, model2):
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    for key in state_dict1:
        if torch.equal(state_dict1[key], state_dict2[key]):
            print(f"{True} Parameter {key} is equal.")
        else:
            print(f"{False} Parameter {key} is not equal.")

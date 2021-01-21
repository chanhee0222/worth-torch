# Calculate pair-wise cosine similarity.
import random

import math
import torch
import torch.nn as nn
import numpy as np

from utils.scope import name_scope


def pairwise_cos_sim(inputs, axis, mask_self=True):
    with name_scope("cos_sim_a%d" %axis):
        assert axis in [0, 1]

        trans_a = (axis == 0)
        trans_b = not trans_a
        mat_a = inputs.transpose(-1, -2) if trans_a else inputs
        mat_b = inputs.transpose(-1, -2) if trans_b else inputs

        dot_prod = mat_a.matmul(mat_b)

        magnitude = inputs.pow(2.0).sum(axis, keepdim=True).sqrt()
        mag_a = magnitude.transpose(-1, -2) if trans_a else magnitude
        mag_b = magnitude.transpose(-1, -2) if trans_b else magnitude
        magnitude = mag_a.matmul(mag_b)
        cos_sim = dot_prod / (magnitude + 1e-6)

        if mask_self:
            # Mask out cosine similarity values with itself, which is always 1.
            eye_mat = cos_sim.new_empty(cos_sim.shape)
            nn.init.eye_(eye_mat)
            cos_mask = cos_sim.new_ones(cos_sim.shape) - eye_mat
            cos_sim = cos_sim * cos_mask

        assert inputs.shape[1-axis] == cos_sim.shape[0] == cos_sim.shape[1]

    return cos_sim


def test_pairwise_cos_sim():
    source = torch.randn(100, 128)

    for axis in range(2):
        output_1 = pairwise_cos_sim(source, axis)
        size = output_1.shape[0]
        output_2 = np.ndarray([size, size])

        input = source if axis==1 else source.t()

        for r in range(size):
            for c in range(size):
                vec_a = input[r]
                vec_b = input[c]

                cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                sim = cos(vec_a, vec_b)
                output_2[r][c] = sim

        np.testing.assert_almost_equal(output_1, output_2, decimal=6)


def merge_rows(weight, proj, row1_idx, row2_idx):
    was_sparse = proj.is_sparse
    if was_sparse:
        proj = proj.to_dense()
    # Reduce the number of rows in weight, by merging row2_idx to row1_idx. (row2_idx is removed)
    target_trans_mat = proj.t()

    vec1 = weight[row1_idx]
    vec2 = weight[row2_idx]

    # vec1_mag = vec1.pow(2.0).sum().sqrt()
    # vec2_mag = vec2.pow(2.0).sum().sqrt()
    # sim = vec1.dot(vec2) /vec1_mag / vec2_mag

    ratio = vec2.sum() / (vec1.sum() + 1e-10)
    # This scale makes sure that pairwise cosine similarity of the columns stays the same.
    scale = math.sqrt(math.pow(ratio, 2.0) + 1)

    # diff = torch.sum(torch.abs(torch.abs(vec1 * ratio) - torch.abs(vec2)))
    # print(sim)
    # print(diff)

    # print("Merging %d(M=%f) and %d(M=%f). S=%f, R=%f" % (row1_idx, vec1_mag, row2_idx, vec2_mag, sim, ratio))

    new_weight = torch.cat((weight[:row2_idx], weight[row2_idx + 1:]), dim=0)
    new_trans_mat = torch.cat(
        (target_trans_mat[:row2_idx], target_trans_mat[row2_idx + 1:]), dim=0)
    row1_idx = row1_idx if row2_idx > row1_idx else row1_idx - 1
    new_weight[row1_idx] = new_weight[row1_idx] * scale
    new_trans_mat[row1_idx] = (new_trans_mat[row1_idx] + target_trans_mat[row2_idx] * ratio) / scale

    new_trans_mat = new_trans_mat.t()
    if was_sparse:
        new_trans_mat = new_trans_mat.to_sparse()

    new_weight = new_weight.contiguous()
    new_trans_mat = new_trans_mat.contiguous()

    return new_weight, new_trans_mat


def test_merge_rows():
    test_size = 100
    input_size = 500
    output_size = 400
    scale = 1.0
    w_size = [input_size - 2, output_size]
    weight = torch.randn(w_size, dtype=torch.float32)
    # weight = torch.arange(0, w_size[0] * w_size[1], dtype=torch.float32).view(w_size)
    bias = torch.randn(w_size[1], dtype=weight.dtype)
    # bias = None

    # Manipulate the weight.
    weight = torch.cat((weight, torch.unsqueeze(weight[0] * 1.7, 0)), dim=0)
    weight = torch.cat((weight, torch.unsqueeze(weight[0] * -0.7, 0)), dim=0)
    print("Weight:")
    print(weight)

    proj = weight.new_empty([input_size, input_size])
    nn.init.eye_(proj)

    random_input = torch.randn([test_size, input_size], dtype=weight.dtype, device=weight.device)
    # random_input = torch.ones([test_size, input_size], dtype=weight.dtype, device=weight.device)

    from models.ffnn import WorthDense
    layer = WorthDense(output_size, False, True, "outer_l2_sum")
    _ = layer(random_input)
    layer.weight = nn.Parameter(weight, requires_grad=True)
    layer.bias = nn.Parameter(bias, requires_grad=True)
    init_output = layer(random_input)

    # init_output = nn.functional.linear(random_input, weight.mul(scale).t(), bias)
    if bias is not None:
        init_cos_sim = pairwise_cos_sim(torch.cat((weight, bias.unsqueeze(0)), dim=0), 0)
    else:
        init_cos_sim = pairwise_cos_sim(weight.mul(scale), 0)

    init_other_cos_sim = pairwise_cos_sim(weight.mul(scale)[:-2], 1)

    assert init_cos_sim.shape[0] == output_size

    new_weight, new_trans_mat = merge_rows(weight, proj, 0, input_size - 1)
    new_weight, new_trans_mat = merge_rows(new_weight, new_trans_mat, 0, input_size - 2)
    print("Proj:")
    print(new_trans_mat)
    print("Input:")
    print(random_input)
    proj_input = nn.functional.linear(random_input, new_trans_mat.t(), None)
    print("Proj input:")
    print(proj_input)
    print("Proj weight:")
    print(new_weight)
    layer.weight = nn.Parameter(new_weight, requires_grad=True)
    layer.in2weight_mat = new_trans_mat

    new_output = layer(random_input)

    # new_output = nn.functional.linear(proj_input, new_weight.mul(scale).t(), bias)

    if bias is not None:
        new_cos_sim = pairwise_cos_sim(torch.cat((new_weight, bias.unsqueeze(0)), dim=0), 0)
    else:
        new_cos_sim = pairwise_cos_sim(new_weight.mul(scale), 0)

    new_other_cos_sim = pairwise_cos_sim(new_weight.mul(scale), 1)

    assert new_cos_sim.shape[0] == output_size

    print("Init cos sim:")
    print(init_cos_sim)

    print("New cos sim:")
    print(new_cos_sim)

    print("Gold:")
    print(init_output)
    print("Output:")
    print(new_output)

    np.testing.assert_almost_equal(init_output.tolist(), new_output.tolist(), decimal=5)
    np.testing.assert_almost_equal(init_cos_sim.tolist(), new_cos_sim.tolist(), decimal=5)
    np.testing.assert_almost_equal(init_other_cos_sim.tolist(), new_other_cos_sim.tolist(), decimal=7)


def test_merge_rows2():
    test_size = 100
    input_size = 50
    output_size = 40
    scale = 1.0
    w_size = [input_size - 2, output_size]
    weight = torch.randn(w_size, dtype=torch.float32)
    # weight = torch.arange(0, w_size[0] * w_size[1], dtype=torch.float32).view(w_size)
    bias = torch.randn(w_size[1], dtype=weight.dtype)
    # bias = None

    # Manipulate the weight.
    weight = torch.cat((torch.ones([1, output_size], dtype=weight.dtype, device=weight.device), weight), dim=0)
    # weight[0][-1] = 1.0
    weight = torch.cat((weight, torch.unsqueeze(weight[0] * 1.1, 0)), dim=0)
    weight[-1][-1] *= 1.001
    # weight = torch.cat((weight, torch.unsqueeze(weight[0] * -0.7, 0)), dim=0)
    # print("Weight:")
    # print(weight)

    proj = weight.new_empty([input_size, input_size])
    nn.init.eye_(proj)

    # random_input = torch.randn([test_size, input_size], dtype=weight.dtype, device=weight.device)
    random_input = torch.ones([test_size, input_size], dtype=weight.dtype, device=weight.device)

    from models.ffnn import WorthDense
    layer = WorthDense(output_size, False, True, "outer_l2_sum")
    _ = layer(random_input)
    layer.weight = nn.Parameter(weight, requires_grad=True)
    layer.bias = nn.Parameter(bias, requires_grad=True)
    layer.in2weight_mat = proj
    init_output = layer(random_input)

    if bias is not None:
        init_cos_sim = pairwise_cos_sim(torch.cat((weight, bias.unsqueeze(0)), dim=0), 0)
    else:
        init_cos_sim = pairwise_cos_sim(weight.mul(scale), 0)

    assert init_cos_sim.shape[0] == output_size

    # print("Input:")
    # print(random_input)

    layer.optimize_weight(0, 0.9999)
    print(layer)
    new_weight = layer.weight

    new_output = layer(random_input)

    if bias is not None:
        new_cos_sim = pairwise_cos_sim(torch.cat((new_weight, bias.unsqueeze(0)), dim=0), 0)
    else:
        new_cos_sim = pairwise_cos_sim(new_weight.mul(scale), 0)

    assert new_cos_sim.shape[0] == output_size

    # print("Init cos sim:")
    # print(init_cos_sim)
    #
    # print("New cos sim:")
    # print(new_cos_sim)
    #
    # print("Gold:")
    # print(init_output)
    # print("Output:")
    # print(new_output)

    print(torch.abs(init_output - new_output).sum())

    np.testing.assert_almost_equal(init_output.tolist(), new_output.tolist(), decimal=4)
    np.testing.assert_almost_equal(init_cos_sim.tolist(), new_cos_sim.tolist(), decimal=4)


def merge_cols(weight, proj, col1_idx, col2_idx, bias):
    was_sparse = proj.is_sparse
    if was_sparse:
        proj = proj.to_dense()
    # Reduce the number of columns in weight, by merging col2_idx to col1_idx. (col2_idx is removed)
    target_trans_mat = proj
    if bias is not None:
        weight = torch.cat((weight, bias.unsqueeze(0)), dim=0)

    weight = weight.t()

    vec1 = weight[col1_idx]
    vec2 = weight[col2_idx]

    # vec1_mag = torch.sqrt(vec1.pow(2).sum() + 1e-6)
    # vec2_mag = torch.sqrt(vec2.pow(2).sum() + 1e-6)
    # sim = vec1.dot(vec2) /vec1_mag / vec2_mag

    ratio = vec2.sum() / (vec1.sum() + 1e-10)

    # Calculate the cosine scale without the bias.
    cos_scale = math.sqrt(math.pow(ratio, 2.0) + 1)

    # print("Merging %d(M=%f) and %d(M=%f). S=%f, R=%f" % (col1_idx, vec1_mag, col2_idx, vec2_mag, sim, ratio))

    new_weight = torch.cat((weight[:col2_idx], weight[col2_idx + 1:]), dim=0)
    new_trans_mat = torch.cat(
        (target_trans_mat[:col2_idx], target_trans_mat[col2_idx + 1:]), dim=0)
    col1_idx = col1_idx if col2_idx > col1_idx else col1_idx - 1
    new_weight[col1_idx] = new_weight[col1_idx] * cos_scale
    new_trans_mat[col1_idx] = (new_trans_mat[col1_idx] + target_trans_mat[col2_idx] * ratio) / cos_scale

    new_weight = new_weight.t()
    if bias is not None:
        new_weight, new_bias = new_weight[:-1], new_weight[-1]
    else:
        new_bias = None

    if was_sparse:
        new_trans_mat = new_trans_mat.to_sparse()

    new_weight = new_weight.contiguous()
    new_trans_mat = new_trans_mat.contiguous()
    if new_bias is not None:
        new_bias = new_bias.contiguous()

    return new_weight, new_trans_mat, new_bias


def test_merge_cols():
    test_size = 100
    input_size = 500
    output_size = 400
    scale = 0.01
    w_size = [input_size, output_size - 2]
    weight = torch.randn(w_size, dtype=torch.float32)
    # weight = torch.arange(0, w_size[0] * w_size[1], dtype=torch.float32).view(w_size)
    bias = torch.randn(w_size[1], dtype=weight.dtype)
    # bias = torch.ones(w_size[1], dtype=weight.dtype, device=weight.device)
    # bias = None

    if bias is not None:
        weight = torch.cat((weight, bias.unsqueeze(0)), dim=0)

    # Manipulate the weight.
    weight = weight.t()
    weight = torch.cat((weight, torch.unsqueeze(weight[0] * -1.0, 0)), dim=0)
    weight = torch.cat((weight, torch.unsqueeze(weight[0] * 1.6, 0)), dim=0)
    weight = weight.t()

    if bias is not None:
        weight, bias = weight[:-1], weight[-1]

    print("Weight:")
    print(weight)

    proj = weight.new_empty([output_size, output_size])
    nn.init.eye_(proj)

    random_input = torch.randn([test_size, input_size], dtype=weight.dtype, device=weight.device)
    # random_input = torch.ones([test_size, input_size], dtype=weight.dtype, device=weight.device)

    init_output = nn.functional.linear(random_input, weight.mul(scale).t(), bias)
    init_cos_sim = pairwise_cos_sim(weight.mul(scale), 1)
    assert init_cos_sim.shape[0] == input_size

    new_weight, new_trans_mat, new_bias = merge_cols(weight, proj, 0, output_size - 1, bias)
    new_weight, new_trans_mat, new_bias = merge_cols(new_weight, new_trans_mat, 0, output_size - 2, new_bias)
    print("Proj:")
    print(new_trans_mat)
    print("Input:")
    print(random_input)
    print("Proj weight:")
    print(new_weight)
    new_output = nn.functional.linear(random_input, new_weight.mul(scale).t(), new_bias)
    print("New output:")
    print(new_output)
    new_cos_sim = pairwise_cos_sim(new_weight.mul(scale), 1)
    assert new_cos_sim.shape[0] == input_size

    proj_output = nn.functional.linear(new_output, new_trans_mat.t(), None)
    print("Proj output:")
    print(proj_output)

    print("Init cos sim:")
    print(init_cos_sim)

    print("New cos sim:")
    print(new_cos_sim)

    print("Gold:")
    print(init_output)
    print("Output:")
    print(proj_output)

    np.testing.assert_almost_equal(init_output.tolist(), proj_output.tolist(), decimal=5)
    np.testing.assert_almost_equal(init_cos_sim.tolist(), new_cos_sim.tolist(), decimal=5)


def triu_argmax(input):
    # Make sure the input is a square matrix.
    assert len(input.shape) == 2
    assert input.shape[0] == input.shape[1]

    mask = torch.ones_like(input).triu(1).bool()
    masked = torch.masked_select(input, mask)
    k = torch.argmax(masked).cpu()
    n = input.shape[0]
    row = n - 2 - int(np.sqrt(-8 * k + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
    col = k + row + 1 - n*(n-1)/2 + (n-row)*((n-row)-1)/2
    row = torch.tensor(row, dtype=torch.long, device=input.device)
    col = torch.tensor(col, dtype=torch.long, device=input.device)

    return row, col


def test_triu_argmax():
    size = 5
    a = torch.randn(size, size)
    print(a)

    mask = torch.ones_like(a).triu(1).bool()
    masked = torch.masked_select(a, mask)
    max_val = masked.max()

    row, col = triu_argmax(a)

    print(max_val, row, int(col))

    assert a[row][col] == max_val


def triu_argtopk(input, k):
    # Make sure the input is a square matrix.
    assert len(input.shape) == 2
    assert input.shape[0] == input.shape[1]

    mask = torch.ones_like(input).triu(1).bool()
    masked = torch.masked_select(input, mask)
    assert masked.shape[0] >= k
    _, indices = torch.topk(masked, k)
    result = []
    for idx in indices:
        n = input.shape[0]
        row = n - 2 - int(np.sqrt(-8 * idx + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
        col = idx + row + 1 - n*(n-1)/2 + (n-row)*((n-row)-1)/2
        result.append((row, col))

    return result


def test_triu_argtopk():
    size = 50
    k = 30
    a = torch.randn(size, size)
    print(a)

    mask = torch.ones_like(a).triu(1).bool()
    masked = torch.masked_select(a, mask)
    max_vals, _ = masked.topk(k)
    print(max_vals)

    indices = triu_argtopk(a, k)
    print(indices)

    for i, (row, col) in enumerate(indices):
        assert a[row][col] == max_vals[i]


def triu_argmin(input):
    # Make sure the input is a square matrix.
    assert len(input.shape) == 2
    assert input.shape[0] == input.shape[1]

    mask = torch.ones_like(input).triu(1).bool()
    masked = torch.masked_select(input, mask)
    k = torch.argmin(masked).cpu()
    n = input.shape[0]
    row = n - 2 - int(np.sqrt(-8 * k + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
    col = k + row + 1 - n*(n-1)/2 + (n-row)*((n-row)-1)/2

    return row, col


def test_triu_argmin():
    size = 5
    a = torch.randn(size, size)
    print(a)

    mask = torch.ones_like(a).triu(1).bool()
    masked = torch.masked_select(a, mask)
    max_val = masked.min()

    row, col = triu_argmin(a)

    print(max_val, row, int(col))

    assert a[row][col] == max_val


def merge_feature_space(weight2out, in2weight):
    was_w2o_sparse = weight2out.is_sparse
    if was_w2o_sparse:
        weight2out = weight2out.to_dense()

    was_i2w_sparse = in2weight.is_sparse
    if was_i2w_sparse:
        in2weight = in2weight.to_dense()

    weight2out_shape = weight2out.shape
    in2weight_shape = in2weight.shape

    # in2weight_ne = in2weight.ne(0.0).type(in2weight.dtype)
    weight2out_ne = weight2out.gt(0.0).type(weight2out.dtype)
    merge_pairs = weight2out_ne.sum(dim=1).ge(2.0).nonzero()

    if len(merge_pairs) > 0:
        target_ids = weight2out[merge_pairs[0]].gt(0.0)
    else:
        weight2out_ne = weight2out.lt(0.0).type(weight2out.dtype)
        merge_pairs = weight2out_ne.sum(dim=1).ge(2.0).nonzero()
        if len(merge_pairs) > 0:
            target_ids = weight2out[merge_pairs[0]].lt(0.0)

    if len(merge_pairs) > 0:
        column_ids = torch.where(target_ids.squeeze())[0]
        # print(column_ids)
        # print(merge_pairs[0])

        idx1, idx2 = sorted(column_ids[:2])

        a1 = weight2out[merge_pairs[0],idx1]
        a2 = weight2out[merge_pairs[0],idx2]

        i2w_del_vec = in2weight[idx2]
        new_i2w_chunks = [
            in2weight[:idx2],
            in2weight[idx2+1:]
        ]
        new_i2w = torch.cat(new_i2w_chunks, dim=0)
        new_i2w[idx1] = new_i2w[idx1] * a1 / (a1 + a2) + i2w_del_vec * a2 / (a1 + a2)

        w2o_del_vec = weight2out[:, idx2]
        new_w2o_chunks = [
            weight2out[:, :idx2],
            weight2out[:, idx2+1:]
        ]
        new_w2o = torch.cat(new_w2o_chunks, dim=1)
        new_w2o[:, idx1] = new_w2o[:, idx1] + w2o_del_vec

        if was_w2o_sparse:
            new_w2o = new_w2o.to_sparse()

        if was_i2w_sparse:
            new_i2w = new_i2w.to_sparse()

        return new_w2o, new_i2w

    return None


def merge_feature_space_old(weight2out, in2weight):
    was_w2o_sparse = weight2out.is_sparse
    if was_w2o_sparse:
        weight2out = weight2out.to_dense()

    was_i2w_sparse = in2weight.is_sparse
    if was_i2w_sparse:
        in2weight = in2weight.to_dense()

    weight2out_shape = weight2out.shape
    in2weight_shape = in2weight.shape

    in2weight_ne = in2weight.ne(0.0).type(in2weight.dtype)
    weight2out_ne = weight2out.gt(0.0).type(weight2out.dtype)
    merge_pairs = torch.matmul(weight2out_ne, in2weight_ne).ge(2.0).nonzero()
    if len(merge_pairs) == 0:
        weight2out_ne = weight2out.lt(0.0).type(weight2out.dtype)
        merge_pairs = torch.matmul(weight2out_ne, in2weight_ne).ge(2.0).nonzero()

    if len(merge_pairs) > 0:
        idx1, idx2 = merge_pairs[0]
        target_mask = weight2out_ne[idx1] * in2weight_ne[:, idx2]
        target_ids = target_mask.nonzero()
        print(target_ids)
        # print(weight2out[idx1])
        # print(in2weight[:, idx2])

        k_i2w = torch.sum(weight2out[idx1] * in2weight[:, idx2] * target_mask.type(weight2out.dtype))
        k_w2o = torch.sum(weight2out[idx1] * target_mask.type(weight2out.dtype))
        k_i2w = k_i2w / k_w2o
        # print(k_i2w)
        # print(k_w2o)

        new_w2o_chunks = []
        new_i2w_chunks = []
        for idx in reversed(target_ids):
            new_w2o_chunks.append(weight2out[:, idx+1:])
            weight2out = weight2out[:, :idx]

            new_i2w_chunks.append(in2weight[idx+1:])
            in2weight = in2weight[:idx]

        new_w2o_chunks.append(weight2out)
        new_i2w_chunks.append(in2weight)

        new_w2o_chunks = list(reversed(new_w2o_chunks))
        new_i2w_chunks = list(reversed(new_i2w_chunks))

        vec = torch.zeros(weight2out_shape[0], dtype=new_w2o_chunks[0].dtype, device=new_w2o_chunks[0].device)
        vec[idx1] = k_w2o
        new_w2o_chunks.append(vec.unsqueeze(1))

        vec = torch.zeros(in2weight_shape[1], dtype=new_i2w_chunks[0].dtype, device=new_i2w_chunks[0].device)
        vec[idx2] = k_i2w
        new_i2w_chunks.append(vec.unsqueeze(0))

        new_w2o = torch.cat(new_w2o_chunks, 1)
        new_i2w = torch.cat(new_i2w_chunks, 0)

        if was_w2o_sparse:
            new_w2o = new_w2o.to_sparse()

        if was_i2w_sparse:
            new_i2w = new_i2w.to_sparse()

        return new_w2o, new_i2w

    return None


def test_merge_feature_space():
    in_size = 15
    feature_size = 80
    out_size = 24
    activation = nn.ReLU()

    test_size = 100

    def merge_helper(mat, axis, target_cnt):
        if axis == 1:
            mat = mat.t()
        while mat.shape[0] > target_cnt:
            indices = random.sample(range(mat.shape[0]), 2)
            indices = sorted(indices)
            new_mat = torch.cat((mat[:indices[1]], mat[indices[1]+1:]), 0)
            new_mat[indices[0]] = new_mat[indices[0]] + mat[indices[1]]
            mat = new_mat

        if axis == 1:
            mat = mat.t()

        return mat

    weight2out_mat = torch.randn(feature_size, dtype=torch.float32).diag()
    weight2out_mat = merge_helper(weight2out_mat, 0, in_size)

    in2weight_mat = torch.randn(feature_size, dtype=torch.float32).diag()
    in2weight_mat = merge_helper(in2weight_mat, 1, out_size)

    # print(weight2out_mat)
    # print(in2weight_mat)

    random_input = torch.randn([test_size, in_size], dtype=torch.float32)

    layer_out = torch.matmul(random_input, weight2out_mat)
    layer_out = activation(layer_out)
    init_output = torch.matmul(layer_out, in2weight_mat)

    result = merge_feature_space(weight2out_mat, in2weight_mat)
    if result is None:
        return
    weight2out_mat, in2weight_mat = result

    layer_out = torch.matmul(random_input, weight2out_mat)
    layer_out = activation(layer_out)
    new_output = torch.matmul(layer_out, in2weight_mat)

    np.testing.assert_almost_equal(init_output.tolist(), new_output.tolist(), decimal=5)


if __name__ == "__main__":
    for i in range(1000):
        test_merge_feature_space()
    # test_merge_rows2()
    # test_merge_cols()

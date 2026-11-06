"""
Vertex orders for making surfaces: the Hilbert curve (`hilbert_encode`, `hilbert_decode`)
and the Morton curve (`morton_encode`, `morton_decode`). Needs torch, the `mesh` extra.
"""

import torch
from typing import Optional, Union


#######################################################################################
# Hilbert order
#######################################################################################
#
# Hilbert Order
# Modified from https://github.com/PrincetonLIPS/numpy-hilbert-curve
#
# Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Kaixin Xu
# Please cite our work if the code is helpful to you.


def right_shift(binary, k=1, axis=-1):
    """Right shift an array of binary values.

    Parameters:
    -----------
     binary: An ndarray of binary values.

     k: The number of bits to shift. Default 1.

     axis: The axis along which to shift.  Default -1.

    Returns:
    --------
     Returns an ndarray with zero prepended and the ends truncated, along
     whatever axis was specified.

    Called by `binary2gray` and `gray2binary`.
    In: binary (n, nbits) uint8 or bool tensor; k, int or a 1-element int tensor; axis, int.
    Out: a tensor of binary's shape and dtype.
    """

    # If we're shifting the whole thing, just return zeros.
    if binary.shape[axis] <= k:
        return torch.zeros_like(binary)

    # Determine the slicing pattern to eliminate just the last one.
    slicing = [slice(None)] * len(binary.shape)  # one slice per axis of binary
    slicing[axis] = slice(None, -k)
    shifted = torch.nn.functional.pad(  # binary's shape: zeros, then binary less its last k along the last axis
        binary[tuple(slicing)], (k, 0), mode="constant", value=0
    )

    return shifted


def binary2gray(binary, axis=-1):
    """Convert an array of binary values into Gray codes.

    This uses the classic X ^ (X >> 1) trick to compute the Gray code.

    Parameters:
    -----------
     binary: An ndarray of binary values.

     axis: The axis along which to compute the gray code. Default=-1.

    Returns:
    --------
     Returns an ndarray of Gray codes.

    Called by `hilbert_decode`.
    In: binary (n, nbits) uint8 tensor of bits; axis, int. Out: (n, nbits) bool tensor.
    """
    shifted = right_shift(binary, axis=axis)  # (n, nbits) uint8

    # Do the X ^ (X >> 1) trick.
    gray = torch.logical_xor(binary, shifted)  # (n, nbits) bool

    return gray


def gray2binary(gray, axis=-1):
    """Convert an array of Gray codes back into binary values.

    Parameters:
    -----------
     gray: An ndarray of gray codes.

     axis: The axis along which to perform Gray decoding. Default=-1.

    Returns:
    --------
     Returns an ndarray of binary values.

    Called by `hilbert_encode`.
    In: gray (n, nbits) uint8 tensor of Gray-code bits; axis, int. Out: (n, nbits) bool tensor.
    """

    # Loop the log2(bits) number of times necessary, with shift and xor.
    shift = 2 ** (torch.Tensor([gray.shape[axis]]).log2().ceil().int() - 1)  # (1,) int32 tensor: the first shift
    while shift > 0:
        gray = torch.logical_xor(gray, right_shift(gray, shift))  # (n, nbits) bool
        shift = torch.div(shift, 2, rounding_mode="floor")
    return gray


def hilbert_encode(locs, depth, num_dims=3):
    """Encode an array of locations in a hypercube into a Hilbert integer.

    This is a vectorized-ish version of the Hilbert curve implementation by John
    Skilling as described in:

    Skilling, J. (2004, April). Programming the Hilbert curve. In AIP Conference
      Proceedings (Vol. 707, No. 1, pp. 381-387). American Institute of Physics.

    Params:
    -------
     locs - An ndarray of locations in a hypercube of num_dims dimensions, in
            which each dimension runs from 0 to 2**depth-1.  The shape can
            be arbitrary, as long as the last dimension of the same has size
            num_dims.

     num_dims - The dimensionality of the hypercube. Integer.

     depth - There are 2^depth points per axis of the hypercube. Integer.

    Returns:
    --------
     The output is a flat ndarray of int64 integers, one per location: the
     input's shape excluding the last dimension, which needs to be num_dims,
     flattened into one (0-d for a single location).

    Called by `lodos.col.sdf.io.reorder_v`, as its `encode` for `encoding="hilbert"`.
    In: locs (n, num_dims) int tensor, each coordinate in [0, 2**depth); depth, int; num_dims, int.
    Out: (n,) int64 tensor, the keys (0-d for n = 1).
    """

    # Keep around the original shape for later.
    orig_shape = locs.shape  # torch.Size: (n, num_dims)
    bitpack_mask = 1 << torch.arange(0, 8).to(locs.device)  # (8,) int64: 1, 2, ..., 128
    bitpack_mask_rev = bitpack_mask.flip(-1)  # (8,) int64: 128, ..., 1

    if orig_shape[-1] != num_dims:
        raise ValueError("""
      The shape of locs was surprising in that the last dimension was of size
      %d, but num_dims=%d.  These need to be equal.
      """ % (orig_shape[-1], num_dims))

    if num_dims * depth > 63:
        raise ValueError("""
      num_dims=%d and num_bits=%d for %d bits total, which can't be encoded
      into a int64.  Are you sure you need that many points on your Hilbert
      curve?
      """ % (num_dims, depth, num_dims * depth))

    # Treat the location integers as 64-bit unsigned and then split them up into
    # a sequence of uint8s.  Preserve the association by dimension.
    locs_uint8 = locs.long().view(torch.uint8).reshape((-1, num_dims, 8)).flip(-1)  # (n, num_dims, 8) uint8, MSB first

    # Now turn these into bits and truncate to num_bits.
    gray = (  # (n, num_dims, depth) uint8: each coordinate's low depth bits, most significant first
        locs_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0)
        .byte()
        .flatten(-2, -1)[..., -depth:]
    )

    # Run the decoding process the other way.
    # Iterate forwards through the bits.
    for bit in range(0, depth):  # bit: int
        # Iterate forwards through the dimensions.
        for dim in range(0, num_dims):  # dim: int
            # Identify which ones have this bit active.
            mask = gray[:, dim, bit]  # (n,) uint8

            # Where this bit is on, invert the 0 dimension for lower bits.
            gray[:, 0, bit + 1 :] = torch.logical_xor(
                gray[:, 0, bit + 1 :], mask[:, None]
            )

            # Where the bit is off, exchange the lower bits with the 0 dimension.
            to_flip = torch.logical_and(  # (n, depth - bit - 1) bool
                torch.logical_not(mask[:, None]).repeat(1, gray.shape[2] - bit - 1),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            gray[:, dim, bit + 1 :] = torch.logical_xor(
                gray[:, dim, bit + 1 :], to_flip
            )
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

    # Now flatten out.
    gray = gray.swapaxes(1, 2).reshape((-1, depth * num_dims))

    # Convert Gray back to binary.
    hh_bin = gray2binary(gray)  # (n, depth * num_dims) bool

    # Pad back out to 64 bits.
    extra_dims = 64 - depth * num_dims  # int
    padded = torch.nn.functional.pad(hh_bin, (extra_dims, 0), "constant", 0)  # (n, 64) bool

    # Convert binary values into uint8s.
    hh_uint8 = (  # (n, 8) uint8: the key's bytes, least significant first
        (padded.flip(-1).reshape((-1, 8, 8)) * bitpack_mask)
        .sum(2)
        .squeeze()
        .type(torch.uint8)
    )

    # Convert uint8s into uint64s.
    hh_uint64 = hh_uint8.view(torch.int64).squeeze()  # (n,) int64

    return hh_uint64


def hilbert_decode(idx, depth, num_dims=3):
    """Decode an array of Hilbert integers into locations in a hypercube.

    This is a vectorized-ish version of the Hilbert curve implementation by John
    Skilling as described in:

    Skilling, J. (2004, April). Programming the Hilbert curve. In AIP Conference
      Proceedings (Vol. 707, No. 1, pp. 381-387). American Institute of Physics.

    Params:
    -------
     idx - An ndarray of Hilbert integers.  Must be an integer dtype and
           cannot have fewer bits than num_dims * depth.

     num_dims - The dimensionality of the hypercube. Integer.

     depth - The number of bits for each dimension. Integer.

    Returns:
    --------
     The output is an ndarray of int64 integers with the same shape as idx
     but with an additional dimension of size num_dims.

    Not called in the repository.
    In: idx, int tensor of keys, (n,) or any shape; depth, int, the bits per axis; num_dims, int.
    Out: (*idx.shape, num_dims) int64 tensor, the locations.
    """

    if num_dims * depth > 64:
        raise ValueError("""
      num_dims=%d and num_bits=%d for %d bits total, which can't be encoded
      into a uint64.  Are you sure you need that many points on your Hilbert
      curve?
      """ % (num_dims, depth, num_dims * depth))

    # Handle the case where we got handed a naked integer.
    idx = torch.atleast_1d(idx)  # the keys, at least 1-d

    # Keep around the shape for later.
    orig_shape = idx.shape  # torch.Size
    bitpack_mask = 2 ** torch.arange(0, 8).to(idx.device)  # (8,) int64: 1, 2, ..., 128
    bitpack_mask_rev = bitpack_mask.flip(-1)  # (8,) int64: 128, ..., 1

    # Treat each of the hilberts as a s equence of eight uint8.
    # This treats all of the inputs as uint64 and makes things uniform.
    hh_uint8 = idx.ravel().type(torch.int64).view(torch.uint8).reshape((-1, 8)).flip(-1)  # (n, 8) uint8, MSB first

    # Turn these lists of uints into lists of bits and then truncate to the size
    # we actually need for using Skilling's procedure.
    hh_bits = (  # (n, num_dims * depth) uint8: each key's low bits, most significant first
        hh_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0)
        .byte()
        .flatten(-2, -1)[:, -num_dims * depth :]
    )

    # Take the sequence of bits and Gray-code it.
    gray = binary2gray(hh_bits)  # (n, num_dims * depth) bool

    gray = gray.reshape((-1, depth, num_dims)).swapaxes(1, 2)

    # Iterate backwards through the bits.
    for bit in range(depth - 1, -1, -1):  # bit: int
        # Iterate backwards through the dimensions.
        for dim in range(num_dims - 1, -1, -1):  # dim: int
            # Identify which ones have this bit active.
            mask = gray[:, dim, bit]  # (n,) bool

            # Where this bit is on, invert the 0 dimension for lower bits.
            gray[:, 0, bit + 1 :] = torch.logical_xor(
                gray[:, 0, bit + 1 :], mask[:, None]
            )

            # Where the bit is off, exchange the lower bits with the 0 dimension.
            to_flip = torch.logical_and(  # (n, depth - bit - 1) bool
                torch.logical_not(mask[:, None]),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            gray[:, dim, bit + 1 :] = torch.logical_xor(
                gray[:, dim, bit + 1 :], to_flip
            )
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

    # Pad back out to 64 bits.
    extra_dims = 64 - depth  # int
    padded = torch.nn.functional.pad(gray, (extra_dims, 0), "constant", 0)  # (n, num_dims, 64) bool

    # Now chop these up into blocks of 8.
    locs_chopped = padded.flip(-1).reshape((-1, num_dims, 8, 8))  # (n, num_dims, 8, 8) bool

    # Take those blocks and turn them unto uint8s.
    locs_uint8 = (locs_chopped * bitpack_mask).sum(3).squeeze().type(torch.uint8)  # (n, num_dims, 8) uint8

    # Finally, treat these as uint64s.
    flat_locs = locs_uint8.view(torch.int64)  # (n, num_dims, 1) int64

    # Return them in the expected shape.
    return flat_locs.reshape((*orig_shape, num_dims))


#######################################################################################
# Morton order
#######################################################################################
#
# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------


class KeyLUT:
    """
    The Morton code's lookup tables, per device: the key bits of every 8-bit x, y and z coordinate, and the 3-bit
    x, y and z of every 9-bit key.

    Built once, as the module's `_key_lut`; read by `morton_encode` and `morton_decode`.
    """

    def __init__(self):
        """
        Builds the tables on the CPU.

        Called on import, for `_key_lut`.
        """
        r256 = torch.arange(256, dtype=torch.int64)  # (256,) int64: every 8-bit coordinate
        r512 = torch.arange(512, dtype=torch.int64)  # (512,) int64: every 9-bit key
        zero = torch.zeros(256, dtype=torch.int64)  # (256,) int64
        device = torch.device("cpu")  # torch.device

        self._encode = {  # {torch.device: 3 (256,) int64 tensors}: the key bits of an 8-bit x, y and z
            device: (
                self.xyz2key(r256, zero, zero, 8),
                self.xyz2key(zero, r256, zero, 8),
                self.xyz2key(zero, zero, r256, 8),
            )
        }
        self._decode = {device: self.key2xyz(r512, 9)}  # {torch.device: 3 (512,) int64}: x, y, z per 9-bit key

    def encode_lut(self, device=torch.device("cpu")):
        """
        The encode tables on `device`, copied there from the CPU on first use.

        Called by `morton_encode`.
        In: device, torch.device. Out: three (256,) int64 tensors, the key bits of an 8-bit x, y and z.
        """
        if device not in self._encode:
            cpu = torch.device("cpu")  # torch.device
            self._encode[device] = tuple(e.to(device) for e in self._encode[cpu])
        return self._encode[device]

    def decode_lut(self, device=torch.device("cpu")):
        """
        The decode tables on `device`, copied there from the CPU on first use.

        Called by `morton_decode`.
        In: device, torch.device. Out: three (512,) int64 tensors, the 3-bit x, y and z of each 9-bit key.
        """
        if device not in self._decode:
            cpu = torch.device("cpu")  # torch.device
            self._decode[device] = tuple(e.to(device) for e in self._decode[cpu])
        return self._decode[device]

    def xyz2key(self, x, y, z, depth):
        """
        Interleaves the low `depth` bits of x, y and z into Morton keys, x's bit the highest of each triple.

        Called by `__init__`.
        In: x, y, z, (n,) int64 tensors; depth, int. Out: (n,) int64 tensor.
        """
        key = torch.zeros_like(x)  # (n,) int64
        for i in range(depth):  # i: int, a bit
            mask = 1 << i  # int: bit i
            key = (
                key
                | ((x & mask) << (2 * i + 2))
                | ((y & mask) << (2 * i + 1))
                | ((z & mask) << (2 * i + 0))
            )
        return key

    def key2xyz(self, key, depth):
        """
        Splits Morton keys back into their x, y and z, the inverse of `xyz2key`.

        Called by `__init__`.
        In: key, (n,) int64 tensor; depth, int, the bits per axis to read. Out: x, y, z, (n,) int64 tensors.
        """
        x = torch.zeros_like(key)  # (n,) int64
        y = torch.zeros_like(key)  # (n,) int64
        z = torch.zeros_like(key)  # (n,) int64
        for i in range(depth):  # i: int, a bit
            x = x | ((key & (1 << (3 * i + 2))) >> (2 * i + 2))
            y = y | ((key & (1 << (3 * i + 1))) >> (2 * i + 1))
            z = z | ((key & (1 << (3 * i + 0))) >> (2 * i + 0))
        return x, y, z


_key_lut = KeyLUT()  # the Morton lookup tables, built on import


def morton_encode(
    locs: torch.Tensor,
    depth: int,
):
    r"""Encodes :attr:`x`, :attr:`y`, :attr:`z` coordinates to the shuffled keys
    based on pre-computed look up tables. The speed of this function is much
    faster than the method based on for-loop.

    Args:
      locs: shape(B, 3)
      depth (int): The depth of the shuffled key, and must be smaller than 17 (< 17).

    Called by `lodos.col.sdf.io.reorder_v`, as its `encode` for `encoding="morton"`.
    In: locs (n, 3) int tensor; depth, int. Out: (n,) int64 tensor, the keys.
    """

    x, y, z = torch.unbind(locs, dim=-1)  # (n,) each, locs' dtype
    EX, EY, EZ = _key_lut.encode_lut(x.device)  # (256,) int64 each
    x, y, z = x.long(), y.long(), z.long()

    mask = 255 if depth > 8 else (1 << depth) - 1  # int: the low byte's bits
    key = EX[x & mask] | EY[y & mask] | EZ[z & mask]  # (n,) int64: the low 8 bits' key
    if depth > 8:
        mask = (1 << (depth - 8)) - 1
        key16 = EX[(x >> 8) & mask] | EY[(y >> 8) & mask] | EZ[(z >> 8) & mask]  # (n,) int64: the high bits' key
        key = key16 << 24 | key

    return key


def morton_decode(idx: torch.Tensor, depth: int):
    r"""Decodes the point index to :attr:`x`, :attr:`y`, :attr:`z` coordinates
    and the batch index based on pre-computed look up tables.

    Args:
      idx (torch.Tensor): The shuffled key.
      depth (int): The depth of the shuffled key, and must be smaller than 17 (< 17).

    Not called in the repository.
    In: idx, (n,) int64 tensor of keys; depth, int. Out: (n, 3) int64 tensor, x, y and z.
    """

    DX, DY, DZ = _key_lut.decode_lut(idx.device)  # (512,) int64 each
    x, y, z = torch.zeros_like(idx), torch.zeros_like(idx), torch.zeros_like(idx)  # (n,) int64 each

    idx = idx & ((1 << 48) - 1)  # (n,) int64: the keys' low 48 bits, the batch index dropped

    n = (depth + 2) // 3  # int: the 9-bit groups to read
    for i in range(n):  # i: int, a group
        k = idx >> (i * 9) & 511  # (n,) int64: group i
        x = x | (DX[k] << (i * 3))
        y = y | (DY[k] << (i * 3))
        z = z | (DZ[k] << (i * 3))

    return torch.stack([x, y, z], dim=-1)
